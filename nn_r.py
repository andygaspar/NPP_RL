import torch
import torch.nn as nn
import numpy as np
from torch import optim

from Instance.instance import Instance, get_feature_size
from torch.distributions import Categorical, Normal

from Solver.solver import GlobalSolver


class Net(nn.Module):
    def __init__(self, input_size, hidden, n_paths):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_paths * 2)
        self.act = nn.ReLU()
        self.n_paths = n_paths

    def forward(self, x):
        l_p = x[:, : self.n_paths]
        n_p = x[:, self.n_paths: self.n_paths * 2]
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.n_paths, 2)

        mu = torch.clamp(x[:, :, :1].view(-1, self.n_paths), l_p, n_p)
        sigma = torch.clamp(torch.exp(x[:, :, 1:]).view(-1, self.n_paths), 1, 10)
        if torch.isnan(mu).any():
            print("Warning: mu contains NaN, replacing with 0", x[:, :, :1])
            mu = torch.nan_to_num(mu, nan=0.0)

        if torch.isnan(sigma).any():
            print("Warning: sigma contains NaN, replacing with 1.0")
            sigma = torch.nan_to_num(sigma, nan=1.0)
        m = Normal(mu, sigma)

        # sample action, get log probability
        action = m.sample()
        log_action = m.log_prob(action)

        # return action.item(), log_action, mu[:, 0].item(), sigma[:, 0].item()
        # vals = torch.clamp(action, l_p, n_p)
        vals = action
        return vals, log_action, mu, sigma, x[:, :, :1]


class Agent:

    def __init__(self, net: Net, lr, wd):
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    # Funzione per generare un'istanza casuale e vettorizzare l'input
    @staticmethod
    def make_instance_flat(instance: Instance):
        l_p = [p.L_p for p in inst.paths]
        n_p = [p.N_p for p in inst.paths]
        rows = l_p + n_p
        for c in instance.commodities:
            rows += [c.n_users, c.c_od] + list(c.c_p_vector)
        # Creazione di un vettore unico
        X_flat = np.array(rows).flatten()
        return torch.tensor(X_flat, dtype=torch.float32).unsqueeze(0)

    def get_action(self, instance: Instance, eval=False):
        X_flat = self.make_instance_flat(instance)
        if eval:
            with torch.no_grad():
                return self.net(X_flat)
        else:
            return self.net(X_flat)

    def train(self, log_action, reward):
        loss = (-reward * log_action).sum()
        loss.backward()
        self.optimizer.step()





N_COMM = 4
N_PATHS = 4
SEED = 0
HIDEEN = 64

lr = 0.0001
wd = 0.01

torch.manual_seed(SEED)

# Crea una singola istanza



net = Net(get_feature_size(N_PATHS, N_COMM), HIDEEN, N_PATHS)

agent = Agent(net, lr, wd)


inst = Instance(n_paths=N_PATHS, n_commodities=N_COMM, seed=SEED)
solver = GlobalSolver(inst)
l_p = [p.L_p for p in inst.paths]
n_p = [p.N_p for p in inst.paths]
print(l_p, n_p)
solver.solve()
print(solver.obj)
print(solver.solution)

for _ in range(100000):
    inst = Instance(n_paths=N_PATHS, n_commodities=N_COMM, seed=SEED)
    prices, log_prices, mu, sigma, x = agent.get_action(inst)
    res = inst.compute_solution_value_with_tol(prices)
    agent.train(log_prices, res)
    #
    if _ % 100 == 0:
        print(_, "objval", res, prices.cpu().tolist(), mu.cpu().tolist(), sigma.cpu().tolist(), x)
        # print("Output rete (non allenata):", prices, log_prices)
