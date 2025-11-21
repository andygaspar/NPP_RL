import torch
import torch.nn as nn
import numpy as np
from torch import optim

from Instance.instance import Instance, get_feature_size
from torch.distributions import Categorical, Normal, Beta

from Solver.solver import GlobalSolver

import neptune

#run = neptune.init_run(
#    project="federicocamerota/npp-rl",
#    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNDM4YTIyMS0zM2M0LTQ5YjItYjBhZi00NDNhZmFjYWZmYTMifQ==",
#)

GASPA_MAGIC_NUMBER = 100


class Net(nn.Module):
    def __init__(self, input_size, hidden, n_paths):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)

        self.best_fc1 = nn.Linear(n_paths, hidden)
        self.best_fc2 = nn.Linear(hidden, hidden)
        self.best_fc3 = nn.Linear(hidden, hidden)

        self.act_fc1 = nn.Linear(2 * hidden, 2*hidden)
        self.act_fc2 = nn.Linear(2*hidden, 2*hidden)
        self.act_fc3 = nn.Linear(2*hidden, 2*n_paths)

        self.act = nn.LeakyReLU()
        self.soft_plus = nn.Softplus()
        self.n_paths = n_paths

    def forward(self, x, best_x):
        l_p = x[:, : self.n_paths]
        n_p = x[:, self.n_paths: self.n_paths * 2]
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x) + x)
        x = self.fc3(x) + x

        best_x = self.act(self.best_fc1(best_x))
        best_x = self.act(self.best_fc2(best_x) + best_x)
        best_x = self.best_fc3(best_x) + best_x

        act_x = torch.cat([x, best_x], dim=-1)
        act_x = self.act(self.act_fc1(act_x))
        act_x = self.act(self.act_fc2(act_x) + act_x)
        act_x = self.act_fc3(act_x)

        act_x = act_x.view(-1, self.n_paths, 2)
        # Stable parameter computation with constraints
        alpha_raw = act_x[:, :, 0]
        beta_raw = act_x[:, :, 1]

        # Clip raw values before softplus to prevent explosion
        alpha_raw = torch.clip(alpha_raw, -50, 50)   # [-5, 5]
        beta_raw = torch.clip(beta_raw, -50, 50)
        # Apply softplus with minimum value
        alpha = self.soft_plus(alpha_raw).view(-1, self.n_paths) + 1e-6
        beta = self.soft_plus(beta_raw).view(-1, self.n_paths) + 1e-6
        # alpha = self.soft_plus(x[:, :, :1]).view(-1, self.n_paths)
        # beta = self.soft_plus(x[:, :, 1:]).view(-1, self.n_paths)
        mm = Beta(alpha, beta)
        sample = mm.rsample((GASPA_MAGIC_NUMBER,))
        action = l_p + (n_p - l_p) * sample

        log_action = mm.log_prob(sample)

        return action, log_action, act_x[:, :, :1], alpha, beta


class Agent:

    def __init__(self, net: Net, lr, wd):
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        self.best_val = 1
        self.best_x = torch.zeros(1, N_PATHS)

    def update_best_val(self, obj, p):
        if obj > self.best_val:
            self.best_val = obj
            self.best_x = p


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

    def get_action(self, instance: Instance, best_x, eval=False):
        X_flat = self.make_instance_flat(instance)
        if eval:
            with torch.no_grad():
                X_flat.requires_grad = True
                return self.net(X_flat, best_x)
        else:
            return self.net(X_flat, best_x)

    def train(self, log_action, reward, alpha, beta):
        loss = -(reward/self.best_val * log_action).mean() + 2*(torch.abs(alpha/50).sum() + torch.abs(beta/50).sum())/(alpha.shape[-1]*2)
        loss.backward()
        self.optimizer.step()
        return loss.item()


N_COMM = 4
N_PATHS = 4
SEED = 1
HIDEEN = 64

lr = 0.001
wd = 0.0001

torch.manual_seed(SEED)


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

    best_x = torch.zeros(1, N_PATHS)
    prices, log_prices, x, alpha, beta = agent.get_action(inst, best_x)
    res = []
    for p in prices:
        val = inst.compute_solution_value_with_tol(p)
        res.append([val for _ in range(N_PATHS)])
        agent.update_best_val(val, p)
    res = torch.tensor(res, dtype=torch.float32).unsqueeze(1)
    loss = agent.train(log_prices, res, alpha, beta)
    #
    if _ % 1000 == 0:
        print(_, "objval", res[0][0], loss, prices[0].cpu().tolist(), alpha, beta)
        # print("Output rete (non allenata):", prices, log_prices)


# {$p_{0}$: 11.197899321371724, $p_{1}$: 8.171189523729565, $p_{2}$: 13.455847349152648, $p_{3}$: 4.499325391483689}
