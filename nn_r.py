import torch
import torch.nn as nn
import numpy as np
from torch import optim

from Instance.instance import Instance, get_feature_size
from torch.distributions import Categorical, Normal, Beta
from torchrl.modules import TruncatedNormal

from Solver.solver import GlobalSolver




class Net(nn.Module):
    def __init__(self, input_size, hidden, n_paths):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_paths * 2)
        self.act = nn.LeakyReLU()
        self.soft_plus = nn.Softplus()
        self.n_paths = n_paths
        self.to(self.device)

    def forward(self, x):
        l_p = x[:, : self.n_paths]
        n_p = x[:, self.n_paths: self.n_paths * 2]
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.n_paths, 2)
        # Stable parameter computation with constraints
        alpha_raw = x[:, :, 0]
        beta_raw = x[:, :, 1]

        # Clip raw values before softplus to prevent explosion
        alpha_raw = torch.clip(alpha_raw, 0, 1)   # [-5, 5]
        beta_raw = torch.clip(beta_raw, 0.1, 2)
        # Apply softplus with minimum value
        # alpha = self.soft_plus(alpha_raw).view(-1, self.n_paths) + 1e-6
        # beta = self.soft_plus(beta_raw).view(-1, self.n_paths) + 1e-6
        # alpha = self.soft_plus(x[:, :, :1]).view(-1, self.n_paths)
        # beta = self.soft_plus(x[:, :, 1:]).view(-1, self.n_paths)

        alpha = alpha_raw
        beta = beta_raw

        mm = TruncatedNormal(alpha_raw, torch.exp(beta_raw), high=1, low=0)
        # mm = Beta(alpha, beta)
        sample = mm.sample((16,))
        print(sample[0])
        action = l_p + (n_p - l_p) * sample

        log_action = mm.log_prob(sample)

        return action, log_action, x[:, :, :1], alpha, beta


class Agent:

    def __init__(self, net: Net, lr, wd):
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        self.best_val = 1

    def update_best_val(self, obj):
        if obj > self.best_val:
            self.best_val = obj


    # Funzione per generare un'istanza casuale e vettorizzare l'input
    @staticmethod
    def make_instance_flat(instance: Instance, device):
        l_p = [p.L_p for p in inst.paths]
        n_p = [p.N_p for p in inst.paths]
        rows = l_p + n_p
        for c in instance.commodities:
            rows += [c.n_users, c.c_od] + list(c.c_p_vector)
        # Creazione di un vettore unico
        X_flat = np.array(rows).flatten()
        return torch.tensor(X_flat, dtype=torch.float32, device=device).unsqueeze(0)

    def get_action(self, instance: Instance, device, eval=False):
        X_flat = self.make_instance_flat(instance, device)
        if eval:
            with torch.no_grad():
                X_flat.requires_grad = True
                return self.net(X_flat)
        else:
            return self.net(X_flat)

    def train(self, log_action, reward, alpha, beta):
        loss = -(reward/self.best_val * log_action).mean() /100 #+ 2*(torch.abs(alpha/50).sum() + torch.abs(beta/50).sum())/(alpha.shape[-1]*2)
        loss.backward()
        self.optimizer.step()
        return loss.item()


N_COMM = 4
N_PATHS = 4
SEED = 1
HIDEEN = 64

lr = 0.000001
wd = 0.0001

torch.manual_seed(SEED)

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    prices, log_prices, x, alpha, beta = agent.get_action(inst, device)
    res = []
    for p in prices:
        val = inst.compute_solution_value_with_tol(p)
        res.append([val for _ in range(N_PATHS)])
        agent.update_best_val(val)
    res = torch.tensor(res, dtype=torch.float32, device=device).unsqueeze(1)
    loss = agent.train(log_prices, res, alpha, beta)
    #
    if _ % 1000 == 0:
        print(_, "objval", res[0][0][0].item(), loss, prices[0].cpu().tolist(), alpha, beta)
        # print("Output rete (non allenata):", prices, log_prices)


# {$p_{0}$: 11.197899321371724, $p_{1}$: 8.171189523729565, $p_{2}$: 13.455847349152648, $p_{3}$: 4.499325391483689}
