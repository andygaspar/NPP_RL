import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torchrl.modules import TruncatedNormal

from Instance import instance
from Instance.instance import Instance, get_feature_size
from torch.distributions import Categorical, Normal, Beta

from Solver.solver import GlobalSolver


class Net(nn.Module):
    def __init__(self, input_size, hidden, n_paths, n_samples):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.final = nn.Linear(hidden, n_paths * 2)
        self.act = nn.LeakyReLU()
        self.soft_plus = nn.Softplus()
        self.n_paths = n_paths
        self.n_samples = n_samples

        # Better initialization
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.constant_(self.final.bias, 0.0)

    def forward(self, x):
        l_p = x[:, : self.n_paths]
        n_p = x[:, self.n_paths: self.n_paths * 2]
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.final(x)
        x = x.view(-1, self.n_paths, 2)

        alpha_raw = x[:, :, 0]
        beta_raw = x[:, :, 1]

        # FIXED: Proper alpha range [0.1, 0.9]
        alpha = 0.5 + 0.5 * torch.tanh(alpha_raw)  # [0.1, 0.9]

        # Better beta range for exploration
        beta = 0.05 + 0.2 * torch.sigmoid(beta_raw)  # [0.05, 0.25]

        mm = TruncatedNormal(alpha, beta, low=0, high=1)
        sample = mm.sample((self.n_samples,))
        action = l_p + (n_p - l_p) * sample
        log_action = mm.log_prob(sample)

        return action, log_action, alpha, beta


class Agent:
    def __init__(self, net: Net, lr, wd):
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)


    @staticmethod
    def make_instance_flat(instance: Instance):
        # FIXED: Use np.concatenate, not np.concat
        rows = np.concatenate([instance.lower_bounds, instance.upper_bounds])
        for c in instance.commodities:
            rows = np.concatenate([rows, [c.n_users, c.c_od], c.c_p_vector])
        return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)

    def get_action(self, instance: Instance, eval=False):
        X_flat = self.make_instance_flat(instance)
        if eval:
            with torch.no_grad():
                return self.net(X_flat)
        else:
            return self.net(X_flat)

    def train(self, log_action, reward, baseline):

        advantage = reward - baseline

        # Normalize advantage for stability
        if advantage.std() > 0:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        loss = -(advantage * log_action).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()


N_COMM = 10
N_PATHS = 10
SEED = 1
HIDDEN = 64
N_SAMPLES = 100
EPISODES = 3000

lr = 0.001
wd = 0.0001

score_vs_random = 0

torch.manual_seed(SEED)

net = Net(get_feature_size(N_PATHS, N_COMM), HIDDEN, N_PATHS, N_SAMPLES)
agent = Agent(net, lr, wd)

inst = Instance(n_paths=N_PATHS, n_commodities=N_COMM, seed=SEED)
solver = GlobalSolver(inst)
solver.solve()
optimal_value = solver.obj
print(f"Optimal solution: {optimal_value}")

for episode in range(EPISODES):
    prices, log_prices, alpha, beta = agent.get_action(inst)

    # Calculate rewards
    rewards = []
    rand_rewards = []
    best_episode_val = 0

    for i, price_sample in enumerate(prices):
        # Network's action
        val = inst.compute_solution_value_with_tol(price_sample[0].detach().numpy())
        rewards.append(val)
        if val > best_episode_val:
            best_episode_val = val

        # Random baseline
        rand_val = inst.compute_solution_value_with_tol(
            np.random.uniform(inst.lower_bounds, inst.upper_bounds)
        )
        rand_rewards.append(rand_val)
        if val > rand_val:
            score_vs_random += 1

    # Convert to tensor
    reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    max_baseline_reward = max(rand_rewards)


    loss = agent.train(log_prices, reward_tensor, max_baseline_reward)

    if episode % 50 == 0:
        max_idx = np.argmax(rewards)
        rand_max = max(rand_rewards) if rand_rewards else 0
        print(f"E{episode:4d} | "
              f"WinRate: {score_vs_random / ((episode + 1) * N_SAMPLES):.3f} | "
              f"Curr: {best_episode_val:.3f} | "
              f"Rand: {rand_max:.3f} | "
              f"Loss: {loss:.6f} | ")