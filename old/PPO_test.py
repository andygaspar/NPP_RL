import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torchrl.modules import TruncatedNormal

from Instance import instance
from Instance.instance import Instance, get_feature_size
from torch.distributions import Categorical, Normal, Beta

from Solver.genetic_solver import Genetic
from Solver.solver import GlobalSolver


class Net(nn.Module):
    def __init__(self, input_size, hidden, n_paths, n_samples):
        super().__init__()
        # Shared backbone
        self.shared_fc1 = nn.Linear(input_size, hidden)
        self.shared_fc2 = nn.Linear(hidden, hidden)

        # Actor head
        self.actor_fc = nn.Linear(hidden, hidden)
        self.actor_mean = nn.Linear(hidden, n_paths)
        self.actor_std = nn.Linear(hidden, n_paths)

        # Critic head
        self.critic_fc = nn.Linear(hidden, hidden)
        self.critic_out = nn.Linear(hidden, 1)

        self.act = nn.LeakyReLU()
        self.n_paths = n_paths
        self.n_samples = n_samples

        # Initialize weights
        for layer in [self.actor_mean, self.actor_std, self.critic_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # Shared features
        x = self.act(self.shared_fc1(x))
        x = self.act(self.shared_fc2(x))

        # Actor head
        actor_x = self.act(self.actor_fc(x))
        mu_raw = self.actor_mean(actor_x)
        sigma_raw = self.actor_std(actor_x)

        mu = 0.5 + 0.5 * torch.tanh(mu_raw)
        sigma = 0.05 + 0.2 * torch.sigmoid(sigma_raw)  # [0.05, 0.25]

        # Critic head
        critic_x = self.act(self.critic_fc(x))
        value = self.critic_out(critic_x)

        return mu, sigma, value

    def get_action(self, x):
        l_p = x[:, :self.n_paths]
        n_p = x[:, self.n_paths:self.n_paths * 2]

        mu, sigma, value = self.forward(x)

        # Create truncated normal distribution
        dist = TruncatedNormal(mu, sigma, low=0, high=1)
        sample = dist.sample((self.n_samples,))

        # Calculate log probability
        log_prob = dist.log_prob(sample)

        # Transform action
        action = l_p + (n_p - l_p) * sample

        return action, log_prob, value, mu, sigma


class PPOBuffer:
    def __init__(self, buffer_size, n_paths, n_samples):
        self.buffer_size = buffer_size
        self.n_paths = n_paths
        self.n_samples = n_samples
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []

    def store(self, state, action, log_prob, value, reward):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)

    def is_full(self):
        return len(self.states) >= self.buffer_size

    def get(self):
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)

        return states, actions, log_probs, values, rewards


class PPOAgent:
    def __init__(self, net: Net, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, ppo_epochs=4, batch_size=32, wd=0.0):
        self.net = net
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(
            [p for n, p in net.named_parameters() if 'actor' in n or 'shared' in n],
            lr=lr, weight_decay=wd
        )
        self.critic_optimizer = optim.Adam(
            [p for n, p in net.named_parameters() if 'critic' in n or 'shared' in n],
            lr=lr, weight_decay=wd
        )

        # Create buffer
        self.buffer = PPOBuffer(buffer_size=batch_size * 10,
                                n_paths=net.n_paths,
                                n_samples=net.n_samples)

    @staticmethod
    def make_instance_flat(instance: Instance):
        rows = np.concatenate([instance.lower_bounds, instance.upper_bounds])
        for c in instance.commodities:
            rows = np.concatenate([rows, [c.n_users, c.c_od], c.c_p_vector])
        return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)

    def get_action(self, instance: Instance, eval=False):
        X_flat = self.make_instance_flat(instance)

        if eval:
            with torch.no_grad():
                mu, sigma, value = self.net(X_flat)
                # During evaluation, use mean instead of sampling
                dist = TruncatedNormal(mu, sigma, low=0, high=1)
                sample = dist.sample((self.net.n_samples,))
                l_p = X_flat[:, :self.net.n_paths]
                n_p = X_flat[:, self.net.n_paths:self.net.n_paths * 2]
                action = l_p + (n_p - l_p) * sample
                return action, None, value
        else:
            action, log_prob, value, mu, sigma = self.net.get_action(X_flat)
            return action, log_prob, value

    def store_transition(self, state, action, log_prob, value, reward):
        self.buffer.store(state, action, log_prob, value, reward)

    def compute_gae(self, values, rewards, next_value):
        """Compute Generalized Advantage Estimation for single-step episodes"""
        # Since we have single-step episodes (each instance is one step),
        # we can compute advantage directly
        advantages = rewards - values.squeeze()

        # Calculate returns
        returns = advantages + values.squeeze()

        return advantages, returns

    def train(self, next_value=None):
        if not self.buffer.is_full():
            return None, None

        # Get data from buffer
        states, actions, old_log_probs, values, rewards = self.buffer.get()

        # Compute GAE and returns (for single-step episodes)
        advantages, returns = self.compute_gae(values, rewards, next_value)

        # Normalize advantages
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        actor_losses = []
        critic_losses = []

        for _ in range(self.ppo_epochs):
            # Create batches
            batch_size = min(self.batch_size, len(states))
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get new action distribution
                l_p = batch_states[:, :self.net.n_paths]
                n_p = batch_states[:, self.net.n_paths:self.net.n_paths * 2]

                mu, sigma, _ = self.net(batch_states)

                # Create distribution and get new log probs
                dist = TruncatedNormal(mu, sigma, low=0, high=1)
                new_log_probs = dist.log_prob(batch_actions)

                # Calculate ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * batch_advantages

                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()

                # Get value predictions
                _, _, batch_values = self.net(batch_states)
                batch_values = batch_values.squeeze()

                # Critic loss (value loss)
                critic_loss = nn.functional.mse_loss(batch_values, batch_returns)

                # Optimize actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                # Optimize critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        # Clear buffer after training
        self.buffer.clear()

        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0

        return avg_actor_loss, avg_critic_loss


N_COMM = 128
N_PATHS = 128
SEED = 1
HIDDEN = 64
N_SAMPLES = 64
EPISODES = 3000

GA_ITERATIONS = 1000
POPULATION = N_SAMPLES
OFF_SIZE = int(POPULATION / 2)
MUTATION_RATE = 0.02
recombination_size = int(N_PATHS / 2)

lr = 0.001
wd = 0.0001

score_vs_random = 0

torch.manual_seed(SEED)

net = Net(get_feature_size(N_PATHS, N_COMM), HIDDEN, N_PATHS, N_SAMPLES)
agent = PPOAgent(net, lr=lr, gamma=0.99, clip_epsilon=0.2,
                 ppo_epochs=4, batch_size=32, wd=wd)

# inst = Instance(n_paths=N_PATHS, n_commodities=N_COMM, seed=SEED)
# solver = GlobalSolver(inst, time_limit=3600, verbose=True)
# solver.solve()
# optimal_value = solver.obj
# print(f"Optimal solution: {optimal_value}")
won = 0

for episode in range(EPISODES):
    inst = Instance(n_paths=N_PATHS, n_commodities=N_COMM, seed=episode)
    X_flat = agent.make_instance_flat(inst)

    prices, log_prices, values = agent.get_action(inst, eval=False)

    # Calculate rewards
    rewards = []
    rand_rewards = []

    for i, price_sample in enumerate(prices):
        # Network's action
        val = inst.compute_solution_value_with_tol(price_sample[0].detach().numpy())
        rewards.append(val)

        # Random baseline
        rand_val = inst.compute_solution_value_with_tol(
            np.random.uniform(inst.lower_bounds, inst.upper_bounds)
        )
        rand_rewards.append(rand_val)
        if val > rand_val:
            score_vs_random += 1

    # Store transitions in buffer
    for i in range(N_SAMPLES):
        agent.store_transition(X_flat, prices[i:i + 1], log_prices[i:i + 1],
                               values[i:i + 1], rewards[i])

    # Convert to tensor
    reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    max_baseline_reward = max(rand_rewards)

    g = Genetic(inst, pop_size=POPULATION, offs_size=OFF_SIZE,
                mutation_rate=MUTATION_RATE, recombination_size=recombination_size,
                verbose=False, seed=SEED)
    g.run(GA_ITERATIONS)

    # Train with PPO
    actor_loss, critic_loss = agent.train()

    if episode % 50 == 0:
        max_agent = reward_tensor.max().item()

        # Format loss values safely (they might be None initially)
        actor_loss_str = f"{actor_loss:.6f}" if actor_loss is not None else "N/A"
        critic_loss_str = f"{critic_loss:.6f}" if critic_loss is not None else "N/A"
        print(f"E{episode:4d} | "
              f"WinRate: {score_vs_random / ((episode + 1) * N_SAMPLES):.3f} | "
              f"Rand: {max_baseline_reward:.3f} | "
              f"Agent: {max_agent:.3f} | "
              # f"Actor Loss: {actor_loss:.6f} | "
              # f"Critic Loss: {critic_loss:.6f}"
              )

        g = Genetic(inst, pop_size=POPULATION, offs_size=OFF_SIZE,
                    mutation_rate=MUTATION_RATE, recombination_size=recombination_size,
                    verbose=False, seed=SEED)
        g.run(GA_ITERATIONS)

        g_nn = Genetic(inst, pop_size=POPULATION, offs_size=OFF_SIZE,
                       mutation_rate=MUTATION_RATE, recombination_size=recombination_size,
                       verbose=False, seed=SEED)

        prices_init = np.ascontiguousarray(prices.squeeze(1).detach().numpy())
        g_nn.run(GA_ITERATIONS, init_population=prices_init)
        won += (g.best_val <= g_nn.best_val)
        print(f"GA vs GA-NN Win Rate: {won / (episode // 50 + 1):.3f}, "
              f"GA Best: {g.best_val:.3f}, GA-NN Best: {g_nn.best_val:.3f}")