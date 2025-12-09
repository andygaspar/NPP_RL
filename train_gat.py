import random
import numpy as np
import torch

from Instance.gat_instance import create_batch, GATInstance
from Solver.genetic_solver import Genetic
from gat import EGAT

N_COMM = range(5, 10)
N_PATHS = range(5, 10)
SEED = 1
HIDDEN = 64
N_SAMPLES = 20
ITERATIONS = 10000
EPISODE_PER_BATCH = 64

BASELINE_ITERATIONS = 1000
POPULATION = N_SAMPLES

lr = 0.001
wd = 0.0001
agent = EGAT(16, 2, n_samples=N_SAMPLES, lr=lr, wd=wd)

for iteration in range(ITERATIONS):
    instances = [GATInstance(np.random.choice(N_PATHS), np.random.choice(N_COMM), seed=0) for i in range(EPISODE_PER_BATCH)]

    batch = create_batch(instances)

    samples, log_probs = agent(batch)

    rewards, baselines, log_prices = [], [], []
    wins, gaps = 0, []

    for i, inst in enumerate(instances):
        mask = (batch.batch == i) * (batch.x[:, -1] == 1) # get only paths (i.e. x[:, -1 == 1) of the batch
        samples_graph = samples[:, mask]
        log_prices.append(log_probs[:, mask].flatten())

        g = Genetic(inst, pop_size=POPULATION, verbose=False)
        g.run(BASELINE_ITERATIONS)
        baselines += [g.best_val for _ in range(inst.n_paths * N_SAMPLES)]

        g_nn = Genetic(inst, pop_size=POPULATION, verbose=False)
        g_nn.run(1, init_population=inst.rescale_prices(samples_graph))
        # rewards.append([g_nn.best_val for _ in range(inst.n_paths * N_SAMPLES)])
        rewards += np.repeat(g_nn.final_vals, inst.n_paths).tolist()

        wins += g.best_val < g_nn.best_val
        gaps += [g_nn.best_val/g.best_val]

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    baseline_tensor = torch.tensor(baselines, dtype=torch.float32)
    log_prices = torch.cat(log_prices)

    loss = agent.train_policy(log_prices, reward_tensor, baseline_tensor)

    if iteration % 50 == 0:
        max_agent = reward_tensor.max().item()
        print(f"E{iteration:4d} | " 
              f"WinRate: {wins / EPISODE_PER_BATCH :.3f} | "
              f"avg gap: {np.mean(gaps) :.3f} | "
              f"loss: {loss :.3f}")