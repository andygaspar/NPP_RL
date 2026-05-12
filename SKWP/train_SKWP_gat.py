import time

import numpy as np
import torch

from GAT.gat_2 import EGAT2
from SKWP.GA_SKWP import GA_SKWP
from SKWP.SKWP_graph_instance import create_SKWP_batch, SKWPGraph
from GAT.gat import EGAT


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print('Experiments running on', device)


METHOD = 'g'

MIN_SIZE, MAX_SIZE = 20, 60

SAVE = True
file_name = 'SKWP/NET/test_skwp_' + str(MIN_SIZE) + '_' + str(MAX_SIZE) + '__.pth'

M = range(MIN_SIZE, MAX_SIZE)
K = range(MIN_SIZE, MAX_SIZE)
SEED = 1

HIDDEN = 64

N_SAMPLES = 2
ITERATIONS = 500
EPISODE_PER_BATCH = 512

BASELINE_ITERATIONS = 200
POPULATION = N_SAMPLES

lr = 0.001
wd = 0.0001
STD_MAX = 0.1
agent = EGAT(7, 6, HIDDEN, 2, std_max=STD_MAX, lr=lr, wd=wd, device=device)

BEST_GAP = 0
t = time.time()


for iteration in range(ITERATIONS):
    # np.random.seed(SEED)
    instances = [SKWPGraph(np.random.choice(M), 1) for _ in range(EPISODE_PER_BATCH)]

    batch = create_SKWP_batch(instances, device=device)

    samples, log_probs, _ = agent(batch, N_SAMPLES)
    rewards, baselines, log_prices, randoms = [], [], [], []
    wins, gaps = 0, []
    rand_wins, rand_gaps, rand_ga_gaps = 0, [], []

    for i, inst in enumerate(instances):
        # if i % 5 == 0:
        #     print(iteration, i, inst)
        mask = (batch.batch == i) * (batch.x[:, -1] == 1) # get only p controlled by the leader (i.e. x[:, -1 == 1) of the batch
        inst_sample_tensor = samples[:, mask]
        log_prices.append(log_probs[:, mask].flatten())

        g = GA_SKWP(inst, pop_size=POPULATION, method=METHOD)
        g.run(BASELINE_ITERATIONS)
        baselines += [g.best_val for _ in range(inst.L * N_SAMPLES)]

        # rewards.append([g_nn.best_val for _ in range(inst.L * N_SAMPLES)])
        vals, agent_best_val = inst.eval_sample(inst_sample_tensor, method=METHOD)
        rewards += np.repeat(vals, inst.L).tolist()

        random_best_val = inst.random_baseline(N_SAMPLES, method=METHOD)
        randoms += [random_best_val for _ in range(inst.L * N_SAMPLES)]

        wins += g.best_val < agent_best_val
        gaps += [agent_best_val/g.best_val if g.best_val > 0 else 0]
        rand_wins += random_best_val < agent_best_val
        rand_gaps += [agent_best_val/random_best_val if random_best_val > 0 else 0]
        rand_ga_gaps += [g.best_val/random_best_val if g.best_val > 0 else 0]

    if iteration > 0 and BEST_GAP < np.mean(gaps):
        agent.best_gap = np.mean(gaps)
        if SAVE:
            agent.save(file_name, )
        BEST_GAP = agent.best_gap
        print(f"saved check point. Avg gap : {agent.best_gap: .3f}")

    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    baseline_tensor = torch.tensor(baselines, dtype=torch.float32, device=device)
    # baseline_tensor = torch.tensor(randoms, dtype=torch.float32, device=device)

    log_prices = torch.cat(log_prices)

    loss = agent.train_policy(log_prices, reward_tensor, baseline_tensor)

    if iteration % 1 == 0:
        max_agent = reward_tensor.max().item()
        print(f"E{iteration:4d} | " 
              f"Train time {time.time() - t:.1f} || " 
              f"WinRate: {wins / EPISODE_PER_BATCH :.3f} | "
              f"avg gap: {np.mean(gaps) :.3f} | "
              f"loss: {loss :.3f}  ||  "
              f"rand WinRate: {rand_wins / EPISODE_PER_BATCH :.3f} | "
              f"rand avg gap: {np.mean(rand_gaps) :.3f} | "
              f"rand avg ga rand gap: {np.mean(rand_ga_gaps) :.3f} | "
              )

agent.save('SKK/NET/test.pth')
