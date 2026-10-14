import sys
import time

import numpy as np
import torch

from SKWP.SKWP_graph_instance import create_SKWP_batch, SKWPGraph
from SKWP.validation_skwp import load_validation, SIZES
from GAT.policy import TransformerPolicy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print('Experiments running on', device)


METHOD = 'g'

# 'k1': single follower, 'mf': multi-follower (sizes in SKWP/validation_skwp.py)
VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'k1'
M, K = SIZES[VARIANT]

file_name = 'SKWP/NET/skwp_policy_' + VARIANT + '.pth'
print(file_name)

SEED = 1
# one distinct seed per training instance, away from the test seeds (0..N_RUNS-1) and the validation seeds
TRAIN_SEED_MIN = 10_000
torch.manual_seed(SEED)

HIDDEN = 64

N_SAMPLES = 64
ITERATIONS = 3000
EPISODE_PER_BATCH = 32

VALIDATION_EVERY = 50
VALIDATION_SAMPLES = 128

lr = 0.0003
wd = 0.00001
agent = TransformerPolicy(hidden_channels=HIDDEN, lr=lr, wd=wd, device=device)

# fixed validation set, normalised by an offline long-GA reference
validation = load_validation(VARIANT)


def validate():
    """Mean over the validation set of best sampled initial value / reference."""
    agent.eval()
    score = np.mean([inst.eval_sample(agent.get_distribution(inst, VALIDATION_SAMPLES), method=METHOD)[1] / ref
                     for inst, ref in validation])
    agent.train()
    return score


t = time.time()

for iteration in range(ITERATIONS):
    instances = []
    for i in range(EPISODE_PER_BATCH):
        np.random.seed(TRAIN_SEED_MIN + iteration * EPISODE_PER_BATCH + i)
        instances.append(SKWPGraph(np.random.choice(M), np.random.choice(K)))

    batch = create_SKWP_batch(instances, device=device)

    samples, log_probs, mu = agent(batch, N_SAMPLES)

    log_prices, rewards, groups = [], [], []
    mu_std = []

    for i, inst in enumerate(instances):
        mask = (batch.batch == i) * (batch.x[:, -1] == 1) # get only items controlled by the leader (i.e. x[:, -1] == 1) of the batch
        vals, _ = inst.eval_sample(samples[:, mask], method=METHOD)
        log_prices.append(log_probs[:, mask].sum(dim=1))  # joint log-prob of each sample
        rewards += list(vals)
        groups += [i] * N_SAMPLES  # instance index, for the per-instance advantage
        mu_std.append(mu[mask].std().item())

    reward_tensor = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
    group_tensor = torch.tensor(groups, device=device)
    log_prices = torch.cat(log_prices)

    loss = agent.train_policy(log_prices, reward_tensor, group_tensor)

    if (iteration + 1) % VALIDATION_EVERY == 0:
        score = validate()
        if score > agent.best_val:
            agent.best_val = score
            agent.save(file_name)
        print(f"E{iteration + 1:5d} | "
              f"Train time {time.time() - t:.1f} || "
              f"val best/ref: {score:.4f} (best {agent.best_val:.4f}) | "
              f"loss: {loss:.3f} | "
              f"std mu across items: {np.mean(mu_std):.3f}", flush=True)
