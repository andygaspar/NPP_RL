"""
Fixed validation set for model selection, normalised by an offline reference (no exact solver: it does not scale).
Reference of an instance = best of 5 long GA runs (pop 512, 5000 iterations, greedy follower).
Run `python -m skwp.validation k1` (single follower) or `python -m skwp.validation mf` (multi-follower)
once to compute results/validation_skwp_<variant>.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

from skwp.ga import GA_SKWP
from skwp.graph_instance import SKWPGraph

VAL_SEED_MIN = 900_000  # disjoint from the test seeds (0..29) and the training seeds (10_000 + ...)
N_VAL = 64
# (items, followers) sizes of each variant; must match the training script
SIZES = {'k1': (range(20, 50), [1]), 'mf': (range(10, 40), range(10, 40))}
REF_POP, REF_ITERATIONS, REF_RUNS = 512, 5000, 5
METHOD = 'g'


def ref_file(variant):
    return f'results/validation_skwp_{variant}.csv'


def make_instance(M, K, seed):
    np.random.seed(seed)
    return SKWPGraph(M, K)


def validation_specs(variant):
    items, followers = SIZES[variant]
    specs = []
    for k in range(N_VAL):
        np.random.seed(VAL_SEED_MIN + k)
        specs.append((int(np.random.choice(items)), int(np.random.choice(followers)), VAL_SEED_MIN + k))
    return specs


def make_reference(variant):
    rows = []
    for M, K, seed in validation_specs(variant):
        inst = make_instance(M, K, seed)
        vals = []
        for _ in range(REF_RUNS):
            g = GA_SKWP(inst, pop_size=REF_POP, method=METHOD)
            g.run(REF_ITERATIONS)
            vals.append(g.best_val)
        rows.append(dict(M=M, K=K, seed=seed, ref=max(vals)))
        print(rows[-1], flush=True)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(rows).to_csv(ref_file(variant), index=False)


def load_validation(variant):
    """List of (instance, reference value)."""
    if not os.path.exists(ref_file(variant)):
        raise FileNotFoundError(f'{ref_file(variant)} missing: run `python -m skwp.validation {variant}` first')
    df = pd.read_csv(ref_file(variant))
    return [(make_instance(int(r.M), int(r.K), int(r.seed)), r.ref) for r in df.itertuples()]


if __name__ == '__main__':
    make_reference(sys.argv[1])
