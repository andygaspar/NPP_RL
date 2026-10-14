"""
Fixed validation set for model selection, normalised by an offline reference (no exact solver: it does not scale).
Reference of an instance = best of 5 long GA runs (pop 512, 50k iterations): 3 with the default initialisation,
2 with all tolls around 0.70 of their range.
Run `python -m NPP.validation_npp` once to (re)compute NPP/Results/validation_npp.csv.
"""
import os

import numpy as np
import pandas as pd
import torch

from NPP.Instance.gat_instance import GATInstance
from NPP.Solver.genetic_solver import Genetic

VAL_SEED_MIN = 900_000  # disjoint from the test seeds (0..29) and the training seeds (10_000 + ...)
N_VAL = 64
SIZES = range(20, 90)
REF_FILE = 'NPP/Results/validation_npp.csv'
REF_POP, REF_ITERATIONS, REF_RUNS = 512, 50_000, 5


def validation_specs():
    specs = []
    for k in range(N_VAL):
        np.random.seed(VAL_SEED_MIN + k)
        specs.append((int(np.random.choice(SIZES)), int(np.random.choice(SIZES)), VAL_SEED_MIN + k))
    return specs


def reference_value(inst):
    vals = []
    for r in range(REF_RUNS):
        init = None if r < 3 else inst.rescale_prices(
            torch.clamp(0.70 + 0.05 * torch.randn(REF_POP, inst.n_paths), 0, 1))
        g = Genetic(inst, pop_size=REF_POP, verbose=False)
        g.run(REF_ITERATIONS, init_population=init)
        vals.append(g.best_val)
    return max(vals)


def make_reference():
    rows = []
    for n_paths, n_commodities, seed in validation_specs():
        ref = reference_value(GATInstance(n_paths, n_commodities, seed=seed))
        rows.append(dict(paths=n_paths, commodities=n_commodities, seed=seed, ref=ref))
        print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(REF_FILE), exist_ok=True)
    pd.DataFrame(rows).to_csv(REF_FILE, index=False)


def load_validation():
    """List of (instance, reference value)."""
    if not os.path.exists(REF_FILE):
        raise FileNotFoundError(f'{REF_FILE} missing: run `python -m NPP.validation_npp` first')
    df = pd.read_csv(REF_FILE)
    return [(GATInstance(int(r.paths), int(r.commodities), seed=int(r.seed)), r.ref) for r in df.itertuples()]


if __name__ == '__main__':
    make_reference()
