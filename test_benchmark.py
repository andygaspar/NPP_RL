import time

import numpy as np
import pandas as pd
from Instance.gat_instance import GATInstance
from Solver.genetic_solver import Genetic
from Solver.solver import GlobalSolver
from gat import load_agent

file_name = 'NET/test_20_30.pth'

agent = load_agent(file_name)
device = agent.device

BASELINE_ITERATIONS = 10000
POPULATION = 128


CASES = [(20, 20), (56, 56), (90, 90), (180, 180), (360, 360), (720, 720), (20, 360), (360, 20)]

N_RUNS = 10

columns = ['run', 'commodities', 'paths',
           'obj_exact', 'obj_nn', 'obj_ga',
           'mip_GAP', 'status',
           'time_exact', 'time_h', 'time_ga', 'case_num']
case_num = 0

df = pd.DataFrame(columns=columns)

for case in CASES:
    paths, comm = case
    wins, gaps, exact_gaps = 0, [], []
    for run in range(N_RUNS):
        instance = GATInstance(paths, comm, seed=run)

        g = Genetic(instance, pop_size=POPULATION, verbose=False)
        g.run(BASELINE_ITERATIONS)

        g_nn = Genetic(instance, pop_size=POPULATION, verbose=False)
        net_time = time.time()
        samples = agent.get_distribution(instance, POPULATION)
        net_time = time.time() - net_time
        g_nn.run(BASELINE_ITERATIONS, init_population=instance.rescale_prices(samples))
        wins += g.best_val <= g_nn.best_val
        gaps += [g_nn.best_val / g.best_val]

        solver = GlobalSolver(instance, time_limit=max(g_nn.time, 60), verbose=False)
        solver.solve()
        exact_gaps += [g_nn.best_val / solver.obj]
        df.loc[df.shape[0]] = [run, comm, paths,
                               solver.obj, g_nn.best_val, g.best_val,
                               solver.final_gap, solver.m.status,
                               solver.time, g_nn.time + net_time, g.time, case_num]
    case_num += 1
    df.to_csv('results/test.csv')

    print(paths, comm, wins / N_RUNS, np.mean(gaps), np.mean(exact_gaps))
