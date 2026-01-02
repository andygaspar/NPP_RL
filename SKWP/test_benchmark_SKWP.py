import time

import numpy as np
import pandas as pd
from SKWP.SKWP_graph_instance import SKWPGraph
from SKWP.GA_SKWP import GA_SKWP
from GAT.gat import load_agent
from SKWP.skwp_solver import SKWP

file_name = 'SKWP/NET/test_15_25.pth'
# df_exact = pd.read_csv('NPP/Results/exact_results.csv')

agent = load_agent(file_name)
device = agent.device

BASELINE_ITERATIONS = 10000
POPULATION = 128

TIME_LIMIT = 1800


CASES = [(20, 20), (56, 56), (90, 90), (180, 180), (360, 360), (450, 450), (20, 450), (450, 20)]
case_num = dict(zip(CASES, range(len(CASES))))

N_RUNS = 10


columns = ['run', 'commodities', 'paths',
           'obj_exact', 'obj_nn_ga', 'obj_nn_sample', 'obj_ga',
           'mip_GAP', 'status',
           'time_exact', 'time_nn_ga','time_nn', 'time_ga', 'case_num']

df = pd.DataFrame(columns=columns)

for case in CASES:
    paths, comm = case
    wins, gaps, exact_gaps = 0, [], []
    for run in range(N_RUNS):
        np.random.seed(run)
        instance = SKWPGraph(paths, comm)

        ga = GA_SKWP(instance, pop_size=POPULATION)
        ga.run(BASELINE_ITERATIONS)

        ga_nn = GA_SKWP(instance, pop_size=POPULATION)
        net_time = time.time()
        samples = agent.get_distribution(instance, POPULATION)

        _, net_best_sample = instance.eval_sample(samples)
        net_time = time.time() - net_time

        mean_val = instance.eval(agent.get_mean(instance))

        ga_nn.run(BASELINE_ITERATIONS, init_population=instance.rescale_p(samples))
        wins += ga.best_val <= ga_nn.best_val
        gaps += [ga_nn.best_val / ga.best_val] if ga.best_val > 0 else ([-1] if ga_nn.best_val > 0 else [-2])

        solver = SKWP(instance)
        obj, sol = solver.solve(verbose = False)

        exact_gaps += [ga_nn.best_val / solver.obj]
        df.loc[df.shape[0]] = [run, comm, paths,
                               solver.obj, ga_nn.best_val, net_best_sample, mean_val, ga.best_val,
                               -1, -1,
                               solver.time, ga_nn.time + net_time, net_time, ga.time, case_num]

        # solver = df_exact[(df_exact.case_num == case_num[case]) & (df_exact.run == run)].iloc[0]
        # exact_gaps += [ga_nn.best_val / solver.obj_exact]
        # df.loc[df.shape[0]] = [run, comm, paths,
        #                        solver.obj_exact, ga_nn.best_val, net_best_sample, mean_val, ga.best_val,
        #                        solver.mip_GAP, solver.status,
        #                        solver.time_exact, ga_nn.time + net_time, net_time, ga.time, case_num[case]]

    print(paths, comm, wins / N_RUNS, np.mean(gaps), np.mean(exact_gaps))
    df.to_csv('SKWP/Results/test_.csv')


