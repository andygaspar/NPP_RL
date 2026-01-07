import time

import numpy as np
import pandas as pd
from NPP.Instance.gat_instance import GATInstance
from NPP.Solver.genetic_solver import Genetic
from GAT.gat import load_agent
from NPP.Solver.solver import GlobalSolver

file_name = 'NPP/NET/test_15_25.pth'

df_exact = pd.read_csv('NPP/Results/exact_results.csv')

agent = load_agent(file_name)
device = agent.device

BASELINE_ITERATIONS = 10000
POPULATION = 128

TIME_LIMIT = 1800


CASES = [(20, 20), (56, 56), (90, 90), (180, 180), (360, 360), (450, 450), (20, 450), (450, 20)]
case_num = dict(zip(CASES, range(len(CASES))))

N_RUNS = 10


columns = ['run', 'commodities', 'paths',
           'obj_exact', 'obj_nn_ga', 'obj_nn_sample', 'obj_nn_mean', 'obj_ga',
           'mip_GAP', 'status',
           'time_exact', 'time_nn_ga','time_nn', 'time_ga', 'case_num']

df = pd.DataFrame(columns=columns)

for case in CASES:
    paths, comm = case
    wins, gaps, exact_gaps = 0, [], []
    for run in range(N_RUNS):
        instance = GATInstance(paths, comm, seed=run)

        ga = Genetic(instance, pop_size=POPULATION, verbose=False)
        ga.run(BASELINE_ITERATIONS)

        ga_nn = Genetic(instance, pop_size=POPULATION, verbose=False)
        net_time = time.time()
        samples = agent.get_distribution(instance, POPULATION)

        _, net_best_sample = instance.eval_sample(samples)
        net_time = time.time() - net_time

        mean_val = instance.eval(agent.get_mean(instance))


        ga_nn.run(BASELINE_ITERATIONS, init_population=instance.rescale_prices(samples))
        wins += ga.best_val <= ga_nn.best_val
        gaps += [ga_nn.best_val / ga.best_val] if ga.best_val > 0 else ([-1] if ga_nn.best_val > 0 else [-2])

        solver = GlobalSolver(instance, time_limit=TIME_LIMIT, verbose=False)
        # solver.solve()
        # exact_gaps += [ga_nn.best_val / solver.obj]
        # df.loc[df.shape[0]] = [run, comm, paths,
        #                        solver.obj, ga_nn.best_val, net_best_sample, mean_val, ga.best_val,
        #                        solver.final_gap, solver.m.status,
        #                        solver.time, ga_nn.time + net_time, net_time, ga.time, case_num]

        solver = df_exact[(df_exact.case_num == case_num[case]) & (df_exact.run == run)].iloc[0]
        exact_gaps += [ga_nn.best_val / solver.obj_exact]
        df.loc[df.shape[0]] = [run, comm, paths,
                               solver.obj_exact, ga_nn.best_val, net_best_sample, mean_val, ga.best_val,
                               solver.mip_GAP, solver.status,
                               solver.time_exact, ga_nn.time + net_time, net_time, ga.time, case_num[case]]

    print(paths, comm, wins / N_RUNS, np.mean(gaps), np.mean(exact_gaps))
    df.to_csv('NPP/Results/test_.csv')


#
# import pandas as pd
#
# df_1 = pd.read_csv('Results/test_450.csv')
# df_2 = pd.read_csv('Results/test.csv')
#
# df = pd.concat([df_1, df_2])
#
#
# df.columns
#
#
# df_exact = df[['run', 'commodities', 'paths', 'obj_exact', 'mip_GAP', 'status', 'time_exact', 'case_num']].copy()
# df_exact.case_num = df_exact.apply(lambda row: case_num[(int(row.commodities), int(row.paths))], axis=1)
#
# df_exact.to_csv('Results/exact_results.csv', index=False)
