import time

import numpy as np
import pandas as pd
# from sympy.codegen import Print

from common.initialisations import init_lhc, init_sobol, init_halton, init_kronecker
from pp.graph_instance import GTNInstance
from pp.genetic_solver import Genetic
from common.policy import load_policy
from pp.exact_solver import PPSolver
import os

# os.system('chmod +x install.sh')
# os.system('./install.sh')

# time.sleep(1)

file_name = 'models/pp_policy.pth'

# exact values: read from the saved results (Gurobi, 1800 s time limit) instead of solving again
RUN_EXACT = False
EXACT_FILE = 'results/pp_exact.csv'
df_exact = None if RUN_EXACT else pd.read_csv(EXACT_FILE)


agent = load_policy(file_name)
device = agent.device

BASELINE_ITERATIONS = 10000
POPULATION = 128

TIME_LIMIT = 1800


CASES = [(20, 20), (56, 56), (90, 90), (180, 180), (360, 360), (450, 450), (20, 450), (450, 20)]
case_num = dict(zip(CASES, range(len(CASES))))

N_RUNS = 30

columns = ['run', 'commodities', 'paths',
           'obj_exact', 'obj_nn_ga', 'obj_nn_sample', 'obj_nn_mean', 'obj_ga',
           'obj_ga_lhc', 'obj_ga_sobol', 'obj_ga_halton', 'obj_ga_kronecker',
           'mip_GAP', 'status',
           'time_exact', 'time_nn_ga','time_nn', 'time_ga',
           'time_ga_lhc', 'time_ga_sobol', 'time_ga_halton', 'time_ga_kronecker', 'case_num']

df = pd.DataFrame(columns=columns)

for case in CASES:
    paths, comm = case
    wins, gaps, exact_gaps = 0, [], []
    print('case:', case)
    for run in range(N_RUNS):
        instance = GTNInstance(paths, comm, seed=run)

        ga = Genetic(instance, pop_size=POPULATION, verbose=False)
        ga.run(BASELINE_ITERATIONS)

        ga_lhc = Genetic(instance, pop_size=POPULATION, verbose=False)
        lhc_sample = init_lhc(instance.n_paths, POPULATION, instance.lower_bounds, instance.upper_bounds)
        ga_lhc.run(BASELINE_ITERATIONS, init_population=lhc_sample)


        ga_sobol = Genetic(instance, pop_size=POPULATION, verbose=False)
        sobol_sample = init_sobol(instance.n_paths, POPULATION, instance.lower_bounds, instance.upper_bounds)
        ga_sobol.run(BASELINE_ITERATIONS, init_population=sobol_sample)

        ga_halton = Genetic(instance, pop_size=POPULATION, verbose=False)
        halton_sample = init_halton(instance.n_paths, POPULATION, instance.lower_bounds, instance.upper_bounds)
        ga_halton.run(BASELINE_ITERATIONS, init_population=halton_sample)

        ga_kronecker = Genetic(instance, pop_size=POPULATION, verbose=False)
        kronecker_sample = init_kronecker(instance.n_paths, POPULATION, instance.lower_bounds, instance.upper_bounds)
        ga_kronecker.run(BASELINE_ITERATIONS, init_population=kronecker_sample)



        ga_nn = Genetic(instance, pop_size=POPULATION, verbose=False)
        net_time = time.time()
        samples = agent.get_distribution(instance, POPULATION)

        _, net_best_sample = instance.eval_sample(samples)
        net_time = time.time() - net_time

        mean_val = instance.eval(agent.get_mean(instance))


        ga_nn.run(BASELINE_ITERATIONS, init_population=instance.rescale_prices(samples))
        wins += ga.best_val <= ga_nn.best_val
        gaps += [ga_nn.best_val / ga.best_val] if ga.best_val > 0 else ([-1] if ga_nn.best_val > 0 else [-2])
        if RUN_EXACT:
            solver = PPSolver(instance, time_limit=TIME_LIMIT, verbose=False)
            solver.solve()
            solver_obj, solver_gap, solver_status, solver_time = solver.obj, solver.final_gap, solver.m.status, solver.time
        else:
            ex = df_exact[(df_exact.paths == paths) & (df_exact.commodities == comm) & (df_exact.run == run)]
            solver_obj, solver_gap, solver_status, solver_time = ((ex.iloc[0].obj_exact, ex.iloc[0].mip_GAP, ex.iloc[0].status, ex.iloc[0].time_exact)
                                                                  if len(ex) else (np.nan,) * 4)

        exact_gaps += [ga_nn.best_val / solver_obj]
        df.loc[df.shape[0]] = [run, comm, paths,
                               solver_obj, ga_nn.best_val, net_best_sample, mean_val, ga.best_val,
                               ga_lhc.best_val, ga_sobol.best_val, ga_halton.best_val, ga_kronecker.best_val,
                               solver_gap, solver_status,
                               solver_time, ga_nn.time + net_time, net_time, ga.time,
                               ga_lhc.time, ga_sobol.time, ga_halton.time, ga_kronecker.time, case_num[case]]

        # solver = df_exact[(df_exact.case_num == case_num[case]) & (df_exact.run == run)].iloc[0]
        # exact_gaps += [ga_nn.best_val / solver.obj]
        # df.loc[df.shape[0]] = [run, comm, paths,
        #                        solver.obj, ga_nn.best_val, net_best_sample, mean_val, ga.best_val,
        #                        solver.mip_GAP, solver.status,
        #                        solver.time_exact, ga_nn.time + net_time, net_time, ga.time, case_num[case]]

        print(ga.best_val, ga_lhc.best_val, ga_sobol.best_val, ga_halton.best_val, ga_kronecker.best_val, ga_nn.best_val, solver_obj)

    print(paths, comm, wins / N_RUNS, np.mean(gaps), np.mean(exact_gaps))
    df.to_csv('results/pp_results.csv')

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
