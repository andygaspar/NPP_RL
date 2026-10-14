import time

import numpy as np
import pandas as pd

from Initialisations.initialisations import init_lhc, init_sobol, init_halton, init_kronecker
from SKWP.SKWP_graph_instance import SKWPGraph
from SKWP.GA_SKWP import GA_SKWP
import sys

from GAT.policy import load_policy
from SKWP.skwp_solver import SKWP


EXTENDED = True
# 'k1': single-follower cases, 'mf': multi-follower cases
VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'k1'
model_name = 'skwp_policy_' + VARIANT

file_name = 'SKWP/NET/' + model_name + '.pth'

agent = load_policy(file_name)
device = agent.device

BASELINE_ITERATIONS = 1000
POPULATION = 128

TIME_LIMIT = 1800
METHOD = 'g'

CASES = [(10, 10), (15, 15), (20, 20), (56, 56), (90, 90), (180, 180), (360, 360), (450, 450), (10, 360), (360, 10), (20, 450), (450, 20), (30, 720), (720, 30)]
SINGLE_COMMODITY = [(20, 1), (56, 1), (90, 1), (180, 1), (360, 1), (450, 1), (720, 1), (900, 1)]
CASES = SINGLE_COMMODITY if VARIANT == 'k1' else CASES

# exact values: read from the saved results instead of solving again
RUN_EXACT = False
EXACT_FILE = 'results_skwp_new.csv'
df_exact = None if RUN_EXACT else pd.read_csv(EXACT_FILE)


SOLVER_CASES = [(10, 10), (15, 15)]

case_num = dict(zip(CASES, range(len(CASES))))
print(case_num)

N_RUNS = 30


columns = ['run', 'K', 'M',
           'obj_exact', 'obj_nn_ga', 'obj_nn_sample', 'obj_nn_mean', 'obj_ga',
           'obj_ga_lhc', 'obj_ga_sobol', 'obj_ga_halton', 'obj_ga_kronecker',
           'mip_GAP', 'status',
           'time_exact', 'time_nn_ga','time_nn', 'time_ga',
           'time_ga_lhc', 'time_ga_sobol', 'time_ga_halton', 'time_ga_kronecker', 'case_num']

df = pd.DataFrame(columns=columns)
# exact_columns = ['run', 'commodities', 'paths',
#            'obj_exact', 'mip_GAP', 'status',
#            'time_exact', 'case_num']
# df_exact = pd.DataFrame(columns=exact_columns)

for case in CASES:
    print(case)
    M, K = case
    wins, gaps, exact_gaps = 0, [], []
    for run in range(N_RUNS):
        np.random.seed(run)
        instance = SKWPGraph(M, K, extended=EXTENDED)

        ga = GA_SKWP(instance, pop_size=POPULATION, method=METHOD)
        ga.run(BASELINE_ITERATIONS)

        ga_lhc = GA_SKWP(instance, pop_size=POPULATION, method=METHOD)
        lhc_sample = init_lhc(instance.L, POPULATION, np.zeros(instance.L), instance.max_w)
        ga_lhc.run(BASELINE_ITERATIONS, init_population=lhc_sample)

        ga_sobol = GA_SKWP(instance, pop_size=POPULATION, method=METHOD)
        sobol_sample = init_sobol(instance.L, POPULATION, np.zeros(instance.L), instance.max_w)
        ga_sobol.run(BASELINE_ITERATIONS, init_population=sobol_sample)

        ga_halton = GA_SKWP(instance, pop_size=POPULATION, method=METHOD)
        halton_sample = init_halton(instance.L, POPULATION, np.zeros(instance.L), instance.max_w)
        ga_halton.run(BASELINE_ITERATIONS, init_population=halton_sample)

        ga_kronecker = GA_SKWP(instance, pop_size=POPULATION, method=METHOD)
        kronecker_sample = init_kronecker(instance.L, POPULATION, np.zeros(instance.L), instance.max_w)
        ga_kronecker.run(BASELINE_ITERATIONS, init_population=kronecker_sample)



        ga_nn = GA_SKWP(instance, pop_size=POPULATION, method=METHOD)
        net_time = time.time()
        samples = agent.get_distribution(instance, POPULATION)

        _, net_best_sample = instance.eval_sample(samples, method=METHOD)
        net_time = time.time() - net_time

        mean_val = instance.eval(agent.get_mean(instance), method=METHOD)

        ga_nn.run(BASELINE_ITERATIONS, init_population=instance.rescale_w(samples))
        wins += ga.best_val <= ga_nn.best_val
        gaps += [ga_nn.best_val / ga.best_val] if ga.best_val > 0 else ([-1] if ga_nn.best_val > 0 else [-2])

        ex = None if RUN_EXACT else df_exact[(df_exact.M == M) & (df_exact.K == K) & (df_exact.run == run) & (df_exact.obj_exact > 0)]
        if case in SOLVER_CASES and not RUN_EXACT and len(ex):
            solver_time, solver_obj, solver_final_gap, solver_status = (
                ex.iloc[0].time_exact, ex.iloc[0].obj_exact, ex.iloc[0].mip_GAP, ex.iloc[0].status)
        elif case in SOLVER_CASES and RUN_EXACT:
            solver = SKWP(instance)
            obj, sol = solver.solve(verbose=False, time_limit=TIME_LIMIT)
            solver_time, solver_obj, solver_final_gap, solver_status = (
                solver.time, solver.obj, solver.final_gap, solver.model.status)
        else:
            solver_time, solver_obj, solver_final_gap, solver_status = [-1] * 4

        print(ga.best_val, ga_lhc.best_val, ga_sobol.best_val, ga_halton.best_val, ga_kronecker.best_val, ga_nn.best_val, solver_obj)

        df.loc[df.shape[0]] = [run, K, M,
                               solver_obj, ga_nn.best_val, net_best_sample, mean_val, ga.best_val,
                               ga_lhc.best_val, ga_sobol.best_val, ga_halton.best_val, ga_kronecker.best_val,
                               solver_final_gap, solver_status,
                               solver_time, ga_nn.time + net_time, net_time, ga.time,
                               ga_lhc.time, ga_sobol.time, ga_halton.time, ga_kronecker.time, case_num[case]]

        #exact_gaps += [ga_nn.best_val / solver_obj]
        #print(ga_nn.best_val / ga.best_val, ga_nn.best_val / solver_obj)
        #df.loc[df.shape[0]] = [run, K, M,
        #                       solver_obj, ga_nn.best_val, net_best_sample, mean_val, ga.best_val,
        #                       solver_final_gap, solver_status,
        #                       solver_time, ga_nn.time + net_time, net_time, ga.time, case_num[case]]
        # df_exact.loc[df_exact.shape[0]] = [run, K, M,
        #                        solver_obj,
        #                        solver_final_gap, solver_status,
        #                        solver_time, case_num[case]]

    # print(M, K, wins / N_RUNS, np.mean(gaps), np.mean(exact_gaps))
    # df_exact.to_csv('SKWP/Results/' + 'exact_results.csv', index=False)

    df.to_csv('SKWP/Results/results_skwp_' + VARIANT + '.csv', index=False)


