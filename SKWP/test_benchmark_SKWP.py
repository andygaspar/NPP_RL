import time

import numpy as np
import pandas as pd
from SKWP.SKWP_graph_instance import SKWPGraph
from SKWP.GA_SKWP import GA_SKWP
from GAT.gat import load_agent
from SKWP.skwp_solver import SKWP

EXTENDED = True
model_name = 'test_skwp_20_50_2026-02-27 22:48:24'

file_name = 'SKWP/NET/' + model_name + '.pth'

agent = load_agent(file_name)
device = agent.device

BASELINE_ITERATIONS = 1000
POPULATION = 128

TIME_LIMIT = 1800
METHOD = 'g'

CASES = [(10, 10), (15, 15), (20, 20), (56, 56), (90, 90), (180, 180), (360, 360), (450, 450), (10, 360), (360, 10), (20, 450), (450, 20), (30, 720), (720, 30)]
SINGLE_COMMODITY = [(20, 1), (56, 1), (90, 1), (180, 1), (360, 1), (450, 1), (720, 1), (900, 1)]
CASES = SINGLE_COMMODITY + CASES


SOLVER_CASES = CASES

case_num = dict(zip(CASES, range(len(CASES))))
print(case_num)

N_RUNS = 30


columns = ['run', 'commodities', 'paths',
           'obj_exact', 'obj_nn_ga', 'obj_nn_sample', 'obj_nn_mean', 'obj_ga',
           'mip_GAP', 'status',
           'time_exact', 'time_nn_ga','time_nn', 'time_ga', 'case_num']

df = pd.DataFrame(columns=columns)
exact_columns = ['run', 'commodities', 'paths',
           'obj_exact', 'mip_GAP', 'status',
           'time_exact', 'case_num']
df_exact = pd.DataFrame(columns=exact_columns)

for case in CASES:
    print(case)
    paths, comm = case
    wins, gaps, exact_gaps = 0, [], []
    for run in range(N_RUNS):
        np.random.seed(run)
        instance = SKWPGraph(paths, comm, extended=EXTENDED)

        #ga = GA_SKWP(instance, pop_size=POPULATION, method=METHOD)
        #ga.run(BASELINE_ITERATIONS)

        #ga_nn = GA_SKWP(instance, pop_size=POPULATION, method=METHOD)
        #net_time = time.time()
        #samples = agent.get_distribution(instance, POPULATION)

        #_, net_best_sample = instance.eval_sample(samples, method=METHOD)
        #net_time = time.time() - net_time

        #mean_val = instance.eval(agent.get_mean(instance), method=METHOD)

        #ga_nn.run(BASELINE_ITERATIONS, init_population=instance.rescale_w(samples))
        #wins += ga.best_val <= ga_nn.best_val
        #gaps += [ga_nn.best_val / ga.best_val] if ga.best_val > 0 else ([-1] if ga_nn.best_val > 0 else [-2])

        if case in SOLVER_CASES:
            solver = SKWP(instance)
            obj, sol = solver.solve(verbose=False, time_limit=TIME_LIMIT)
            solver_time, solver_obj, solver_final_gap, solver_status = (
                solver.time, solver.obj, solver.final_gap, solver.model.status)
        else:
            solver_time, solver_obj, solver_final_gap, solver_status = [-1] * 4

        #exact_gaps += [ga_nn.best_val / solver_obj]
        #print(ga_nn.best_val / ga.best_val, ga_nn.best_val / solver_obj)
        #df.loc[df.shape[0]] = [run, comm, paths,
        #                       solver_obj, ga_nn.best_val, net_best_sample, mean_val, ga.best_val,
        #                       solver_final_gap, solver_status,
        #                       solver_time, ga_nn.time + net_time, net_time, ga.time, case_num[case]]
        df_exact.loc[df_exact.shape[0]] = [run, comm, paths,
                               solver_obj,
                               solver_final_gap, solver_status,
                               solver_time, case_num[case]]

    print(paths, comm, wins / N_RUNS, np.mean(gaps), np.mean(exact_gaps))
    #df.to_csv('SKWP/Results/' + model_name + '.csv')
    df_exact.to_csv('SKWP/Results/' + 'exact_results.csv', index=False)


