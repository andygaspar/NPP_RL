import time

import numpy as np
import pandas as pd
from SKWP.SKWP_graph_instance import SKWPGraph
from SKWP.GA_SKWP import GA_SKWP
from GAT.gat import load_agent
from SKWP.skwp_solver import SKWP

file_name = 'SKWP/NET/test_skwp_15_30.pth'
# df_exact = pd.read_csv('NPP/Results/exact_results.csv')

agent = load_agent(file_name)
device = agent.device

BASELINE_ITERATIONS = 1000
POPULATION = 128

TIME_LIMIT = 1800


CASES = [(10, 10), (15, 15), (20, 20), (30, 30), (60, 60), (90, 90), (10, 90), (90, 10)]
SOLVER_CASES = [(10, 10), (15, 15)]
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

        ga_nn.run(BASELINE_ITERATIONS, init_population=instance.rescale_w(samples))
        wins += ga.best_val <= ga_nn.best_val
        gaps += [ga_nn.best_val / ga.best_val] if ga.best_val > 0 else ([-1] if ga_nn.best_val > 0 else [-2])


        if case in SOLVER_CASES:
            solver = SKWP(instance)
            obj, sol = solver.solve(verbose=False)
            solver_time, solver_obj, solver_final_gap, solver_status = (
                solver.time, solver.obj, solver.final_gap, solver.model.status)
        else:
            solver_time, solver_obj, solver_final_gap, solver_status = [-1] * 4

        exact_gaps += [ga_nn.best_val / solver_obj]
        df.loc[df.shape[0]] = [run, comm, paths,
                               solver_obj, ga_nn.best_val, net_best_sample, mean_val, ga.best_val,
                               solver_final_gap, solver_status,
                               solver_time, ga_nn.time + net_time, net_time, ga.time, case_num[case]]

    print(paths, comm, wins / N_RUNS, np.mean(gaps), np.mean(exact_gaps))
    df.to_csv('SKWP/Results/test_.csv')


