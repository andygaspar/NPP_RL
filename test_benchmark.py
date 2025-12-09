import numpy as np

from Instance.gat_instance import GATInstance
from Solver.genetic_solver import Genetic
from Solver.solver import GlobalSolver
from gat import EGAT

file_name = 'NET/test_1000.pth'

# Carica
agent = EGAT(hidden_channels=64, out_channels=2, lr=0.001, wd=0.0001)
agent.load(file_name, device='cpu')

BASELINE_ITERATIONS = 10000
POPULATION = 128

# N_PATHS = [20, 30, 40, 50]
# N_COMMS = [20, 30, 40, 50]
N_PATHS = [100, 200]
N_COMMS = [100, 200]

N_RUNS = 10

for path in N_PATHS:
    for comm in N_COMMS:
        wins, gaps, exact_gaps = 0, [], []
        for run in range(N_RUNS):
            instance = GATInstance(path, comm, seed=run)

            g = Genetic(instance, pop_size=POPULATION, verbose=False)
            g.run(BASELINE_ITERATIONS)

            g_nn = Genetic(instance, pop_size=POPULATION, verbose=False)
            samples = agent.get_distribution(instance, POPULATION)
            g_nn.run(BASELINE_ITERATIONS, init_population=instance.rescale_prices(samples))
            wins += g.best_val <= g_nn.best_val
            gaps += [g_nn.best_val/g.best_val]

            solver = GlobalSolver(instance, time_limit=g_nn.time, verbose=False)
            solver.solve()
            exact_gaps += [g_nn.best_val / solver.obj]

        print(path, comm, wins/N_RUNS, np.mean(gaps), np.mean(exact_gaps))



