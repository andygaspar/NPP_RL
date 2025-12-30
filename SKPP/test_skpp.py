import time

import numpy as np

from SKPP.GA_SKPP import GA_SKPP
from SKPP.skpp_instance import SPKK_instance
from SKPP.skpp_solver import SKPP, SKPP_pop, SKPP_greedy_pop

np.random.seed(0)

pb = SPKK_instance(8, 10)

t = time.time()
mpk = SKPP(pb)
obj, sol = mpk.solve(verbose = False)
sol = np.array([sol])
print(sol)
mpk_pop = SKPP_pop(pb,1)
_, val = mpk_pop.solve(sol.copy())
mpk_pop = SKPP_greedy_pop(pb,1)
print(sol)
_, val_greedy = mpk_pop.solve(sol)
print('obj', obj, val, val_greedy, time.time() - t)

## Run the GA
pop_size = 2
generations = 200

method = 'greedy'  # 'exact'

t = time.time()
ga = GA_SKPP(pb, pop_size, method)
ga.run(generations)
print('time', time.time() - t)

method = 'exact'

t = time.time()
ga = GA_SKPP(pb, pop_size, method)
ga.run(generations)
print('time', time.time() - t)
print(ga.best_solution)


