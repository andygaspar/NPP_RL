import time

import numpy as np

from SKPP.GA_SKPP import GA_SKPP
from SKPP.skpp_instance import SPKK_instance
from SKPP.skpp_solver import SKPP, SKPP_pop

np.random.seed(0)

pb = SPKK_instance(15, 10)

t = time.time()
mpk = SKPP(pb)
obj, sol = mpk.solve(verbose = True)
sol = np.array([sol])
mpk_pop = SKPP_pop(pb,1)
val = mpk_pop.solve(sol)
print('obj', obj, val, time.time() - t)

## Run the GA
pop_size = 128
generations = 20

method = 'greedy'  # 'exact'
#
t = time.time()
ga = GA_SKPP(pb, pop_size, method)
ga.run(generations, verbose = True)
print('time', time.time() - t)

method = 'exact'

t = time.time()
ga = GA_SKPP(pb, pop_size, method)
ga.run(generations, verbose = True)
print('time', time.time() - t)
print(ga.best_solution)


