import time

import numpy as np

from SKPP.GA import  GA_MBK
from SKPP.skpp_solver import MKP, MKP_pop
from SKPP.skpp_instance import generate_instance

np.random.seed(0)

pb = generate_instance(10, 10)

t = time.time()
mpk = MKP(pb)
obj, sol = mpk.solve(verbose = False)
print(sol)
sol = np.array([sol])
mpk_pop = MKP_pop(pb,1)
val = mpk_pop.solve(sol)
print('obj', obj, val, time.time() - t)

## Run the GA
pop_size = 128
generations = 200

method = 'greedy'  # 'exact'

t = time.time()
ga = GA_MBK(pb, pop_size, generations, method)
ga.solve()
print('time', time.time() - t)

method = 'exact'

t = time.time()
ga = GA_MBK(pb, pop_size, generations, method)
ga.solve()
print('time', time.time() - t)
print(ga.best_solution)


