import time

import numpy as np

from KP_Solver.knap_cpp import KnapCpp
from SKPP.skpp_instance import SPKK_instance
from SKPP.skpp_solver import SKPP_pop

np.random.seed(0)

pop_size = 64

pb = SPKK_instance(40, 90)

sol = np.random.uniform(0, pb.p[0, : pb.L], size=(pop_size, pb.L))


tt = time.time()
mpk_pop = SKPP_pop(pb,pop_size, preset=False)
_, val_greedy = mpk_pop.solve(sol)
print('gurobi', time.time() - tt)

tt = time.time()
bb = KnapCpp(pb, pop_size)
val_cc = bb.solve_cpp(sol)
print('cc', time.time() - tt)





pass

# t = time.time()
# mpk = SKPP(pb)
# obj, sol = mpk.solve(verbose = False)
# sol = np.array([sol])
# print(sol)
# mpk_pop = SKPP_pop(pb,1)
# _, val = mpk_pop.solve(sol.copy())
# mpk_pop = SKPP_greedy_pop(pb,1)
# print(sol)
# _, val_greedy = mpk_pop.solve(sol)
# print('obj', obj, val, val_greedy, time.time() - t)
#
# ## Run the GA
# pop_size = 2
# generations = 200
#
# method = 'greedy'  # 'exact'
#
# t = time.time()
# ga = GA_SKPP(pb, pop_size, method)
# ga.run(generations)
# print('time', time.time() - t)
#
# method = 'exact'
#
# t = time.time()
# ga = GA_SKPP(pb, pop_size, method)
# ga.run(generations)
# print('time', time.time() - t)
# print(ga.best_solution)



