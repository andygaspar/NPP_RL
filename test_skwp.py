import time

import numpy as np

from KP_Solver.knap_cpp import KnapCpp
from SKWP.GA_SKWP import GA_SKWP
from SKWP.skwp_instance import SKWP_instance
from SKWP.skwp_solver import SKWP

pop_size = 128


for i in range(2, 3):
    np.random.seed(i)
    pb = SKWP_instance(3, 2)

    ga = GA_SKWP(pb, pop_size, method='g')
    ga.run(1000, verbose=False)


    # sol, _ = pb.compute_obj(ga.best_solution.reshape(1, -1))
    # ga.best_solution

    solver_3 = SKWP(pb)
    obj_3, w_3, sol_item = solver_3.solve_all(verbose=False, time_limit=3600)


    # obj_4, w_4, sol_item_ = solver_3.solve_all(verbose=False, time_limit=3600)

    # obj_3, w_3 = solver_3.solve_k3(verbose=False, time_limit=3600)

    # print(sol_item, 'kkk')
    sol, _ = pb.compute_obj(w_3.reshape(1, -1))
    # print(obj_4, obj_3)

    # print()
    ga_sol, _ = pb.compute_obj(ga.best_solution.reshape(1, -1))
    print(i, ga.best_val, ga_sol, solver_3.obj,  sol) #, ga.best_val/solver_3.obj)
    # print(ga.time, solver_3.time)
# solver = SKWP(pb)
# obj, w_exact = solver.solve(verbose=True, time_limit=3600)


# print(w_exact)
# solver_2 = SKWP(pb)
# obj_2, w_2, x = solver_2.solve_k2(verbose=False, time_limit=3600)
# print(obj, obj_2, ga.best_val)


# ga_val, sol_ga = pb.compute_obj(ga.best_solution.reshape(1, -1))
# print(pb.compute_obj(w_exact.reshape(1, -1))[0], pb.compute_obj(w_3.reshape(1, -1))[0], ga_val)
# print(obj, obj_3, ga.best_val)