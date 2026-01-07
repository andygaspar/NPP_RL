import copy
import ctypes
import multiprocessing
import os

import numpy as np
from networkx.algorithms.efficiency_measures import efficiency
from numpy.ctypeslib import ndpointer


class Result(ctypes.Structure):
    _fields_ = [
        ('solution', ctypes.POINTER(ctypes.c_bool)),
    ]


class KnapCpp:
    def __init__(self, pb, pop_size):
        self.lib = ctypes.CDLL('KP_Solver/knap_bridge.so')
        self.pop_size = pop_size
        self.p = np.array([pb.p for _ in range(self.pop_size)], dtype=float)
        self.c = np.stack((pb.c,) * pop_size)
        self.w = np.array([pb.w for _ in range(self.pop_size)])
        self.K = pb.K
        self.L = pb.L
        self.M = pb.M

        self.numProcs = os.cpu_count()

        self.lib.knapsack_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.knapsack_.restype = ctypes.POINTER(Result)

        self.lib.free_result.argtypes = [ctypes.POINTER(Result)]

    def solve(self, p, w, c):
        p = np.ascontiguousarray(p, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        c = np.ascontiguousarray(c, dtype=np.float64)
        return self.lib.knapsack_(p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  ctypes.c_int(self.M), ctypes.c_int(self.K), ctypes.c_int(self.pop_size),
                                  ctypes.c_int(self.numProcs))

    def solve_skpp(self, p_new):
        p_new = np.stack((p_new,) * self.K, axis=1)
        p = self.p.copy()
        p[:, :, :self.L] = p_new

        res = self.solve(p, self.w, self.c)
        sol = np.ctypeslib.as_array(res.contents.solution, shape=(self.pop_size, self.K, self.M))
        vals = ((self.p - p) * sol)[:, :, :self.L].sum(axis=-1).sum(axis=-1)
        self.lib.free_result(res)
        return vals, vals.max()

    def solve_skwp(self, w_new):
        w_new = np.stack((w_new,) * self.K, axis=1)
        w = self.w.copy()
        w[:, :, :self.L] = w_new

        res = self.solve(self.p, w, self.c)
        sol = np.ctypeslib.as_array(res.contents.solution, shape=(self.pop_size, self.K, self.M))

        vals = (w * sol)[:, :, :self.L].sum(axis=-1).sum(axis=-1)
        self.lib.free_result(res)
        return vals, vals.max()