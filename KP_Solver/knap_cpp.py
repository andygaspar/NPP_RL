import copy
import ctypes
import multiprocessing
import os
import time

import numpy as np
import torch
from networkx.algorithms.efficiency_measures import efficiency
from numpy.ctypeslib import ndpointer


class Result(ctypes.Structure):
    _fields_ = [
        ('solution', ctypes.POINTER(ctypes.c_bool)),
    ]


class KnapCpp:
    def __init__(self, pb, batch_size):
        self.lib = ctypes.CDLL('KP_Solver/knap_bridge.so')
        self.batch_size = batch_size

        adjusted_p = pb.p + pb.adjustment_p

        self.p = np.ascontiguousarray(np.array([adjusted_p for _ in range(self.batch_size)], dtype=float))
        self.c = np.ascontiguousarray(np.stack((pb.c,) * batch_size))
        self.w = np.ascontiguousarray(np.array([pb.w for _ in range(self.batch_size)]))
        # p = np.ascontiguousarray(p, dtype=np.float64)
        # w = np.ascontiguousarray(w, dtype=np.float64)
        # c = np.ascontiguousarray(c, dtype=np.float64)
        self.K = pb.K
        self.L = pb.L
        self.M = pb.M

        self.numProcs = os.cpu_count()

        self.lib.knapsack_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.knapsack_.restype = ctypes.POINTER(Result)

        self.lib.free_result.argtypes = [ctypes.POINTER(Result)]

        self.lib.solve_greedy_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    def solve(self, p, w, c):
        p = np.ascontiguousarray(p, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        c = np.ascontiguousarray(c, dtype=np.float64)
        return self.lib.knapsack_(p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  ctypes.c_int(self.M), ctypes.c_int(self.K), ctypes.c_int(self.batch_size),
                                  ctypes.c_int(self.numProcs))

    def solve_skwp_greedy_cpp(self, w, L):
        # p = np.ascontiguousarray(p, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        # c = np.ascontiguousarray(c, dtype=np.float64)
        v = np.ascontiguousarray(np.zeros(self.batch_size), dtype=np.float64)
        self.lib.solve_greedy_(self.p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                               w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                               self.c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                              v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                               ctypes.c_int(self.M), ctypes.c_int(self.K), ctypes.c_int(self.batch_size), ctypes.c_int(L),
                               ctypes.c_int(self.numProcs))
        return v

    def solve_skpp(self, p_new):
        p_new = np.stack((p_new,) * self.K, axis=1)
        p = self.p.copy()
        p[:, :, :self.L] = p_new

        res = self.solve(p, self.w, self.c)
        sol = np.ctypeslib.as_array(res.contents.solution, shape=(self.batch_size, self.K, self.M))
        vals = ((self.p - p) * sol)[:, :, :self.L].sum(axis=-1).sum(axis=-1)
        self.lib.free_result(res)
        return vals, vals.max()

    def solve_skwp(self, w_new):
        w_new = np.stack((w_new,) * self.K, axis=1)
        w = self.w.copy()
        w[:, :, :self.L] = w_new

        res = self.solve(self.p, w, self.c)
        sol = np.ctypeslib.as_array(res.contents.solution, shape=(self.batch_size, self.K, self.M))

        vals = (w * sol)[:, :, :self.L].sum(axis=-1).sum(axis=-1)
        self.lib.free_result(res)
        return vals, vals.max()

    def solve_skwp_greedy(self, w_new_):

        w_new = np.stack((w_new_,) * self.K, axis=1)
        w = self.w.copy()
        w[:, :, :self.L] = w_new
        # t = time.time()
        # efficiency = self.p / (w + 1e-12)
        # indexes = np.argsort(-efficiency, axis=-1)
        # w_sorted = np.take_along_axis(w, indexes, axis=-1)
        # cs = np.cumsum(w_sorted, axis=-1)
        # sol = (cs < self.c[:, :, np.newaxis])
        # np.put_along_axis(sol, indexes, sol, axis=-1)
        # vals__ = (w * sol)[:, :, :self.L].sum(axis=-1).sum(axis=-1)
        # print(time.time() - t, 'np')
        # t = time.time()
        vals = self.solve_skwp_greedy_cpp(w, self.L)
        # #
        # print(time.time() - t, 'torch')
        return vals, vals.max()

    def get_solution_x(self, w_sol):
        p = self.p.copy()
        w = self.w.copy()
        w[:, :, :self.L] = w_sol
        efficiency = p / w
        indexes = np.argsort(-efficiency, axis=-1)
        w_sorted = np.take_along_axis(w, indexes, axis=-1)
        cs = np.cumsum(w_sorted, axis=-1)
        sol = (cs < self.c[:, :,  np.newaxis])
        np.put_along_axis(sol, indexes, sol, axis=-1)
        return sol