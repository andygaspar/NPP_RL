import itertools
import time

import gurobipy as gb
from gurobipy import GRB
import numpy as np
from scipy._lib.cobyqa import problem

from SKWP.skwp_instance import SKWP_instance


class SKWP:

    def __init__(self, problem: SKWP_instance):

        self.model = gb.Model("SKWP")

        self.inst = problem
        self.p = problem.p
        self.w = problem.w
        self.c = problem.c
        self.obj = None
        self.time = None
        self.final_gap = None

        self.x = None

    def solve(self, verbose=False, time_limit=None):
        if not verbose:
            self.model.setParam('OutputFlag', 0)
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit)

        tt = time.time()
        w = self.model.addMVar(self.inst.L)
        t = self.model.addMVar((self.inst.K, self.inst.L))
        d_L = self.model.addMVar((self.inst.K, self.inst.L))  # deficiency
        d = self.w / self.p
        x = self.model.addMVar((self.inst.K, self.inst.M), vtype=GRB.BINARY)
        W = self.c

        self.model.addConstr(t.sum(axis=1) + (self.w[:, self.inst.L:] * x[:, self.inst.L:]).sum(axis=1) <= self.c, name='t ')
        self.model.addConstr(t >= w - (1 - x[:, :self.inst.L]) * W.max(), name='cap ')
        self.model.addConstr(t <= w, name='t < w ')

        self.model.addConstr(t <= x[:, :self.inst.L] * np.repeat(W.reshape(self.inst.K, 1), self.inst.L, axis=1), name='t < xW ')
        self.model.addConstr(d_L == w / self.p[:, :self.inst.L], name='d_L ')

        max_d = self.c / self.p.min(axis=1)
        for i in range(self.inst.L):
            for j in range(self.inst.L):
                self.model.addConstr(d_L[:, i] - (1 - x[:, i]) * max_d <= d_L[:, j] + x[:, j] * max_d,
                                     name='d_Li < d_Lj ' + str(i) + ' ' + str(j))

        for i in range(self.inst.L):
            for j in range(self.inst.L, self.inst.M):
                self.model.addConstr(d_L[:, i] - (1 - x[:, i]) * max_d <= d[:, j] + x[:, j] * max_d,
                                     name='d_Li < dj ' + str(i) + ' ' + str(j))

        self.model.setObjective(t.sum(), gb.GRB.MAXIMIZE)
        self.model.optimize()
        self.time = time.time() - tt

        if self.model.Status == GRB.INFEASIBLE:
            self.model.computeIIS()
            for c in self.model.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')
        if self.model.Status == GRB.UNBOUNDED:
            print('unbounded')
        self.obj = self.model.objVal

        self.final_gap = self.model.MIPGap

        return self.model.objVal,  w.x



#
# class SKWP_greedy:
#
#     def __init__(self, problem: SKWP_instance):
#
#         self.model = gb.Model("SKWP")
#
#         self.inst = problem
#         self.p = problem.p
#         self.w = problem.w
#         self.c = problem.c
#         self.obj = None
#         self.time = None
#         self.final_gap = None
#
#         self.x = self.model.addMVar((self.inst.K, self.inst.M), vtype=GRB.BINARY, name='x')
#
#
#     def solve(self, verbose=False, init_solution=None, x_solution=None):
#         if not verbose:
#             self.model.setParam('OutputFlag', 0)
#         tt = time.time()
#
#         w = self.model.addMVar(self.inst.L, name='w')
#         z = self.model.addMVar((self.inst.K, self.inst.M, self.inst.M), vtype=GRB.BINARY, name='z')
#         if init_solution is not None:
#             self.model.addConstr(w == init_solution, name='init_solution')
#             self.model.addConstr(self.x == x_solution, name='x_init_solution')
#             pass
#         t = self.model.addMVar((self.inst.K, self.inst.L), name='t')
#         # self.model.addConstr(z == 1 - self.x)
#         for k in range(self.inst.K):
#
#             self.model.addConstr(t[k].sum() + (self.w[k, self.inst.L:] * self.x[k, self.inst.L:]).sum()
#                                  <= self.c[k], name='cap' + str(k))
#
#             # self.model.addConstr(z[k, 0, 1] == 0)
#             for i in range(self.inst.M):
#                 for j in range(self.inst.M):
#                     self.model.addConstr(z[k, i, j] >= self.x[k, i] - self.x[k, j], name='z1' + str(k) + ' ' + str(i) + ' ' + str(j))
#                     self.model.addConstr(z[k, i, j] <= 1 - self.x[k, j], name='z2' + str(k) + ' ' + str(i) + ' ' + str(j))
#
#             for i in range(self.inst.L):
#                 self.model.addConstr(t[k, i] <= self.x[k, i] * self.inst.max_w[i], name='t < x ' + str(k) + ' ' + str(i))
#                 self.model.addConstr(w[i] - t[k, i] <= (1 - self.x[k, i]) * self.inst.max_w[i], name='w - t >  ' + str(k) + ' ' + str(i))
#                 self.model.addConstr(t[k, i] <= w[i], name='t > w ' + str(k) + ' ' + str(i))
#
#                 for j in range(self.inst.L):
#                     self.model.addConstr(t[k, i] / self.p[k, i] <=
#                                          w[j] / self.p[k, j] + (1 - z[k, i, j]) * self.inst.max_e_inv[k],
#                                          name='eff t < t ' + str(k) + ' ' + str(i) + ' ' + str(j))
#
#                 for j in range(self.inst.L, self.inst.M):
#                     self.model.addConstr(t[k, i] / self.p[k, i] <=
#                                          self.w[k, j] / self.p[k, j] + (1 - z[k, i, j]) * self.inst.max_e_inv[k],
#                                          name='eff t < x ' + str(k) + ' ' + str(i) + ' ' + str(j))
#
#             for i in range(self.inst.L, self.inst.M):
#                 for j in range(self.inst.L):
#                     self.model.addConstr(self.w[k, i] * self.x[k, i] / self.p[k, i]
#                                          <= w[j] / self.p[k, j] + (1 - z[k, i, j]) * self.inst.max_e_inv[k],
#                                          name='eff x < t ' + str(k) + ' ' + str(i) + ' ' + str(j))
#
#                 for j in range(self.inst.L, self.inst.M):
#                     self.model.addConstr(self.w[k, i] * self.x[k, i] / self.p[k, i]
#                                          <= self.w[k, j] / self.p[k, j] + (1 - z[k, i, j]) * self.inst.max_e_inv[k],
#                                          name='eff x < x  ' + str(k) + ' ' + str(i) + ' ' + str(j))
#
#         self.model.setObjective(t.sum(), gb.GRB.MAXIMIZE)
#
#         if verbose:
#             print('Constraints time', time.time() - tt)
#
#         # self.model.setParam('DualReductions', 0)
#         self.model.optimize()
#
#
#
#         self.time = time.time() - tt
#
#         if self.model.Status == GRB.INFEASIBLE:
#             self.model.computeIIS()
#             for c in self.model.getConstrs():
#                 if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')
#         if self.model.Status == GRB.UNBOUNDED:
#             print('unbounded')
#         self.obj = self.model.objVal
#
#         self.final_gap = self.model.MIPGap
#         return self.model.objVal,  w.x
#
#
