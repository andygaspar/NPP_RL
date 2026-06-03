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

    def solve(self, verbose=False, time_limit=None, xx=None, ww=None):
        if not verbose:
            self.model.setParam('OutputFlag', 0)

        if self.inst.K >= 360 and self.inst.M >= 360:
            self.model.setParam('Threads', 4)
            self.model.setParam('SoftMemLimit', 90)

        tt = time.time()
        w = self.model.addMVar(self.inst.L)
        t = self.model.addMVar((self.inst.K, self.inst.L))
        d_L = self.model.addMVar((self.inst.K, self.inst.L))  # deficiency
        d = self.w / self.p
        x = self.model.addMVar((self.inst.K, self.inst.M), vtype=GRB.BINARY)
        W = self.c

        if xx is not None:
            self.model.addConstr(x == xx)
        if ww is not None:
            self.model.addConstr(w == ww)
        # init sol

        e_max = 1 / d
        idx_e_max = np.argmax(e_max[:, self.inst.L:], axis=1) + self.inst.L
        idx_p_min = np.argmin(self.p[:, :self.inst.L], axis=1)
        w_max = self.w[range(self.inst.K), idx_e_max]
        w_init = np.ones((self.inst.K, self.inst.L)) * 10 ** 5 + 1
        w_init[range(self.inst.K), idx_p_min] = w_max / self.p[range(self.inst.K), idx_p_min]
        w_init = w_init.min(axis=0)
        w_init[w_init > 10 ** 5] = 0
        w.Start = w_init

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

        constr_time = time.time() - tt
        self.model.setObjective(t.sum(), gb.GRB.MAXIMIZE)
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit - constr_time)

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

    def solve_k1(self, verbose=False, time_limit=None, xx=None, ww=None):
        if not verbose:
            self.model.setParam('OutputFlag', 0)

        if self.inst.K >= 360 and self.inst.M >= 360:
            self.model.setParam('Threads', 4)
            self.model.setParam('SoftMemLimit', 90)

        tt = time.time()
        w = self.model.addMVar((self.inst.F + 2, self.inst.L), name='w')
        x = self.model.addMVar((self.inst.F + 2, self.inst.L), vtype=GRB.BINARY, name='x')
        y = self.model.addMVar(self.inst.F + 2, vtype=GRB.BINARY, name='y')

        idx_eff = np.argsort(-self.inst.p[0, self.inst.L:] / self.inst.w[0, self.inst.L:])

        p_F = np.zeros(self.inst.F + 2)
        w_F = np.zeros(self.inst.F + 2)

        p_F[0] = self.inst.p.max() + 1000000
        p_F[-1] = 1 #self.inst.p.min() / 2
        sorted_p = self.inst.p[0, idx_eff + self.inst.L]
        p_F[1: -1] = sorted_p

        w_F[0] = self.inst.w.min() / 2
        w_F[-1] = self.inst.c[0]
        sorted_w = self.inst.w[0, idx_eff + self.inst.L]
        w_F[1: -1] = sorted_w

        # p_F = self.inst.p[0, idx_eff + self.inst.L]
        # w_F = self.inst.w[0, idx_eff + self.inst.L]
        cap_adjusted = self.c[0] + w_F[0]

        self.model.addConstr(x.sum(axis=0) <= 1, name='x < 1 ')
        # self.model.addConstr(x[2].sum() >= 1, name='x forced ')
        # self.model.addConstr(y[1] >= 1, name='x forced ')

        # self.model.addConstr(x.sum() <= 1, name='x > 1 ')
        # print('w_F')
        # print(w_F)
        # print(cap_adjusted)
        for i in range(self.inst.F + 2):
            for j in range(self.inst.L):
                self.model.addConstr(x[i, j] + y[i] <= 1, name='x + y < 1 ' + str(i) + ' ' + str(j))

                self.model.addConstr(w[i, j] <= (self.inst.p[0, j] * w_F[i] / p_F[i]) * x[i, j],
                                     name='p * w / p ' + str(i) + ' ' + str(j))

        self.model.addConstr(
            cap_adjusted * (1 - y[0]) - w[0, :].sum() <= w_F[0] - 2e-9, name='< w_F 0')

        for i in range(1, self.inst.F + 2):
            self.model.addConstr(
                cap_adjusted * (1 - y[i]) - w[:i + 1, :].sum() - (w_F[:i] * y[:i]).sum() <=
                w_F[i] - 2e-9, name='< w_F --' + str(i))

        self.model.addConstr(w.sum() + (w_F * y).sum() <= cap_adjusted, name='cap ')

        constr_time = time.time() - tt

        self.model.setObjective(w.sum(), gb.GRB.MAXIMIZE)
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit - constr_time)

        self.model.optimize()
        self.time = time.time() - tt

        if self.model.Status == GRB.INFEASIBLE or self.model.Status == 4:
            self.model.computeIIS()
            for c in self.model.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')
        if self.model.Status == GRB.UNBOUNDED:
            print('unbounded')
        self.obj = self.model.objVal
        self.final_gap = self.model.MIPGap

        sol = np.zeros_like(self.inst.p, dtype=bool)
        sol[0, self.inst.L:] = y.x[1: -1][np.argsort(idx_eff)]
        sol[0, :self.inst.L] = x.x.sum(axis=0) >= 1

        w_sol = w.x.sum(axis=0)

        w_copy = self.inst.w.copy()
        w_copy[0, :self.inst.L] = w_sol

        # print('cap', self.inst.c)
        # print('w', w_copy)
        # print('inef ', w_copy/self.inst.p)
        # print('sol ', sol)
        # print('used_cap', (w_copy * sol).sum())

        return self.model.objVal, w_sol, sol

    def solve_k2(self, verbose=False, time_limit=None, xx=None, ww=None):

        if not verbose:
            self.model.setParam('OutputFlag', 0)

        if self.inst.K >= 360 and self.inst.M >= 360:
            self.model.setParam('Threads', 4)
            self.model.setParam('SoftMemLimit', 90)

        tt = time.time()
        w = self.model.addMVar((self.inst.F, self.inst.L), name='w')
        x = self.model.addMVar((self.inst.F, self.inst.L), vtype=GRB.BINARY, name='x')
        y = self.model.addMVar(self.inst.F, vtype=GRB.BINARY, name='y')


        idx_eff = np.argsort(-self.inst.p[0, self.inst.L:] / self.inst.w[0, self.inst.L:])

        p_F = self.inst.p[0, idx_eff + self.inst.L]
        w_F = self.inst.w[0, idx_eff + self.inst.L]
        # cap_adjusted = self.c[0] + self.inst.w.min() / 2

        self.model.addConstr(x.sum(axis=0) <= 1, name='x < 1 ')
        # self.model.addConstr(x[2].sum() >= 1, name='x forced ')
        # self.model.addConstr(y[1] >= 1, name='x forced ')

        for i in range(self.inst.F - 1):
            self.model.addConstr(y[i] >= y[i + 1], name='y order ' + str(i))
            self.model.addConstr(y[i] * self.inst.L >= x[i + 1, :].sum(), name='y order ' + str(i))

        for i in range(self.inst.F):
            for j in range(self.inst.L):
                self.model.addConstr(
                    w[i, j] <= (self.inst.p[0, j] * w_F[i] / p_F[i]) * x[i, j],
                    name='p * w / p ' + str(i) + ' ' + str(j))

        self.model.addConstr(w.sum() + (w_F * y).sum() <= self.c[0], name='cap ')

        constr_time = time.time() - tt

        self.model.setObjective(w.sum(), gb.GRB.MAXIMIZE)
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit - constr_time)

        self.model.optimize()
        self.time = time.time() - tt

        if self.model.Status == GRB.INFEASIBLE or self.model.Status == 4:
            self.model.computeIIS()
            for c in self.model.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')
        if self.model.Status == GRB.UNBOUNDED:
            print('unbounded')
        self.obj = self.model.objVal
        self.final_gap = self.model.MIPGap

        sol = np.zeros_like(self.inst.p, dtype=bool)
        sol[0, self.inst.L:] = y.x[np.argsort(idx_eff)]
        sol[0, :self.inst.L] = x.x.sum(axis=0) >= 1
        return self.model.objVal, w.x.sum(axis=0) + 2*self.c[0] * (1 - w.x.sum(axis=0).astype(bool)), w.x, x.x, y.x

    def solve_k3(self, verbose=False, time_limit=None, xx=None, ww=None):

        if not verbose:
            self.model.setParam('OutputFlag', 0)

        if self.inst.K >= 360 and self.inst.M >= 360:
            self.model.setParam('Threads', 4)
            self.model.setParam('SoftMemLimit', 90)

        tt = time.time()
        w = self.model.addMVar((self.inst.L), name='w')
        w_k = self.model.addMVar((self.inst.K, self.inst.F + 1, self.inst.L), name='w')
        x = self.model.addMVar((self.inst.K, self.inst.F + 1, self.inst.L), vtype=GRB.BINARY, name='x')
        y = self.model.addMVar((self.inst.K, self.inst.F), vtype=GRB.BINARY, name='y')
        # if ww is not None:
        #     self.model.addConstr(w == ww, name='w  ww 0')
        # cap_adjusted = self.c[0] + self.inst.w.min() / 2
        # for j in range(self.inst.L):
        #     self.model.addConstr(x[:, :, j].sum() * self.inst.c.max() >= w[j])

        self.model.addConstr(x[0, :, 1].sum() == 1)

        for k in range(self.inst.K):

            for j in range(self.inst.L):
                for jj in range(j+1, self.inst.L):
                    self.model.addConstr(w[j]/self.inst.p[k, j] >= (w_k[k, :, jj].sum(axis=0)) / self.inst.p[k, jj]
                                         - self.c[k]*x[k,:, jj].sum())
                    self.model.addConstr(w[jj] / self.inst.p[k, jj] >= (w_k[k, :, j].sum(axis=0)) / self.inst.p[k, j]
                                         - self.c[k] * x[k, :, j].sum())

            self.model.addConstr(w_k[k].sum(axis=0) >= w - (1 - x[k].sum(axis=0)) * self.inst.c.max(), name='w wk > ' + str(k))
            self.model.addConstr(w_k[k].sum(axis=0) <= w + (1 - x[k].sum(axis=0)) * self.inst.c.max(), name='w wk < ' + str(k))

            idx_eff = np.argsort(-self.inst.p[k, self.inst.L:] / self.inst.w[k, self.inst.L:])
            p_F = self.inst.p[k, idx_eff + self.inst.L]
            w_F = self.inst.w[k, idx_eff + self.inst.L]

            self.model.addConstr(x[k].sum(axis=0) <= 1, name='x < 1 ' + str(k))
            # self.model.addConstr(x[2].sum() >= 1, name='x forced ')
            # self.model.addConstr(y[1] >= 1, name='x forced ')

            for i in range(self.inst.F - 1):
                self.model.addConstr(y[k, i] >= y[k, i + 1],
                                     name='y order ' + str(k) + ' ' + str(i))

            for i in range(self.inst.F):
                self.model.addConstr(y[k, i] * self.inst.L >= x[k, i + 1, :].sum(),
                                     name='y order ' + str(k) + ' ' + str(i))

            for j in range(self.inst.L):
                for i in range(self.inst.F):
                    self.model.addConstr(
                        w_k[k, i, j] <= (self.inst.p[k, j] * w_F[i] / p_F[i]) * x[k, i, j],
                        name='p * w / p ' + str(k) + ' ' + str(i) + ' ' + str(j))

                self.model.addConstr(
                    w_k[k, -1, j] <= self.c[k] * x[k, -1, j],
                    name='p * w / p ' + str(k) + ' ' + str(-1) + ' ' + str(j))
                self.model.addConstr(
                    w_k[k, -1, j] >= (self.inst.p[k, j] * w_F[-1] / p_F[-1]) * x[k, -1, j],
                    name='p * w / p ' + str(k) + ' ' + str(-1) + ' ' + str(j))

            self.model.addConstr(w_k[k].sum() + (w_F * y[k]).sum() <= self.c[k], name='cap ' + str(k) + ' ')

        constr_time = time.time() - tt

        self.model.setObjective(w_k.sum(), gb.GRB.MAXIMIZE)
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit - constr_time)

        self.model.optimize()
        self.time = time.time() - tt

        if self.model.Status == GRB.INFEASIBLE or self.model.Status == 4:
            self.model.computeIIS()
            for c in self.model.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')
        if self.model.Status == GRB.UNBOUNDED:
            print('unbounded')
        self.obj = self.model.objVal
        self.final_gap = self.model.MIPGap
        return self.model.objVal, w.x, w_k.x, x.x, y.x


#
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
