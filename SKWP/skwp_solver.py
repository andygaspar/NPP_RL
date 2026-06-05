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
        self.model.setParam('FeasibilityTol', 1e-9)
        self.model.setParam('IntFeasTol', 1e-9)
        self.model.setParam('NumericFocus', 3)  # Alta precisione numerica
        self.inst = problem
        self.p = problem.p
        self.w = problem.w
        self.c = problem.c
        self.M = problem.M
        self.K = problem.K
        self.L = problem.L
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
        return self.model.objVal, w.x

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
        p_F[-1] = 1  #self.inst.p.min() / 2
        sorted_p = self.inst.p[0, idx_eff + self.inst.L]
        p_F[1: -1] = sorted_p

        w_F[0] = self.inst.w.min() / 2
        w_F[-1] = self.inst.c[0]
        sorted_w = self.inst.w[0, idx_eff + self.inst.L]
        w_F[1: -1] = sorted_w

        cap_adjusted = self.c[0] + w_F[0]

        self.model.addConstr(x.sum(axis=0) <= 1, name='x < 1 ')

        for i in range(self.inst.F + 2):
            for j in range(self.inst.L):
                self.model.addConstr(x[i, j] + y[i] <= 1, name='x + y < 1 ' + str(i) + ' ' + str(j))

                self.model.addConstr(w[i, j] <= (self.inst.p[0, j] * w_F[i] / p_F[i]) * x[i, j],
                                     name='p * w / p ' + str(i) + ' ' + str(j))

        self.model.addConstr(
            cap_adjusted * (1 - y[0]) - w[0, :].sum() <= w_F[0] - 1e-8, name='< w_F 0')

        for i in range(1, self.inst.F + 2):
            self.model.addConstr(
                cap_adjusted * (1 - y[i]) - w[:i + 1, :].sum() - (w_F[:i] * y[:i]).sum() <=
                w_F[i] - 1e-7, name='< w_F --' + str(i))

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

    def solve_all(self, verbose=False, time_limit=None):
        if not verbose:
            self.model.setParam('OutputFlag', 0)

        if self.inst.K >= 360 and self.inst.M >= 360:
            self.model.setParam('Threads', 4)
            self.model.setParam('SoftMemLimit', 90)

        tt = time.time()
        w = self.model.addMVar((self.L,), name='w')
        w_k = self.model.addMVar((self.K, self.L, self.M), name='w_k')
        w_k_in = self.model.addMVar((self.K, self.M), name='w_k_in')
        w_leader_in = self.model.addMVar((self.K, self.M), name='w_leader_in')
        x = self.model.addMVar((self.K, self.M, self.M), vtype=GRB.BINARY, name='x')
        y = self.model.addMVar((self.K, self.M), vtype=GRB.BINARY, name='y')

        self.model.addConstr(x.sum(axis=-1) == 1, name='xi = 1 ')
        self.model.addConstr(x.sum(axis=1) == 1, name='xj = 1 ')
        for k in range(self.K):

            for j in range(self.M - 1):

                self.model.addConstr(
                    (w_k[k, :self.L, j] / self.p[k, :self.L]).sum() + (x[k, :self.L, j] * self.w[k, self.L:] / self.p[k, self.L:]).sum()
                    <= (w_k[k, :self.L, j + 1] / self.p[k, :self.L]).sum()
                    + (x[k, :self.L, j + 1] * self.w[k, self.L:] / self.p[k, self.L:]).sum(),
                    name='ineff order ' + str(k) + ' ' + str(j))

                for i in range(self.L):
                    self.model.addConstr(
                        w_k[k, i, j] <= x[k, i, j] * self.c[k]
                    )

            self.model.addConstr(
                w_k[k, :self.L, 0].sum() + (self.w[k, self.L:] * x[k, self.L:, 0]).sum() >= self.c[k] - y[k, 0] * self.c[k],
                name='force cap ' + str(k) + ' ' + str(0)
            )

            self.model.addConstr(
                w_k[k, :self.L, 0].sum() + (self.w[k, self.L:] * x[k, self.L:, 0]).sum() <= w_k_in[k, 0] + (1 - y[k, 0]) * self.c[k],
                name = 'force cap * 1 ' + str(k) + ' ' + str(0)
            )

            self.model.addConstr(
                w_k_in[k, 0] <= y[k, 0] * self.c[k],
                name='force cap * 2 ' + str(k) + ' ' + str(0)
            )

            self.model.addConstr(
                w_k[k, :self.L, 0].sum() + (self.w[k, self.L:] * x[k, self.L:, 0]).sum() >= w_k_in[k, 0]
            )

            self.model.addConstr(
                w_leader_in[k, 0] <= w_k[k, :self.L, 0].sum()
            )

            self.model.addConstr(
                w_leader_in[k, 0] <= y[k, 0] * self.c[k]
            )

            for j in range(1, self.M):
                self.model.addConstr(
                    w_k[k, :self.L, j].sum() + (self.w[k, self.L:] * x[k, self.L:, j]).sum() <= w_k_in[k, j] + (1 - y[k, j]) * self.c[k],
                    name='force w if y ' + str(k) + ' ' + str(j)
                )

                self.model.addConstr(
                    w_k_in[k, j] <= y[k, j] * self.c[k]
                )

                self.model.addConstr(
                    w_k[k, :self.L, j].sum() + (self.w[k, self.L:] * x[k, self.L:, j]).sum() >= w_k_in[k, j]
                )

                self.model.addConstr(
                    w_k_in[k, :j + 1].sum() >= self.c[k] - y[k, j] * self.c[k]
                )

                self.model.addConstr(
                    w_leader_in[k, j] <= w_k[k, :self.L, j].sum()
                )

                self.model.addConstr(
                    w_leader_in[k, j] <= y[k, j] * self.c[k]
                )



            self.model.addConstr(w == w_k[k].sum(axis=-1), name='w <= w_k' + str(k) + ' ')
            # self.model.addConstr(w <= w_k[k].sum(axis=0) + (1 - x[k].sum(axis=0)) * self.inst.c[k], name='w <= w_k' + str(k) + ' ')
            # self.model.addConstr(w >= w_k[k].sum(axis=0) - (1 - x[k].sum(axis=0)) * self.inst.c[k], name='w >= w_k' + str(k) + ' ')

        # self.model.addConstr(w <= x.sum(axis=0).sum(axis=0) * self.inst.c.max())

        constr_time = time.time() - tt

        self.model.setObjective(w_leader_in.sum(), gb.GRB.MAXIMIZE)
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
        w_copy = self.inst.w.copy()
        w_sol = w.x
        for k in range(self.inst.K):
            w_copy[:, :self.inst.L] = w_sol
            idx_eff = np.argsort(-self.inst.p[k, self.inst.L:] / self.inst.w[k, self.inst.L:])

            sol[k, self.inst.L:] = y.x[k, 1: -1][np.argsort(idx_eff)]
            sol[k, :self.inst.L] = x.x[k].sum(axis=0) >= 1

        print('cap', self.inst.c)
        print('w', w_copy)
        print('inef ', w_copy / self.inst.p)
        print('sol ', sol)
        print('used_cap', (w_copy * sol).sum(axis=1))

        return self.model.objVal, w_sol, sol
