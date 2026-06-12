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
        self.M = problem.M
        self.K = problem.K
        self.L = problem.L
        self.obj = None
        self.time = None
        self.final_gap = None

        self.x = None

    def solve(self, verbose=False, time_limit=None):
        if self.inst.K == 1:
            return self.solve_k1(verbose=verbose, time_limit=time_limit)
        else:
            return self.solve_all(verbose=verbose, time_limit=time_limit)

    def solve_all(self, verbose=False, time_limit=None):
        if not verbose:
            self.model.setParam('OutputFlag', 0)
        self.model.setParam('FeasibilityTol', 1e-9)
        self.model.setParam('IntFeasTol', 1e-9)
        self.model.setParam('NumericFocus', 3)  # Alta precisione numerica
        if self.inst.K >= 360 and self.inst.M >= 360:
            self.model.setParam('Threads', 4)
            self.model.setParam('SoftMemLimit', 90)

        tt = time.time()
        w = self.model.addMVar((self.L,), name='w')
        w_k = self.model.addMVar((self.K, self.M, self.M), name='w_k')
        w_k_in = self.model.addMVar((self.K, self.M), name='w_k_in')
        w_leader_in = self.model.addMVar((self.K, self.M), name='w_leader_in')
        x = self.model.addMVar((self.K, self.M, self.M), vtype=GRB.BINARY, name='x')
        y = self.model.addMVar((self.K, self.M), vtype=GRB.BINARY, name='y')

        # compute init solution with most efficient
        min_w_idxs = np.unravel_index(np.argmin(self.p[:, :self.L]), self.p[:, :self.L].shape)
        max_feasible_efficiency = (self.p[:, self.L:] / self.w[:, self.L:] *
                                   (self.w[:, self.L:] <= np.repeat(self.c[:, None], self.M - self.L, axis=1) )).max()
        init_w_val = self.p[:, :self.L].min() / max_feasible_efficiency

        BigM = max(self.c.max() + 1, self.w.max() + 1)
        w_init = np.ones(self.L) * BigM
        w_init[min_w_idxs[1]] = init_w_val
        w.start = w_init

        w_copy = self.inst.w.copy()
        w_k_copy = np.zeros((self.K, self.M, self.M))
        _, sol = self.inst.compute_obj(w_init.reshape(1, -1))

        for k in range(self.K):
            w_copy[k, :self.inst.L] = w_init
            idx_eff = np.argsort(w_copy[k] / self.inst.p[k])
            x_sol = np.zeros((self.M, self.M))
            x_sol[idx_eff, range(self.M)] = 1
            x[k].start = x_sol
            y[k].start = sol[k, idx_eff]
            w_k_copy[k, idx_eff, range(self.M)] = w_copy[k, idx_eff]
            # self.model.addConstr(x[k] == x_sol)
            # self.model.addConstr(y[k] == sol[k, idx_eff])

        w_k.start = w_k_copy
        # self.model.addConstr(w_k == w_k_copy)

        self.model.addConstr(x.sum(axis=-1) == 1, name='xi = 1 ')
        self.model.addConstr(x.sum(axis=1) == 1, name='xj = 1 ')


        # print('cap    ', self.c.max())
        for k in range(self.K):

            for j in range(self.M):

                if j < self.M - 1:
                    self.model.addConstr(
                        (w_k[k, :, j] / self.p[k, :]).sum() <= (w_k[k, :, j + 1] / self.p[k, :]).sum(),
                        name='ineff order ' + str(k) + ' ' + str(j))

                for i in range(self.L):
                    self.model.addConstr(
                        w_k[k, i, j] <= x[k, i, j] * BigM, name='force w_in if x ' + str(k) + ', ' + str(i) + ' ' + str(j)
                    )

                for i in range(self.L, self.M):
                    self.model.addConstr(
                        w_k[k, i, j] == x[k, i, j] * self.w[k, i], name='force w_k of F  = self.w ' + str(k) + ', ' + str(i) + ' ' + str(j)
                    )

            for j in range(self.M):
                ##### SE CI SONO OGGETTI CON PESO SUPERIORE ALLA CAPACITA' USARE C[K] COME UPPER NON E' SUFFICENTE
                self.model.addConstr(
                    w_k[k, :, j].sum() <= w_k_in[k, j] + (1 - y[k, j]) * BigM, #self.c[k],
                    name='force w_in > w if y ' + str(k) + ' ' + str(j)
                )

                self.model.addConstr(
                    w_k_in[k, j] <= y[k, j] * self.c[k], name='force w_in < 0 if y 0 ' + str(k) + ' ' + str(j)
                )

                self.model.addConstr(
                    w_k[k, :, j].sum() >= w_k_in[k, j], name='force w > w_in ' + str(k) + ' ' + str(j)
                )

                self.model.addConstr(
                    w_k_in[k, :j].sum() + w_k[k, :, j].sum() >= self.c[k] + 1e-8 - y[k, j] * self.c[k], name='force  y=1 if cap ' + str(k) + ' ' + str(j)
                )

                self.model.addConstr(
                    w_k_in[k, :j + 1].sum() + 1e-8 <= self.c[k] + (1 - y[k, j]) * BigM, name='force *** y=1 if cap ' + str(k) + ' ' + str(j)
                )

            for j in range(self.M):

                self.model.addConstr(
                    w_leader_in[k, j] <= w_k[k, :self.L, j].sum()
                )

                self.model.addConstr(
                    w_leader_in[k, j] <= y[k, j] * self.c[k]
                )
            ##### CREA PROBLEMI, MA MI SEMBRA CHE SENZA QUESTO NON STIAMO FORZANDO LE Y AD ESSERE 0 QUANDO L'OGGETTO CORRISPONDENTE ECCEDE LA CAPACITA' (RESIDUA)
            ##### MA SOLO AD ESSERE 1 QUANDO STA SOTTO LA CAPACITA (RESIUDA)
            # self.model.addConstr(w_k_in.sum(axis=-1) <= self.c, name='cap ' + str(k))
            self.model.addConstr(w == w_k[k, :self.L].sum(axis=-1), name='w == w_k' + str(k) + ' ')


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
            w_copy[k, :self.inst.L] = w_sol
            idx_eff = np.argsort(w_copy[k] / self.inst.p[k])
            sol[k] = y.x[k][np.argsort(idx_eff)]
            #
            # sol = y.x == 1
            #sol[k, self.inst.L:] = y.x[k, 1: -1][np.argsort(idx_eff)]
            #sol[k, :self.inst.L] = x.x[k].sum(axis=0) >= 1

        # print('cap', self.inst.c)
        # print('w', w_copy)
        # print('inef ', w_copy / self.inst.p)
        # print('sol ', sol)
        # print('used_cap', (w_copy * sol).sum(axis=1))
        obj, _ = self.inst.compute_obj(w_sol.reshape(1, -1))
        return obj, w_sol

    def solve_k1(self, verbose=False, time_limit=None):
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
        obj, _ = self.inst.compute_obj(w_sol.reshape(1, -1))

        return obj, w_sol

# 0.02328722462856877 * w_k_copy[0,0,0] + -0.02328722462856877 * w_k_copy[0,0,1] + 0.015397881251539788 * w_k_copy[0,1,0] + -0.015397881251539788 * w_k_copy[0,1,1] + 0.022562667809841837 * w_k_copy[0,2,0] + -0.022562667809841837 * w_k_copy[0,2,1]
