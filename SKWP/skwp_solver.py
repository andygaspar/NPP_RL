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
        self.obj = None
        self.time = None

        self.x = self.model.addMVar((self.inst.K, self.inst.M), vtype=GRB.BINARY)


    def solve(self, verbose=False):
        if not verbose:
            self.model.setParam('OutputFlag', 0)
        tt = time.time()
        combs = {}
        z = {}
        M = 10000
        N = 2000
        w = self.model.addMVar(self.inst.L)
        t = self.model.addMVar((self.inst.K, self.inst.L))
        for k in range(self.inst.K):
            lst = list(range(self.inst.M))
            for r in range(1, len(lst) + 1):
                yield from itertools.combinations(lst, r)
            combs[k] = lst
            z[k] = self.model.addMVar(len(combs[k]), vtype=GRB.BINARY)

        for k in range(self.inst.K):
            self.model.addConstr(z[k].sum() == 1, name='z ' + str(k))
            for c_idx, c in enumerate(combs[k]):
                c_l = [i for i in c if i < self.inst.L]
                c_f = [i for i in c if i >= self.inst.L]
                self.model.addConstr(t[k][c_l].sum() + self.w[k][c_f] <= self.c[k] + (1 - z[k][c_idx]) * M, name='w ' + str(c_idx))
                for q_idx, q in enumerate(combs[k]):
                    self.model.addConstr(
                        self.p[k][c].sum() >= self.p[k][q].sum() - (1 - z[k][c_idx]) * M,
                        name = 'comb ' + str(c) + ' ' + str(q)
                    )

                self.model.addConstr(self.x[k][list(c)].sum() >= len(c) * z[k][c_idx],
                                     name='x > z ' + str(k) + ' ' + str(c))
                self.model.addConstr(self.x[k].sum() <= len(c) + (1 - z[k][c_idx]) * (self.inst.M - len(c)),
                                     name='x < z ' + str(k) + ' ' + str(c))

            for i in range(self.inst.L):
                self.model.addConstr(t[k, i] <= self.x[k, i] * N, name='t < x ' + str(k) + ' ' + str(i))
                self.model.addConstr(w[i] - t[k, i] <= (1 - self.x[k, i]) * N, name='p - t >  ' + str(k) + ' ' + str(i))
                self.model.addConstr(t[k, i] <= w[i], name='t < p ' + str(k) + ' ' + str(i))

        self.model.setObjective(t.sum(), gb.GRB.MAXIMIZE)

        if verbose:
            print('Constraints time', time.time() - tt)

        self.model.optimize()
        self.time = time.time() - tt
        self.obj = self.model.objVal
        if self.model.Status == GRB.INFEASIBLE:
            self.model.computeIIS()
            for c in self.model.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')

        return self.model.objVal,  w.x


