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

        self.x = self.model.addMVar((self.inst.K, self.inst.M), vtype=GRB.BINARY)


    def remove_sub_optimal(self, comb_k, k):
        p = np.array([self.p[k][c].sum() for c in comb_k])
        to_exclude = np.zeros_like(p, dtype=bool)
        follower_feasible = np.zeros_like(p, dtype=bool)
        for c_idx, c in enumerate(comb_k):
            c_f = [i for i in c if i >= self.inst.L]
            if len(c_f) == len(c):
                to_exclude[c_idx] = True
                if self.w[k][c_f].sum() <= self.c[k]:
                    follower_feasible[c_idx] = True
            elif self.w[k][c_f].sum() > self.c[k]:
                to_exclude[c_idx] = True
        p_follower = p * follower_feasible
        max_idx = np.argmax(p_follower)
        max_val = p_follower[max_idx]
        non_follower = 1 - to_exclude
        non_follower[max_idx] = True
        new_comb_k = [comb_k[i] for i in range(len(comb_k)) if non_follower[i] and (p[i] >= max_val)]
        return new_comb_k

    def solve(self, verbose=False):
        if not verbose:
            self.model.setParam('OutputFlag', 0)
        tt = time.time()
        combs = {}
        z = {}
        s = {}
        M = 1000000
        N = 2000000
        w = self.model.addMVar(self.inst.L)
        t = self.model.addMVar((self.inst.K, self.inst.L))
        for k in range(self.inst.K):
            lst = list(range(self.inst.M))
            combs_k = [list(subset) for r in range(1, len(lst) + 1) for subset in itertools.combinations(lst, r)]
            combs[k] = self.remove_sub_optimal(combs_k, k)
            z[k] = self.model.addMVar(len(combs[k]), vtype=GRB.BINARY)
            s[k] = self.model.addMVar(len(combs[k]), vtype=GRB.BINARY)


        for k in range(self.inst.K):
            # self.model.addConstr(z[k].sum() == 1, name='z ' + str(k))
            for c_idx, c in enumerate(combs[k]):
                c_l = [i for i in c if i < self.inst.L]
                c_f = [i for i in c if i >= self.inst.L]
                self.model.addConstr(t[k][c_l].sum() + self.w[k][c_f].sum() <= self.c[k] + (1 - s[k][c_idx]) * self.w[k][c_f].sum(), name='w ' + str(c_idx))
                for q_idx, q in enumerate(combs[k]):
                    self.model.addConstr(
                        self.p[k][list(c)].sum() >=
                        self.p[k][list(q)].sum() - (1 - z[k][c_idx]) * M - (1 - s[k][c_idx]) * M,
                        name = 'comb ' + str(c) + ' ' + str(q)
                    )

                self.model.addConstr(self.x[k][c].sum() >= len(c) * z[k][c_idx],
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

        self.model.setParam('DualReductions', 0)
        self.model.optimize()
        self.time = time.time() - tt

        if self.model.Status == GRB.INFEASIBLE:
            self.model.computeIIS()
            for c in self.model.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')
        if self.model.Status == GRB.UNBOUNDED:
            print('unbounded')


        return self.model.objVal,  w.x


