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
        combs_bool = {}
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
            combs_bool[k] = np.zeros((len(combs[k]), self.inst.M), dtype=bool)
            z[k] = self.model.addMVar(len(combs[k]), vtype=GRB.BINARY)

            for c_idx, c in enumerate(combs[k]):
                combs_bool[k][c_idx, c] = True

            self.model.addConstr(z[k].sum() == 1, name='z ' + str(k))

            self.model.addConstr((combs_bool[k] * self.x[k]).sum(axis=1) >= combs_bool[k].sum(axis=1) * z[k],
                                 name='x > z ' + str(k))
            self.model.addConstr(self.x[k].sum() <=
                                 combs_bool[k].sum(axis=1) + (1 - z[k]) * (self.inst.M - combs_bool[k].sum(axis=1)),
                                 name='x < z ' + str(k))

            self.model.addConstr(combs_bool[k][:, :self.inst.L] @ t[k] + combs_bool[k][:, self.inst.L:] @ self.w[k][self.inst.L:] <=
                                 self.c[k] + (1 - z[k]) * (combs_bool[k][:, self.inst.L:] @ self.w[k][self.inst.L:]), name='w ')

            n_combs = len(combs[k])

            # Precompute p sums for each combination
            p_sums = combs_bool[k] @ self.p[k]

            # Create all pairs (c, q)
            c_indices = np.repeat(np.arange(n_combs), n_combs)
            q_indices = np.tile(np.arange(n_combs), n_combs)
            total_pairs = len(c_indices)

            # Build coefficient matrices
            p_coeff = np.zeros((total_pairs, n_combs))
            z_coeff = np.zeros((total_pairs, n_combs))

            row_indices = np.arange(total_pairs)

            # p_sums coefficients: p_sums[c] - p_sums[q]
            p_coeff[row_indices, c_indices] = 1
            p_coeff[row_indices, q_indices] -= 1

            # z coefficients: -M*z[c] - M*z[q]
            z_coeff[row_indices, c_indices] = -M
            z_coeff[row_indices, q_indices] = -M

            # RHS: -2M
            rhs = np.full(total_pairs, -2 * M)

            # Add both sets of constraints at once
            self.model.addConstr(
                p_coeff @ p_sums + z_coeff @ z[k] >= rhs,
                name=f"p_comparison_k{k}"
            )

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


