import time

import gurobipy as gb
from gurobipy import GRB
import numpy as np
from scipy._lib.cobyqa import problem

from SKPP.skpp_instance import SKPP_instance


class SKPP:

    def __init__(self, problem: SKPP_instance):

        self.model = gb.Model("BilevelReformulation")

        self.inst = problem
        self.p = problem.p
        self.obj = None
        self.time = None

        self.x = self.model.addMVar((self.inst.K, self.inst.M), vtype=GRB.BINARY)

    @staticmethod
    def maximal_combinations_final(weights, capacity):

        n = len(weights)
        result = []

        def backtrack(start, combo, current_weight):
            # Try to add more items
            added_something = False

            for i in range(start, n):
                new_weight = current_weight + weights[i]
                if new_weight <= capacity:
                    added_something = True
                    combo.append(i)
                    backtrack(i + 1, combo, new_weight)
                    combo.pop()

            # If we couldn't add anything more, save this combination
            if not added_something and combo:
                result.append(tuple(combo))

        backtrack(0, [], 0)
        return result

    def remove_sub_optimal(self, comb_k, k):
        p = np.array([self.p[k][list(c)].sum() for c in comb_k])
        to_exclude = np.zeros_like(p, dtype=bool)
        for c_idx, c in enumerate(comb_k):
            c_f = [i for i in c if i >= self.inst.L]
            if len(c_f) == len(c):
                to_exclude[c_idx] = True

        p_follower = p * to_exclude
        max_idx = np.argmax(p_follower)
        non_follower = 1 - to_exclude
        non_follower[max_idx] = True
        new_comb_k = [comb_k[i] for i in range(len(comb_k)) if non_follower[i]]
        return new_comb_k

    def solve(self, verbose=False):
        if not verbose:
            self.model.setParam('OutputFlag', 0)
        tt = time.time()
        combs = {}
        combs_bool = {}
        z = {}
        M = 10000
        N = 2000
        p = self.model.addMVar(self.inst.L)
        t = self.model.addMVar((self.inst.K, self.inst.L))
        for k in range(self.inst.K):
            combs_k = self.maximal_combinations_final(self.inst.w[k].tolist(), self.inst.c[k])
            combs[k] = self.remove_sub_optimal(combs_k, k)
            combs_bool[k] = np.zeros((len(combs[k]), self.inst.M), dtype=bool)
            for c_idx, c in enumerate(combs[k]):
                combs_bool[k][c_idx, c] = True

            z[k] = self.model.addMVar(len(combs[k]), vtype=GRB.BINARY)

        for k in range(self.inst.K):
            self.model.addConstr(z[k].sum() == 1, name='z ' + str(k))
            self.model.addConstr((combs_bool[k] * self.x[k]).sum(axis=1) >= combs_bool[k].sum(axis=1) * z[k],
                                 name='x > z ' + str(k))
            self.model.addConstr(self.x[k].sum() <=
                                 combs_bool[k].sum(axis=1) + (1 - z[k]) * (self.inst.M - combs_bool[k].sum(axis=1)),
                                 name='x < z ' + str(k))
            # for c_idx, c in enumerate(combs[k]):
            #
            #     self.model.addConstr(self.x[k][list(c)].sum() >= len(c) * z[k][c_idx],
            #                          name='x > z ' + str(k) + ' ' + str(c))
            #     self.model.addConstr(self.x[k].sum() <= len(c) + (1 - z[k][c_idx]) * (self.inst.M - len(c)),
            #                          name='x < z ' + str(k) + ' ' + str(c))

            constant_part = combs_bool[k][:, self.inst.L:] @ self.p[k, self.inst.L:]  # numpy array
            variable_part = combs_bool[k][:, :self.inst.L] @ t[k]  # MVar

            # Combine: numpy array + MVar = MVar
            constr_expr = constant_part + variable_part
            n_combs = constr_expr.shape[0]

            # Pre-calculate size
            total_pairs = n_combs * (n_combs - 1)

            # Create index arrays
            i_indices = np.repeat(np.arange(n_combs), n_combs)
            j_indices = np.tile(np.arange(n_combs), n_combs)

            # Remove i == j cases
            mask = i_indices != j_indices
            i_indices = i_indices[mask]
            j_indices = j_indices[mask]

            # Initialize coefficient matrix
            coeff_matrix = np.zeros((total_pairs, n_combs))

            # Fill using vectorized operations
            row_indices = np.arange(total_pairs)
            coeff_matrix[row_indices, i_indices] = 1
            coeff_matrix[row_indices, j_indices] = -1

            bigM_terms = np.zeros((total_pairs, n_combs))
            bigM_terms[row_indices, i_indices] = M  # Coefficient for z[i]

            # Add constraints
            self.model.addConstr(coeff_matrix @ constr_expr - bigM_terms @ z[k] >= -M)

            for i in range(self.inst.L):
                self.model.addConstr(t[k, i] <= self.x[k, i] * N, name='t < x ' + str(k) + ' ' + str(i))
                self.model.addConstr(p[i] - t[k, i] <= (1 - self.x[k, i]) * N, name='p - t >  ' + str(k) + ' ' + str(i))
                self.model.addConstr(t[k, i] <= p[i], name='t < p ' + str(k) + ' ' + str(i))

        self.model.setObjective(
            (self.p[:, :self.inst.L] * self.x[:, :self.inst.L]).sum() - t.sum(), gb.GRB.MAXIMIZE
        )

        if verbose:
            print('Constraints time', time.time() - tt)

        self.model.optimize()
        self.time = time.time() - tt
        self.obj = self.model.objVal
        if self.model.Status == GRB.INFEASIBLE:
            self.model.computeIIS()
            for c in self.model.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')

        return self.model.objVal,  p.x


