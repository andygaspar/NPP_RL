import time

import gurobipy as gb
from gurobipy import GRB
import numpy as np
from scipy._lib.cobyqa import problem

from skpp_instance import SPKK_instance


class SKPP:

    def __init__(self, problem: SPKK_instance):

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

    def solve(self, verbose=False):
        if not verbose:
            self.model.setParam('OutputFlag', 0)
        tt = time.time()
        combs = {}
        z = {}
        M = 10000
        N = 2000
        p = self.model.addMVar(self.inst.L)
        t = self.model.addMVar((self.inst.K, self.inst.L))
        for k in range(self.inst.K):
            combs[k] = self.maximal_combinations_final(self.inst.w[k].tolist(), self.inst.c[k])
            z[k] = self.model.addMVar(len(combs[k]), vtype=GRB.BINARY)

        for k in range(self.inst.K):
            self.model.addConstr(z[k].sum() == 1, name='z ' + str(k))
            for c_idx, c in enumerate(combs[k]):
                c_l = [i for i in c if i < self.inst.L]
                c_f = [i for i in c if i >= self.inst.L]
                for q_idx, q in enumerate(combs[k]):
                    q_l = [i for i in q if i < self.inst.L]
                    q_f = [i for i in q if i >= self.inst.L]
                    self.model.addConstr(
                        self.p[k][c_f].sum() + t[k][c_l].sum() >=
                        self.p[k][q_f].sum() + p[q_l].sum() - (1 - z[k][c_idx]) * M,
                        name = 'comb ' + str(c) + ' ' + str(q)
                    )

                self.model.addConstr(self.x[k][list(c)].sum() >= len(c) * z[k][c_idx],
                                     name='x > z ' + str(k) + ' ' + str(c))
                self.model.addConstr(self.x[k].sum() <= len(c) + (1 - z[k][c_idx]) * (self.inst.M - len(c)),
                                     name='x < z ' + str(k) + ' ' + str(c))

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


class SKPP_pop:

    def __init__(self, problem: SPKK_instance, pop_size: int, preset=True):
        self.model = gb.Model("BilevelReformulation")
        self.model.setParam('OutputFlag', 0)
        self.preset = preset
        if preset:
        #
            self.model.setParam('Presolve', 2)  # Aggressive presolve
        # #     self.model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
        #     self.model.setParam('Heuristics', 0.8)  # More heuristics
            self.model.setParam('Cuts', 0)  # Moderate cut generation
        self.inst = problem
        self.pop_size = pop_size
        self.p = np.array([problem.p for _ in range(self.pop_size)], dtype=float)
        self.c = np.stack((self.inst.c,) * pop_size)
        self.w = np.array([self.inst.w for _ in range(self.pop_size)])
        self.x = self.model.addMVar((pop_size, self.inst.K, self.inst.M), vtype=GRB.BINARY)
        self.efficiency = np.zeros_like(self.w)
        self.reindex_mat = np.stack((np.stack((range(self.inst.M),) * self.inst.K),) * self.pop_size)

        self.eps = 1e-6

    def solve(self, p_new):
        p_new = np.stack((p_new,) * self.inst.K, axis=1)
        # p_new = np.array([[p_new[i]] * self.inst.K for i in range(self.pop_size)])

        p = self.p.copy()
        p[:, :, :self.inst.L] = p_new

        if self.preset:
            self.x.Start = self.solve_x(p_new)
            self.x.BranchPriority = self.efficiency.astype(int)

        self.model.setObjective((p * self.x).sum() + ((self.p - p) * self.x)[:, :, :self.inst.L].sum() * self.eps
                                , GRB.MAXIMIZE)
        self.model.addConstr(
            (self.x * self.w).sum(axis=2) <= self.inst.c
        )
        self.model.optimize()
        # print(self.x.x)
        # print('ppp', self.model.ObjVal)
        vals = ((self.p - p) * self.x.x)[:, :, :self.inst.L].sum(axis=-1).sum(axis=-1)
        return vals, vals.max()

    def solve_x(self, p_new):
        p = self.p.copy()
        self.efficiency = p / self.w
        indices = np.argsort(-self.efficiency, axis=-1)
        inverse_indices = np.argsort(indices, axis=-1)
        w_sorted = np.take_along_axis(self.w, indices, axis=-1)
        x_sorted = np.zeros_like(p, dtype=bool)
        x_sorted[:, :, 0] = True
        cap_used = w_sorted[:, :, 0]
        for i in range(1, self.inst.M):
            new_cap = cap_used + w_sorted[:, :, i]
            x_sorted[:, :, i] = (new_cap <= self.c)
            cap_used += w_sorted[:, :, i] * x_sorted[:, :, i]

        x = np.take_along_axis(x_sorted, inverse_indices, axis=-1)
        return x


class SKPP_greedy_pop:
    def __init__(self, problem: SPKK_instance, pop_size: int):
        self.inst = problem
        self.pop_size = pop_size
        self.p = np.array([problem.p for _ in range(self.pop_size)], dtype=float)
        self.c = np.stack((self.inst.c,) * self.pop_size)
        self.w = np.array([self.inst.w for _ in range(self.pop_size)])
        self.efficiency = np.zeros_like(self.w)
        self.reindex_mat = np.stack((np.stack((range(self.inst.M),) * self.inst.K),) * self.pop_size)

    def solve(self, p_new):
        p_new = np.stack((p_new,) * self.inst.K, axis=1)
        p = self.p.copy()
        p[:, :, :self.inst.L] = p_new
        self.efficiency = p / self.w
        indices = np.argsort(-self.efficiency, axis=-1)
        inverse_indices = np.argsort(indices, axis=-1)
        w_sorted = np.take_along_axis(self.w, indices, axis=-1)
        x_sorted = np.zeros_like(p, dtype=bool)
        x_sorted[:, :, 0] = True
        cap_used = w_sorted[:, :, 0]
        for i in range(1, self.inst.M):
            new_cap = cap_used + w_sorted[:, :, i]
            x_sorted[:, :, i] = (new_cap <= self.c)
            cap_used += w_sorted[:, :, i] * x_sorted[:, :, i]

        x = np.take_along_axis(x_sorted, inverse_indices, axis=-1)[:, :, :self.inst.L]
        vals = ((self.p[:, :, :self.inst.L] - p_new) * x).sum(axis=-1).sum(axis=-1)
        # print((p * np.take_along_axis(x_sorted, inverse_indices, axis=-1)).sum(axis=-1) )
        return vals, vals.max()




