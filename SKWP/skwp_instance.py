import numpy as np


class SKWP_instance:
    def __init__(self, M, K):
        self.M = int(M)
        self.K = int(K)
        self.c = np.zeros(self.K)
        self.p = np.zeros((self.K, self.M))
        self.w = np.zeros((self.K, self.M))
        self.L = self.M // 2

        self.w = np.random.uniform(1, 100, size=(self.K, self.M))
        # self.c = self.w.sum(axis=1)//4
        # self.p = self.w + 10
        self.c = np.random.uniform(self.w.sum(axis=1)*0.1, self.w.sum(axis=1)*0.9)
        self.p = np.random.uniform(1, 100, size=(self.K, self.M))

        # self.max_w =  self.w[:, :self.L] .max(axis=0)
        self.max_w_ = self.compute_max() * 1.01
        self.max_w = self.c
        self.max_e_inv = (self.p[:, self.L:] / self.w[:, self.L:]).max(axis=1)
        pass

    def compute_max(self):
        p = self.p[:, self.L:]
        w = self.w[:, self.L:]
        efficiency = p / w
        indexes = np.argsort(-efficiency, axis=-1)
        efficiency_sorted = np.take_along_axis(efficiency, indexes, axis=-1)
        w_sorted = np.take_along_axis(w, indexes, axis=-1)
        cs = np.cumsum(w_sorted, axis=-1)
        sol = (cs < self.c[:, np.newaxis])

        if sol.sum() == 0:
            min_efficiency = self.p[:, :self.L].min() / self.c.max()
        else:
            min_efficiency = efficiency_sorted[sol].min()
        max_w = self.p[:, :self.L].max(axis=0) / min_efficiency

        # for i in range(self.L):
        #     max_w[i] = self.p[:, i].max() / min_efficiency
            # for k in range(self.K):
            #     for j in range(1, sol.shape[1]):
            #         if self.p[k, i] / efficiency_sorted[k, j] + cs[k, j - 1] <= self.c[k]:
            #             max_w[i] = self.p[k, i] / efficiency_sorted[k, j]

        return max_w

    def compute_obj(self, w):
        w = np.concat([w, self.w[:, self.L:]], axis=-1)
        efficiency = self.p / w
        indexes = np.argsort(-efficiency, axis=-1)
        w_sorted = np.take_along_axis(w, indexes, axis=-1)
        cs = np.cumsum(w_sorted, axis=-1)
        sol = (cs < self.c[:, np.newaxis])
        reversed_indexes = np.argsort(indexes, axis=-1)  # This gives you the reverse mapping
        # sol_original_order = np.take_along_axis(sol, reversed_indexes, axis=-1)
        # new_sol = np.zeros_like(sol)
        np.put_along_axis(sol, indexes, sol, axis=-1)
        return (w*sol)[:self.L].sum(axis=-1)

    def compute_obj_2(self, w_new_):
        w_new = np.stack((w_new_,) * self.K, axis=1)
        w = self.w.copy()
        w[:, :self.L] = w_new
        efficiency = self.p / w
        identical = np.any(np.diff(efficiency, axis=-1) == 0, axis=-1)
        indexes = np.argsort(-efficiency, axis=-1)
        w_sorted = np.take_along_axis(w, indexes, axis=-1)
        cs = np.cumsum(w_sorted, axis=-1)
        sol = (cs < self.c[ :, np.newaxis])
        np.put_along_axis(sol, indexes, sol, axis=-1)
        vals__ = (w * sol)[:, :self.L].sum(axis=-1).sum(axis=-1)
        return vals__, sol

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


    def sort_values(self):
        for k in range(self.K):
            idx = np.argsort(self.p[k]/self.w[k])[::-1]
            self.p[k] = self.p[k][idx]
            self.w[k] = self.w[k][idx]

    def __repr__(self):
        return 'M:' + str(self.M) + '-K:' + str(self.K)

    def __str__(self):
        return 'M:' + str(self.M) + '-K:' + str(self.K)




