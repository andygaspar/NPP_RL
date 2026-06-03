import numpy as np


class SKWP_instance:
    def __init__(self, M, K):
        self.M = int(M)
        self.K = int(K)
        self.c = np.zeros(self.K)
        self.p = np.zeros((self.K, self.M))
        self.w = np.zeros((self.K, self.M))
        self.L = self.M // 2
        self.F = self.M - self.L

        self.w = np.random.uniform(1, 100, size=(self.K, self.M))
        # self.c = self.w.sum(axis=1)//4
        # self.p = self.w + 10
        self.c = np.random.uniform(self.w.sum(axis=1) * 0.1, self.w.sum(axis=1) * 0.9)
        self.p = np.random.uniform(1, 100, size=(self.K, self.M))

        self.p = np.round(self.p, 3)
        self.c = np.round(self.c, 3)
        self.w = np.round(self.w, 3)

        idx_p = np.argsort(-self.p[:, : self.L], axis=-1)
        adj_p = np.argsort(idx_p, axis=-1)
        self.adjustment_p = np.zeros_like(self.p)
        self.adjustment_p[:, :self.L] += (adj_p + 1) * 1e-16  # self.max_w =  self.w[:, :self.L].max(axis=0)

        # self.max_w =  self.w[:, :self.L] .max(axis=0)
        self.max_w = self.compute_max()
        # self.max_w = self.c
        self.max_e_inv = (self.p[:, self.L:] / self.w[:, self.L:]).max(axis=1)

    def compute_max(self):
        p = self.p[:, self.L:]
        w = self.w[:, self.L:]
        efficiency = p / w
        indexes = np.argsort(-efficiency, axis=-1)
        # efficiency_sorted = np.take_along_axis(efficiency, indexes, axis=-1)
        # w_sorted = np.take_along_axis(w, indexes, axis=-1)
        # cs = np.cumsum(w_sorted, axis=-1)
        # sol = (cs < self.c[:, np.newaxis])
        #
        # min_cap_left = (self.c - (w_sorted * sol).sum(axis=-1)).min()
        #
        # min_efficiency = efficiency_sorted[sol].min()
        # max_w = self.p[:, :self.L].max(axis=0) / min_efficiency
        # max_w[max_w < min_cap_left] = min_cap_left
        # max_w[max_w > self.c.max()] = self.c.max()
        sol = np.zeros_like(w, dtype=bool)
        # Greedy per ogni utente (istanza K)
        for k in range(self.K):
            capacity_left = self.c[k]
            for idx in indexes[k]:
                if w[k, idx] <= capacity_left:  # Se l'item può entrare
                    sol[k, idx] = True
                    capacity_left -= w[k, idx]
                # Altrimenti salta l'item e continua col prossimo

        # Calcola min_cap_left su tutti gli utenti
        w_sorted = w[sol]  # Nota: questo non funziona bene con broadcasting
        # Meglio calcolare per ogni utente:
        min_cap_left = float('inf')
        for k in range(self.K):
            cap_left = self.c[k] - (w[k] * sol[k]).sum()
            min_cap_left = min(min_cap_left, cap_left)

        # Calcola min_efficiency tra gli item selezionati
        min_efficiency = float('inf')
        for k in range(self.K):
            for idx in range(len(p[k])):
                if sol[k, idx]:
                    min_efficiency = min(min_efficiency, efficiency[k, idx])

        # Se nessun item selezionato, usa un default
        if min_efficiency == float('inf'):
            min_efficiency = efficiency.max()  # o un valore di default

        max_w = self.p[:, :self.L].max(axis=0) / min_efficiency
        max_w[max_w < min_cap_left] = min_cap_left
        max_w[max_w > self.c.max()] = self.c.max()

        return max_w

        return max_w

    def compute_obj(self, w_new_):
        w_new = np.stack((w_new_,) * self.K, axis=1)
        w = self.w.copy()
        w[:, :self.L] = w_new
        w = w.round(9)
        inv_efficiency = np.round(w / (self.p), 9) - self.adjustment_p
        # identical = np.any(np.diff(efficiency, axis=-1) == 0, axis=-1)
        indexes = np.argsort(inv_efficiency, axis=-1)
        # print(indexes)
        # w_sorted = np.take_along_axis(w, indexes, axis=-1)
        # cs = np.cumsum(w_sorted, axis=-1)
        # sol = (cs <= self.c[:, np.newaxis])
        # np.put_along_axis(sol, indexes, sol, axis=-1)
        # # print(sol, 'solution')
        # val = (w * sol)[:, :self.L].sum(axis=-1).sum(axis=-1)
        # Inizializza soluzione con zeri
        sol = np.zeros_like(w, dtype=bool)

        # Per ogni utente (istanza K)
        for k in range(self.K):
            capacity_left = self.c[k]
            # Scorre gli item in ordine di efficienza
            for idx in indexes[k]:
                if w[k, idx] <= capacity_left + 1e-8:  # Se l'item può entrare
                    sol[k, idx] = True
                    capacity_left -= w[k, idx]
                    capacity_left = capacity_left.round(9)
                # Altrimenti salta l'item e continua con il prossimo
        val = (w * sol)[:, :self.L].sum(axis=-1).sum(axis=-1)
        return val, sol

    # def compute_obj(self, w):
    #     w = np.concat([w, self.w[:, self.L:]], axis=-1)
    #     efficiency = self.p / w
    #     indexes = np.argsort(-efficiency, axis=-1)
    #     w_sorted = np.take_along_axis(w, indexes, axis=-1)
    #     cs = np.cumsum(w_sorted, axis=-1)
    #     sol = (cs < self.c[:, np.newaxis])
    #     reversed_indexes = np.argsort(indexes, axis=-1)  # This gives you the reverse mapping
    #     # sol_original_order = np.take_along_axis(sol, reversed_indexes, axis=-1)
    #     # new_sol = np.zeros_like(sol)
    #     np.put_along_axis(sol, indexes, sol, axis=-1)
    #     return (w*sol)[:self.L].sum(axis=-1)

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
            idx = np.argsort(self.p[k] / self.w[k])[::-1]
            self.p[k] = self.p[k][idx]
            self.w[k] = self.w[k][idx]

    def __repr__(self):
        return 'M:' + str(self.M) + '-K:' + str(self.K)

    def __str__(self):
        return 'M:' + str(self.M) + '-K:' + str(self.K)
