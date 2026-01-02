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
        self.c = self.w.sum(axis=1)//4
        self.p = self.w + 10

        self.max_w = self.w.max()

    def sort_values(self):
        for k in range(self.K):
            idx = np.argsort(self.p[k]/self.w[k])[::-1]
            self.p[k] = self.p[k][idx]
            self.w[k] = self.w[k][idx]

    def __repr__(self):
        return 'M:' + str(self.M) + '-K:' + str(self.K)

    def __str__(self):
        return 'M:' + str(self.M) + '-K:' + str(self.K)




