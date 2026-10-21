

from scipy.stats import qmc
import numpy as np




def init_lhc(n_paths, pop_size, l_b, u_b):
    lhs = qmc.LatinHypercube(d=n_paths)
    init_pop = lhs.random(pop_size)
    return qmc.scale(init_pop, l_b, u_b)

def init_sobol(n_paths, pop_size, l_b, u_b):
    sobol = qmc.Sobol(d=n_paths)
    init_pop = sobol.random(pop_size)
    return qmc.scale(init_pop, l_b, u_b)

def init_halton(n_paths, pop_size, l_b, u_b):
    halton = qmc.Halton(d=n_paths)
    init_pop = halton.random(pop_size)
    return qmc.scale(init_pop, l_b, u_b)


def init_kronecker(n_paths, pop_size, l_b, u_b):
    x0 = np.zeros(n_paths)
    alpha = _default_irrationals(n_paths)

    i = np.arange(1, pop_size + 1).reshape(-1, 1)   # n = 1, 2, ..., n
    init_pop = (x0 + i * alpha) % 1.0                # fractional part
    return qmc.scale(init_pop, l_b, u_b)

def _default_irrationals(d):
    def first_primes(k):
        primes, num = [], 2
        while len(primes) < k:
            if all(num % p for p in primes):
                primes.append(num)
            num += 1
        return primes
    return np.sqrt(first_primes(d))

# # --- check against the 1D example of the paper ---
# seq_1d = kronecker_sequence(4, 1, x0=[0.0], alpha=[(np.sqrt(5) - 1) / 2])
# print(seq_1d.ravel())
# # expected: [0.618, 0.236, 0.854, 0.472] (approximately)



