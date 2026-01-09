import time

import numpy as np

from KP_Solver.knap_cpp import KnapCpp
from SKWP.skwp_instance import SKWP_instance



class GA_SKWP:
    def __init__(self, instance: SKWP_instance, pop_size, method='e'):

        self.instance = instance
        self.pop_size = pop_size
        self.off_size = pop_size // 2
        self.population = None
        self.fitness = np.ones(self.pop_size + self.off_size) * (-1e4)
        self.solver = KnapCpp(self.instance, pop_size=self.off_size)
        self.best_val = None
        self.best_solution = None
        self.avg_fitness = None
        self.time = None

        self.solver_fun = self.solver.solve_skwp if method == 'e' else self.solver.solve_skwp_greedy

    def run(self, iterations, init_population=None, verbose=False):
        if init_population is None:
            self.population = np.random.uniform(1e-6, self.instance.max_w, size=(self.pop_size + self.off_size, self.instance.L))
        else:
            self.population = np.zeros((self.pop_size + self.off_size, self.instance.L))
            self.population[:self.pop_size] = init_population

        self.time = time.time()
        self.fitness[:self.off_size], _ = self.solver_fun(self.population[:self.off_size])
        self.fitness[self.off_size: self.off_size * 2], _ = self.solver_fun(self.population[self.off_size: self.off_size * 2])
        percentage = self.instance.max_w * 0.5
        # Sort indices by fitness (descending)
        indices = np.argsort(self.fitness)[::-1]
        for gen in range(iterations):
        #     # Evaluate fitness (sum of genes as simple fitness)
        #
        #
        #     # Generate children for remaining half
        #     for i in range(self.off_size):
        #         # Select two random parents from best individuals
        #         parent_indices = np.random.choice(indices[:self.pop_size], 2, replace=False)
        #         parent1 = self.population[parent_indices[0]]
        #         parent2 = self.population[parent_indices[1]]
        #
        #         # Single-point crossover (50% from each parent)
        #         crossover = np.array([True if np.random.uniform() < 0.5 else False for _ in range(self.population.shape[1])])
        #         self.population[indices[self.pop_size + i]] = parent1 * crossover + parent2 * (1 - crossover)
        #
        #         # Add small mutation (5% chance per gene)
        #         mutation_mask = np.random.random(self.population.shape[1]) < 0.02
        #         delta = np.random.uniform(-percentage, percentage, size=self.population.shape[1])
        #         mutation_values = self.population[indices[self.pop_size + i]] + delta
        #         self.population[indices[self.pop_size + i]][mutation_mask] = mutation_values[mutation_mask]
        #         self.population[indices[self.pop_size + i]] = (
        #             np.clip(self.population[indices[self.pop_size + i]], 1e-6, self.instance.max_w))


            parents_idxs = np.random.choice(indices[:self.pop_size], (self.off_size, 2))
            mask = np.random.random((self.off_size, self.instance.L)) > 0.5
            self.population[indices[self.pop_size:]] = (
                    self.population[parents_idxs[:, 0]] * mask + self.population[parents_idxs[:, 1]] * (~mask))

            mask = np.random.random((self.off_size, self.instance.L)) < 0.02
            delta = np.random.uniform(-percentage, percentage, size=self.population.shape[1])
            self.population[indices[self.pop_size:]] += delta * mask
            self.population[indices[self.pop_size:]] = np.clip(self.population[indices[self.pop_size:]], 1e-6, self.instance.max_w)

            self.fitness[indices[self.pop_size:]], _ = self.solver_fun(self.population[indices[self.pop_size:]])
            indices = np.argsort(self.fitness)[::-1]

            # Print progress
            self.best_val = self.fitness[indices[0]]
            self.avg_fitness = np.mean(self.fitness)
            if verbose and (gen % 50 == 0 or gen == iterations - 1):
                print(f"Gen {gen}: Best={self.best_val:.2f}, Avg={self.avg_fitness:.2f}")
        self.time = time.time() - self.time
        self.best_solution = self.population[indices[0]]


