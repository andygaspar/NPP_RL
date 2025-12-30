import numpy as np

from SKPP.skpp_solver import MKP_pop, MKP_greedy_pop
from SKPP.skpp_instance import SPKK_instance

'''
methods: 'exact', 'greedy'
'''


class GA_MBK:
    def __init__(self, instance: SPKK_instance, pop_size, generations, method):

        self.instance = instance
        self.pop_size = pop_size
        self.off_size = pop_size // 2
        self.population = np.random.uniform(0, instance.p[0, : instance.L], size=(self.pop_size + self.off_size, instance.L))
        self.fitness = np.ones(self.pop_size + self.off_size) * (-1e4)
        self.generations = generations
        self.solver = MKP_pop(self.instance, self.off_size) if method == "exact" else MKP_greedy_pop(self.instance, self.off_size)
        self.best_fitness = None
        self.best_solution = None
        self.avg_fitness = None

    def solve(self):
        self.fitness[:self.off_size] = self.solver.solve(self.population[:self.off_size])
        self.fitness[self.off_size: self.off_size * 2] = self.solver.solve(self.population[self.off_size: self.off_size * 2])

        # Sort indices by fitness (descending)
        indices = np.argsort(self.fitness)[::-1]
        for gen in range(self.generations):
            # Evaluate fitness (sum of genes as simple fitness)


            # Generate children for remaining half
            for i in range(self.off_size):
                # Select two random parents from best individuals
                parent_indices = np.random.choice(indices[:self.pop_size], 2, replace=False)
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]

                # Single-point crossover (50% from each parent)
                crossover = np.array([True if np.random.uniform() < 0.5 else False for _ in range(self.population.shape[1])])
                self.population[indices[self.pop_size + i]] = parent1 * crossover + parent2 * (1 - crossover)

                # Add small mutation (5% chance per gene)
                mutation_mask = np.random.random(self.population.shape[1]) < 0.02
                percentage = self.instance.p[0, : self.instance.L] * 0.05
                delta = np.random.uniform(-percentage, percentage)
                mutation_values = self.population[indices[self.pop_size + i]] + delta
                self.population[indices[self.pop_size + i]][mutation_mask] += mutation_values[mutation_mask]
                self.population[indices[self.pop_size + i]] = np.clip(self.population[indices[self.pop_size + i]], 0, self.population[indices[self.pop_size + i]])  # Keep within bounds

            self.fitness[indices[self.pop_size:]] = self.solver.solve(self.population[indices[self.pop_size:]])
            indices = np.argsort(self.fitness)[::-1]

            # Print progress
            self.best_fitness = self.fitness[indices[0]]
            self.avg_fitness = np.mean(self.fitness)
            if gen % 50 == 0 or gen == self.generations - 1:
                print(f"Gen {gen}: Best={self.best_fitness:.2f}, Avg={self.avg_fitness:.2f}")

        self.best_solution = self.population[indices[0]]


