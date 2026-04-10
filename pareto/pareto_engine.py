import numpy as np
import random
from pareto.pareto_utils import pareto_front


class ParetoEngine:
    def __init__(self, evaluate_fn, k=4, pop_size=20, iters=5, mutation_rate=0.2):
        """
        Args:
            evaluate_fn: function(indices, subgroup) -> (accuracy, fairness)
            k: number of demos
            pop_size: population size
            iters: generations
        """
        self.evaluate_fn = evaluate_fn
        self.k = k
        self.pop_size = pop_size
        self.iters = iters
        self.mutation_rate = mutation_rate

    def init_population(self, n):
        population = []
        for _ in range(self.pop_size):
            indices = np.random.choice(n, size=min(self.k, n), replace=False)
            population.append(indices)
        return population

    def mutate(self, individual, n):
        child = individual.copy()
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(child) - 1)
            child[idx] = random.randint(0, n - 1)
        return child

    def evolve(self, subgroup):
        n = len(subgroup)
        if n == 0:
            return []

        population = self.init_population(n)

        for gen in range(self.iters):
            evaluated = []

            for individual in population:
                acc, fair = self.evaluate_fn(individual, subgroup)
                evaluated.append((acc, fair, individual))

            front = pareto_front(evaluated)

            # Keep Pareto front
            new_population = [sol[2] for sol in front]

            # Refill
            while len(new_population) < self.pop_size:
                parent = random.choice(new_population)
                child = self.mutate(parent, n)
                new_population.append(child)

            population = new_population

            print(f"[Pareto] Gen {gen}: front size = {len(front)}")

        # Final front
        final_eval = []
        for individual in population:
            acc, fair = self.evaluate_fn(individual, subgroup)
            final_eval.append((acc, fair, individual))

        return pareto_front(final_eval)