import numpy as np

def dominates(a, b):
    """
    Check if solution a dominates solution b.
    a, b = (accuracy, fairness)
    """
    return (a[0] >= b[0] and a[1] >= b[1]) and (a[0] > b[0] or a[1] > b[1])


def pareto_front(solutions):
    """
    Extract Pareto front from a list of solutions.

    Each solution: (accuracy, fairness, metadata)
    """
    front = []

    for i, sol_i in enumerate(solutions):
        dominated = False
        for j, sol_j in enumerate(solutions):
            if i != j and dominates(sol_j[:2], sol_i[:2]):
                dominated = True
                break

        if not dominated:
            front.append(sol_i)

    return front


def crowding_distance(front):
    """
    Diversity metric (optional, NSGA-II style)
    """
    if len(front) == 0:
        return []

    distances = np.zeros(len(front))

    for m in range(2):  # objectives: accuracy, fairness
        values = [f[m] for f in front]
        sorted_idx = np.argsort(values)

        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = float('inf')

        for i in range(1, len(front) - 1):
            distances[sorted_idx[i]] += (
                values[sorted_idx[i+1]] - values[sorted_idx[i-1]]
            )

    return distances