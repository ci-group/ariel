import numpy as np

def single_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-point crossover: split at random point and swap segments."""
    length = len(parent1)
    point = np.random.randint(1, length)
    offspring1 = np.concatenate([parent1[:point], parent2[point:]])
    offspring2 = np.concatenate([parent2[:point], parent1[point:]])
    return offspring1, offspring2


def two_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-point crossover: swap segment between two random points."""
    length = len(parent1)
    point1, point2 = sorted(np.random.choice(range(1, length), size=2, replace=False))
    offspring1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    offspring2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
    return offspring1, offspring2


def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray, prob: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover: each gene independently chosen from either parent."""
    mask = np.random.random(len(parent1)) < prob
    offspring1 = np.where(mask, parent1, parent2)
    offspring2 = np.where(mask, parent2, parent1)
    return offspring1, offspring2


def blend_crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Blend crossover (BLX-α): offspring values from extended range."""
    min_vals = np.minimum(parent1, parent2)
    max_vals = np.maximum(parent1, parent2)
    range_vals = max_vals - min_vals
    lower = min_vals - alpha * range_vals
    upper = max_vals + alpha * range_vals
    offspring1 = np.random.uniform(lower, upper)
    offspring2 = np.random.uniform(lower, upper)
    return offspring1, offspring2


def simulated_binary_crossover(parent1: np.ndarray, parent2: np.ndarray, eta: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX): mimics single-point crossover."""
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)
    
    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                offspring1[i] = 0.5 * ((parent1[i] + parent2[i]) - beta * abs(parent1[i] - parent2[i]))
                offspring2[i] = 0.5 * ((parent1[i] + parent2[i]) + beta * abs(parent1[i] - parent2[i]))
            else:
                offspring1[i] = parent1[i]
                offspring2[i] = parent2[i]
        else:
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]
    return offspring1, offspring2


def arithmetic_crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Arithmetic crossover: linear combination of parents."""
    if alpha is None:
        alpha = np.random.random(len(parent1))
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = (1 - alpha) * parent1 + alpha * parent2
    return offspring1, offspring2