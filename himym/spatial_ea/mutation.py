import numpy as np

def uniform_mutation(individual: np.ndarray, bounds: tuple, mutation_rate: float = 0.1) -> np.ndarray:
    """Uniform mutation: replace genes with random values."""
    mutant = individual.copy()
    mask = np.random.random(len(individual)) < mutation_rate
    mutant[mask] = np.random.uniform(bounds[0], bounds[1], np.sum(mask))
    return mutant


def gaussian_mutation(individual: np.ndarray, mutation_rate: float = 0.1, 
                     sigma: float = 0.1, bounds: tuple | None = None) -> np.ndarray:
    """Gaussian mutation: add random value from normal distribution."""
    mutant = individual.copy()
    mask = np.random.random(len(individual)) < mutation_rate
    mutant[mask] += np.random.normal(0, sigma, np.sum(mask))
    if bounds is not None:
        mutant = np.clip(mutant, bounds[0], bounds[1])
    return mutant


def polynomial_mutation(individual: np.ndarray, bounds: tuple, 
                       mutation_rate: float = 0.1, eta: float = 20.0) -> np.ndarray:
    """Polynomial mutation: creates small perturbations."""
    mutant = individual.copy()
    
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            u = np.random.random()
            
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
            
            mutant[i] += delta * (bounds[1] - bounds[0])
            mutant[i] = np.clip(mutant[i], bounds[0], bounds[1])
    
    return mutant


def creep_mutation(individual: np.ndarray, bounds: tuple, 
                   mutation_rate: float = 0.1, creep_rate: float = 0.05) -> np.ndarray:
    """Creep mutation: small incremental changes relative to current value."""
    mutant = individual.copy()
    mask = np.random.random(len(individual)) < mutation_rate
    perturbation = np.random.uniform(-creep_rate, creep_rate, np.sum(mask))
    mutant[mask] *= (1 + perturbation)
    mutant = np.clip(mutant, bounds[0], bounds[1])
    return mutant


def non_uniform_mutation(individual: np.ndarray, bounds: tuple, 
                        generation: int, max_generations: int,
                        mutation_rate: float = 0.1, b: float = 5.0) -> np.ndarray:
    """Non-uniform mutation: decreases perturbation as generations progress."""
    mutant = individual.copy()
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            r = np.random.random()
            t = (1 - generation / max_generations) ** b
            if np.random.random() < 0.5:
                delta = (bounds[1] - mutant[i]) * (1 - r ** t)
            else:
                delta = -(mutant[i] - bounds[0]) * (1 - r ** t)
            mutant[i] += delta
    return mutant


def boundary_mutation(individual: np.ndarray, bounds: tuple, 
                     mutation_rate: float = 0.1) -> np.ndarray:
    """Boundary mutation: replace gene with min or max boundary value."""
    mutant = individual.copy()
    mask = np.random.random(len(individual)) < mutation_rate
    boundary_values = np.where(np.random.random(np.sum(mask)) < 0.5, 
                               bounds[0], bounds[1])
    mutant[mask] = boundary_values
    return mutant


def adaptive_gaussian_mutation(individual: np.ndarray, fitness_improvement: bool,
                              mutation_rate: float = 0.1, sigma: float = 0.1,
                              bounds: tuple | None = None,
                              increase_factor: float = 1.2,
                              decrease_factor: float = 0.8) -> tuple:
    """Adaptive Gaussian mutation: adjusts sigma based on fitness improvement."""
    new_sigma = sigma * decrease_factor if fitness_improvement else sigma * increase_factor
    new_sigma = np.clip(new_sigma, 0.001, 1.0)
    mutant = gaussian_mutation(individual, mutation_rate, new_sigma, bounds)
    return mutant, new_sigma


def cauchy_mutation(individual: np.ndarray, mutation_rate: float = 0.1,
                   scale: float = 0.1, bounds: tuple | None = None) -> np.ndarray:
    """Cauchy mutation: uses Cauchy distribution with heavier tails than Gaussian."""
    mutant = individual.copy()
    mask = np.random.random(len(individual)) < mutation_rate
    cauchy_samples = np.random.standard_cauchy(np.sum(mask)) * scale
    mutant[mask] += cauchy_samples
    if bounds is not None:
        mutant = np.clip(mutant, bounds[0], bounds[1])
    return mutant