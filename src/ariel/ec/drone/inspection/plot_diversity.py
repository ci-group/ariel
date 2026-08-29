from matplotlib import pyplot as plt
import numpy as np
from ariel.ec.drone.evaluators.edit_distance import compute_individual_population_edit_distance

def plot_diversity(ax, gens, parameter_limits=None):
    """
    Plot diversity across generations.
    
    Args:
        ax: Matplotlib axis to plot on
        gens: List of generations, each containing population data
        parameter_limits: (min_vals, max_vals) tuple for distance calculation
    """
    # Default parameter limits if not provided
    if parameter_limits is None:
        min_vals = np.array([0.09, 0, 0, 0, 0, 0])
        max_vals = np.array([0.4, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1])
    else:
        min_vals, max_vals = parameter_limits

    diversity = np.zeros(len(gens))
    for g, pop in enumerate(gens):
        novelties = np.zeros(len(pop))
        for i, ind in enumerate(pop):
            novelties[i] = compute_individual_population_edit_distance(ind, pop, min_vals, max_vals)

        diversity[g] = np.mean(novelties)

    ax.plot(diversity)
    ax.set_title('Diversity')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity')
    ax.grid()

def plot_diversity_from_amalgamated(ax, amalgamated, title='Diversity', label=None, color='b',
                                    min_max_params=np.array([[0.09,0.4], [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi], [0,1]])):
    """
    Plot diversity across generations from multiple experimental runs.
    
    Args:
        ax: Matplotlib axis to plot on
        amalgamated: List of experimental runs, each containing generations
        title: Plot title
        label: Legend label
        color: Plot color
        min_max_params: Parameter bounds array of shape (n_params, 2)
    """
    # Extract min/max values from parameter bounds
    min_vals = min_max_params[:, 0]
    max_vals = min_max_params[:, 1]

    diversities = []
    for exp in amalgamated:
        diversity = np.zeros(len(exp))
        for g, pop in enumerate(exp):
            novelties = np.zeros(len(pop))
            for i, ind in enumerate(pop):
                novelties[i] = compute_individual_population_edit_distance(ind, pop, min_vals, max_vals)

            diversity[g] = np.mean(novelties)

        diversities.append(diversity)

    mu = np.mean(diversities, axis=0)
    std = np.std(diversities, axis=0)
    
    ax.plot(mu, label=label, color=color)
    ax.fill_between(np.arange(len(mu)), mu-std, mu+std, alpha=0.2, color=color)
    ax.set_title(title)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity')
    ax.grid(True)
    if label is not None:
        ax.legend()


