import random
import numpy as np
from functools import partial
from deap import base
from deap import creator
from deap import tools
from typing import Callable

def ensure_deap_types(maximize = True):
    """Create classes for Individuals and Fitness.

    Args:
        maximize (bool, optional): Whether individual instances created should have fitnesses 
                                   to be maximized or minimized. Defaults to True.

    Returns:
        fit_type : Class for fitness
        ind_type : Class for individual
    """
    fit_name = "FitnessMax" if maximize else "FitnessMin"
    ind_name = "IndividualMax" if maximize else "IndividualMin"
    
    if not hasattr(creator, fit_name):
        creator.create(fit_name, base.Fitness, weights = (1.0,) if maximize else (-1.0,))
        
    if not hasattr(creator, ind_name):
        creator.create(ind_name, list, fitness = getattr(creator, fit_name))
    fit_type = getattr(creator, fit_name)
    ind_type = getattr(creator, ind_name)
    
    return fit_type, ind_type    


def register_factories(
    t :             base.Toolbox,
    ind_type :      type,
    init_func :     Callable[..., float],
    t_attr_name :   str,
    t_ind_name :    str,
    t_pop_name :    str,
    no_alleles :    int = 10
    ) -> None:
    """
    Creates DEAP toolbox function to generate an allele, an individual, and a population. The distribution 
    to sample the value of each allele from can be specified. Alleles are assumed to be floats. 
    Args:
        t :             DEAP toolbox,
        ind_type :      The individual class/type, 
        init_func :     The function to sample alleles from
        t_attr_name :   The name for the sampling function to be set in toolbox
        t_ind_name :    The name for the individual generating function to be set in the toolbox
        t_pop_name :    The name for the population generating function to be set in the toolbox
        no_alleles :    The number of alleles per individual
    """

    if not hasattr(t, t_attr_name):
        t.register(t_attr_name, init_func)
    if not hasattr(t, t_ind_name):
        t.register(t_ind_name, tools.initRepeat, ind_type, getattr(t, t_attr_name), n=no_alleles)
    if not hasattr(t, t_pop_name):
        t.register(t_pop_name, tools.initRepeat, list, getattr(t, t_ind_name))
        
if __name__=="__main__":
    
    # Ensure that classes for individual instances are there
    _, ind_type = ensure_deap_types(maximize= False)
    toolbox = base.Toolbox()
    
    # We register two functions, one for generating body populations, and 
    # one for generating brain populations. Let's say they differ in 
    # distributions for alleles and number of alleles
    
    # Prefil function arguments, to make them zero-argument callables
    func_gauss = partial(random.gauss, 0.0, 1.0) 
    func_uniform = partial(random.uniform, -2, 2)
    
    # Make the functions (factories) for generating individuals
    register_factories(
        toolbox, ind_type, func_gauss, 
        "attr_float", "IndividualBrain", "PopulationBrain", 10)
    register_factories(
        toolbox, ind_type, func_uniform, 
        "attr_float", "IndividualBody", "PopulationBody", 5)
    
    # Generate populations
    pop_brain = toolbox.PopulationBrain(n = 100)
    pop_body = toolbox.PopulationBody(n = 100)
    print(f"An individual from the brain population has length {len(pop_brain[0])}:\n {pop_brain[0]} \n")
    print(f"An individual from the body population has length {len(pop_body[0])}:\n {pop_body[0]}")

    
