
import random
import mujoco as mj
from deap import base, tools

from scripts_A3.creator_toolbox_prep import ensure_deap_types, register_factories
from scripts_A3.functions import crossover_robot_body
def EA_body(init_func, 
            evaluate_body_func, 
            *evaluate_body_args
            ):
    _, ind_type = ensure_deap_types(maximize=True)
    toolbox = base.Toolbox()
    
    register_factories(
        t=              toolbox,
        ind_type=       ind_type,
        init_func=      init_func,
        t_attr_name=    "attr_float",
        t_ind_name=     "create_body_genome",
        t_pop_name=     "create_body_genome_pop",
        no_alleles=     64,
    )
    # TO-DO once body eval function is created
    toolbox.register(
        "EvaluateBody",
        evaluate_body_func,
        *evaluate_body_args
        ...
    )
    toolbox.register(
        "BodyMate",
        crossover_robot_body,
        alpha = 0.4
    )
    toolbox.register(
        "BodyMutate",
        tools.mutGaussian,
        mu = 0, sigma = 1, indpb = 0.2
    )
    toolbox.register(
        "BodyParentSelect",
        tools.selTournament,
        tournsize = 3,
        k = 100
    )
    toolbox.register(
        "BodySurvivalSelect",
        tools.selBest,
        k = 100
    )
    NGEN = 100
    CXPB = 0.5
    MUTPB = 0.5
    pop_body_genotype = toolbox.create_body_genome_pop(n = 100)
    f_pop_body_genotype = toolbox.map(toolbox.EvaluateBody, pop_body_genotype)
    for ind, f in zip(pop_body_genotype, f_pop_body_genotype):
        ind.fitness.values = f
    
    for g in range(NGEN):
        # Parent select and cloning
        offspring = toolbox.BodyParentSelect(pop_body_genotype)
        offspring = map(toolbox.clone, offspring)
        
        # Variation operations
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.BodyMate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.BodyMutate(mutant)
                del mutant.fitness.values
        
        # Survival selection (invalids are references to offspring!)
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        f_invalids = toolbox.map(toolbox.EvaluateBody, invalids)
        for ind, f in zip(invalids, f_invalids):
            ind.fitness.values = f
        
        pop_body_genotype[:] = offspring
        
    
    
