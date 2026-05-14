"""EAOperation factories for drone morphology evolution.

Each function is decorated with @EAOperation so it can be configured and
composed into ARIEL's EA pipeline. All operations are drone-specific;
standard survivor-selection and tagging are included for convenience.

Typical pipeline (mu+lambda with NEAT crossover):
    pre-loop:
        init_op  = initialize_drones(template_handler=handler)
        eval_op  = evaluate_drones(fitness_mode="pure_hover", n_workers=N)
        initial_pop = eval_op(init_op(initial_pop))

    per generation (passed to EA.operations):
        parent_tag(n=POP_SIZE)
        crossover_drones(template_handler=handler)
        mutate_drones(template_handler=handler)
        evaluate_drones(fitness_mode="pure_hover", n_workers=N)
        truncation_select(n=POP_SIZE)
"""

import multiprocessing as mp
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ariel.body_phenotypes.drone.genome import deserialize_genome, serialize_genome
from ariel.ec.ea import EAOperation
from ariel.ec.individual import Individual
from ariel.ec.population import Population

if TYPE_CHECKING:
    from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
        SphericalAngularDroneGenomeHandler,
    )


# ---------------------------------------------------------------------------
# Multiprocessing worker — must be a module-level function for pickling
# ---------------------------------------------------------------------------

def _drone_eval_worker(args: tuple[dict[str, Any], str]) -> float:
    """Evaluate a single drone genome; called inside a Pool worker process."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    genotype_dict, fitness_mode = args

    from ariel.body_phenotypes.drone.genome import deserialize_genome as _deser

    genome = _deser(genotype_dict)
    valid_mask = ~np.isnan(genome.arms[:, 0])
    phenotype = genome.arms[valid_mask]

    if fitness_mode == "pure_hover":
        from airevolve.evolution_tools.evaluators.hover_fitness import (
            continuous_hover_fitness,
        )
        return continuous_hover_fitness(phenotype)

    from airevolve.evolution_tools.evaluators.unified_fitness import UnifiedFitness
    return UnifiedFitness(fitness_mode=fitness_mode)(phenotype, log_dir=None)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

@EAOperation
def initialize_drones(
    population: Population,
    template_handler: "SphericalAngularDroneGenomeHandler",
) -> Population:
    """Assign random drone genomes to all uninitialized individuals.

    Shares initial innovation IDs across the whole population so that
    NEAT crossover can align genes from generation 0.
    """
    shared_innos = np.arange(template_handler.max_narms, dtype=int)
    for ind in population:
        if ind.requires_init:
            genome = template_handler._generate_random_genome(
                innovation_ids=shared_innos.copy(),
            )
            ind.genotype = serialize_genome(genome)
    return population


@EAOperation
def parent_tag(
    population: Population,
    n: int,
) -> Population:
    """Tag the top-n alive evaluated individuals with ``ps=True`` for crossover.

    All other alive individuals have ``ps`` cleared to False.
    """
    alive_eval = population.alive.evaluated
    ranked = alive_eval.sort(sort="max", attribute="fitness_")
    top_ids = {ind.id for ind in ranked[:n]}
    for ind in population.alive:
        ind.tags["ps"] = ind.id in top_ids
    return population


@EAOperation
def crossover_drones(
    population: Population,
    template_handler: "SphericalAngularDroneGenomeHandler",
) -> Population:
    """Produce one offspring per ps-tagged pair via NEAT crossover.

    Reads ``ind.tags["ps"] == True`` to identify parents. Pairs them in
    random order and appends one child per pair as a new Individual with
    ``requires_eval=True``. Innovation-ID alignment from airevolve's
    SphericalAngularDroneGenomeHandler.crossover() is preserved, with the
    fitter parent's disjoint genes taking priority.
    """
    parents = population.where(lambda ind: bool(ind.tags.get("ps", False))).shuffle()
    for i in range(0, len(parents) - 1, 2):
        pa, pb = parents[i], parents[i + 1]

        ga = deserialize_genome(pa.genotype)
        gb = deserialize_genome(pb.genotype)

        ha = template_handler.copy()
        ha.genome = ga
        ha.fitness = pa.fitness_ or 0.0

        hb = template_handler.copy()
        hb.genome = gb
        hb.fitness = pb.fitness_ or 0.0

        child_handler = ha.crossover(hb)

        child = Individual()
        child.genotype = serialize_genome(child_handler.genome)
        population.append(child)
    return population


@EAOperation
def mutate_drones(
    population: Population,
    template_handler: "SphericalAngularDroneGenomeHandler",
) -> Population:
    """Mutate all alive unevaluated individuals in place.

    Targets freshly created offspring (from crossover_drones or any other
    source) without touching already-evaluated parents.
    """
    for ind in population.alive.unevaluated:
        genome = deserialize_genome(ind.genotype)
        h = template_handler.copy()
        h.genome = genome
        h.mutate()
        ind.genotype = serialize_genome(h.genome)
    return population


@EAOperation
def evaluate_drones(
    population: Population,
    fitness_mode: str = "pure_hover",
    n_workers: int = 1,
) -> Population:
    """Evaluate all unevaluated drone individuals using airevolve's physics.

    For ``fitness_mode="pure_hover"``, calls ``continuous_hover_fitness``
    directly (analytical, microseconds per individual). For other modes,
    delegates to ``UnifiedFitness`` which supports ``"gate"``,
    ``"edit_distance"``, and ``"zero"``.

    A ``multiprocessing.Pool`` of ``n_workers`` processes is spun up and
    closed within each call. Workers set BLAS thread counts to 1 to avoid
    oversubscription on multi-core machines.
    """
    unevaluated = list(population.unevaluated)
    if not unevaluated:
        return population

    eval_args = [(ind.genotype, fitness_mode) for ind in unevaluated]

    if n_workers == 1:
        fitnesses = [_drone_eval_worker(a) for a in eval_args]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            fitnesses = pool.map(_drone_eval_worker, eval_args)

    for ind, fitness in zip(unevaluated, fitnesses, strict=True):
        ind.fitness = fitness

    return population


@EAOperation
def truncation_select(
    population: Population,
    n: int,
) -> Population:
    """Keep the top-n fittest alive individuals; mark the rest as dead."""
    alive = population.alive.evaluated
    ranked = alive.sort(sort="max", attribute="fitness_")
    for i, ind in enumerate(ranked):
        if i >= n:
            ind.alive = False
    return population
