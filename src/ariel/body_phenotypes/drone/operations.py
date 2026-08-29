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
import random
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ariel.body_phenotypes.drone.genome import (
    deserialize_cppn_genome,
    deserialize_genome,
    serialize_cppn_genome,
    serialize_genome,
)
from ariel.ec.ea import EAOperation
from ariel.ec.individual import Individual
from ariel.ec.population import Population

if TYPE_CHECKING:
    from ariel.ec.drone.genome_handlers.cppn_neat_genome_handler import (
        CPPNNeatDroneGenomeHandler,
    )
    from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
        SphericalAngularDroneGenomeHandler,
    )

DecoderName = Literal["spherical", "cppn"]


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
        from ariel.ec.drone.evaluators.hover_fitness import (
            continuous_hover_fitness,
        )
        return continuous_hover_fitness(phenotype)

    from ariel.ec.drone.evaluators.unified_fitness import UnifiedFitness
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


# ---------------------------------------------------------------------------
# CPPN-NEAT EAOperation factories — mirror the spherical-angular set above
# ---------------------------------------------------------------------------


@EAOperation
def initialize_cppn_drones(
    population: Population,
    template_handler: "CPPNNeatDroneGenomeHandler",
) -> Population:
    """Assign random CPPN genomes to all uninitialised individuals.

    Each individual gets its own freshly generated network from the
    template handler. Innovation IDs are managed by the class-level
    ``_innovation_counter`` shared across all CPPN drone handlers so
    that NEAT structural mutations align across generations.
    """
    for ind in population:
        if ind.requires_init:
            child = template_handler.copy()
            child.genome = template_handler._generate_random_genome()
            ind.genotype = serialize_cppn_genome(child.genome)
    return population


@EAOperation
def crossover_cppn_drones(
    population: Population,
    template_handler: "CPPNNeatDroneGenomeHandler",
) -> Population:
    """Produce one offspring per ps-tagged pair via NEAT crossover.

    Parents must already be tagged with ``ind.tags["ps"] == True`` (use
    :func:`parent_tag`). Pairs them in random order and appends one child
    per pair. NEAT alignment is provided by ``CPPNNetwork`` innovation
    numbers; the fitter parent's disjoint genes are inherited preferentially.
    """
    parents = population.where(lambda ind: bool(ind.tags.get("ps", False))).shuffle()
    for i in range(0, len(parents) - 1, 2):
        pa, pb = parents[i], parents[i + 1]

        ha = template_handler.copy()
        ha.genome = deserialize_cppn_genome(pa.genotype)
        ha.fitness = pa.fitness_ or 0.0

        hb = template_handler.copy()
        hb.genome = deserialize_cppn_genome(pb.genotype)
        hb.fitness = pb.fitness_ or 0.0

        child_handler = ha.crossover(hb)

        child = Individual()
        child.genotype = serialize_cppn_genome(child_handler.genome)
        population.append(child)
    return population


@EAOperation
def mutate_cppn_drones(
    population: Population,
    template_handler: "CPPNNeatDroneGenomeHandler",
) -> Population:
    """Mutate every alive, unevaluated CPPN individual in place."""
    for ind in population.alive.unevaluated:
        h = template_handler.copy()
        h.genome = deserialize_cppn_genome(ind.genotype)
        h.mutate()
        ind.genotype = serialize_cppn_genome(h.genome)
    return population


# ---------------------------------------------------------------------------
# Encoding-agnostic MuJoCo hover evaluator
# ---------------------------------------------------------------------------


def _decode_genotype_to_blueprint(
    genotype: dict[str, Any],
    decoder: DecoderName,
    decoder_kwargs: dict[str, Any] | None,
):
    """Worker-side dispatch: stored genotype → DroneBlueprint.

    Returns ``None`` for invalid morphologies (e.g. empty arms / failed
    CPPN decode); callers translate this to a sentinel fitness.
    """
    from ariel.body_phenotypes.drone.decoders import (
        spherical_angular_to_blueprint,
    )

    decoder_kwargs = dict(decoder_kwargs or {})

    if decoder == "spherical":
        genome = deserialize_genome(genotype)
        valid_mask = ~np.isnan(genome.arms[:, 0])
        if not valid_mask.any():
            return None
        # ``spherical_angular_to_blueprint`` ignores NaN rows itself.
        propsize = int(decoder_kwargs.pop("propsize", 2))
        return spherical_angular_to_blueprint(
            genome.arms,
            propsize=propsize,
            **decoder_kwargs,
        )

    if decoder == "cppn":
        from ariel.ec.drone.genome_handlers.cppn_neat_genome_handler import (
            CPPNNeatDroneGenomeHandler,
        )

        handler_kwargs = decoder_kwargs.pop("handler_kwargs", {}) or {}
        propsize = int(decoder_kwargs.pop("propsize", 2))
        net = deserialize_cppn_genome(genotype)
        handler = CPPNNeatDroneGenomeHandler(genome=net, **handler_kwargs)
        phenotype = handler.get_phenotype()
        valid_mask = ~np.isnan(phenotype[:, 0])
        if not valid_mask.any():
            return None
        return spherical_angular_to_blueprint(
            phenotype,
            propsize=propsize,
            **decoder_kwargs,
        )

    msg = f"Unknown decoder: {decoder!r} (expected 'spherical' or 'cppn')"
    raise ValueError(msg)


def _mujoco_hover_eval_worker(
    args: tuple[
        dict[str, Any],
        DecoderName,
        dict[str, Any] | None,
        float,
        float,
        tuple[float, float, float],
        dict[str, float],
    ],
) -> float:
    """Single-individual hover evaluation. Picklable (module-level)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    (
        genotype,
        decoder,
        decoder_kwargs,
        duration,
        warm_up,
        target_position,
        weights,
    ) = args

    try:
        bp = _decode_genotype_to_blueprint(genotype, decoder, decoder_kwargs)
    except Exception:
        return float("inf")
    if bp is None:
        return float("inf")

    try:
        from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
        from ariel.simulation.drone.controllers.lee_control.lee_controller import (
            LeeGeometricControl,
        )
        from ariel.simulation.drone.controllers.lee_control.mujoco_bridge import (
            LeeMujocoHoverBridge,
            hover_fitness_from_log,
            spawn_blueprint_in_world,
        )
        from ariel.simulation.drone.drone_interface import DroneInterface

        propellers = blueprint_to_propellers(bp, convention="ned")
        if not propellers:
            return float("inf")
        quad = DroneInterface(0, propellers=propellers)

        lee_ctrl = LeeGeometricControl(
            quad,
            yawType=1,
            orient="NED",
            auto_scale_gains=True,
            pos_P_gain=np.array([14.3, 14.3, 14.3]),
            vel_P_gain=np.array([9.0, 9.0, 9.0]),
        )

        spawned = spawn_blueprint_in_world(
            bp,
            propellers=propellers,
            target_mass=float(quad.params["mB"]),
            spawn_position=target_position,
        )

        bridge = LeeMujocoHoverBridge(
            quad=quad,
            lee_ctrl=lee_ctrl,
            model=spawned.model,
            data=spawned.data,
            max_thrust_per_motor=spawned.max_thrust_per_motor,
            target_position_enu=target_position,
        )

        log = bridge.run_hover(duration=duration, warm_up=warm_up)
        return float(
            hover_fitness_from_log(
                log,
                target_position_enu=target_position,
                **weights,
            ),
        )
    except Exception:
        return float("inf")


@EAOperation
def evaluate_drones_hover_mujoco(
    population: Population,
    decoder: DecoderName = "spherical",
    decoder_kwargs: dict[str, Any] | None = None,
    duration: float = 1.0,
    warm_up: float = 0.1,
    target_position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    n_workers: int = 1,
    drift_weight: float = 1.0,
    tilt_weight: float = 1.0,
    ctrl_weight: float = 0.05,
) -> Population:
    """Evaluate every alive, unevaluated individual via MuJoCo hover.

    Each genotype is decoded to a :class:`DroneBlueprint`, spawned in a
    fresh ``SimpleFlatWorld``, and flown with the Lee → MuJoCo hover
    bridge for ``duration`` seconds. Fitness is *lower-is-better* — set
    ``is_maximisation=False`` on your :class:`EASettings`.

    Parameters
    ----------
    decoder
        ``"spherical"`` for ``SphericalNeatGenome`` arms, ``"cppn"`` for
        :class:`CPPNNetwork` indirect encoding (decoded via
        :class:`CPPNNeatDroneGenomeHandler`).
    decoder_kwargs
        Extra kwargs passed to the blueprint decoder. For ``"cppn"``,
        must include ``handler_kwargs`` (``min_max_narms``,
        ``num_segments``, ``parameter_limits``, …) so the CPPN can be
        decoded back to a phenotype array.
    duration, warm_up
        Hover-window length and warm-up window (warm-up poses are
        discarded from fitness — Lee is *active* throughout).
    target_position
        ENU hover setpoint.
    n_workers
        ``1`` runs in-process; higher values fan out across a
        ``multiprocessing.Pool`` with the ``spawn`` start method.
    drift_weight, tilt_weight, ctrl_weight
        Forwarded to :func:`hover_fitness_from_log`.
    """
    weights = {
        "drift_weight": float(drift_weight),
        "tilt_weight": float(tilt_weight),
        "ctrl_weight": float(ctrl_weight),
    }

    to_eval = [ind for ind in population.alive.unevaluated]
    if not to_eval:
        return population

    tasks = [
        (
            ind.genotype,
            decoder,
            decoder_kwargs,
            float(duration),
            float(warm_up),
            tuple(target_position),
            weights,
        )
        for ind in to_eval
    ]

    if n_workers == 1:
        fitnesses = [_mujoco_hover_eval_worker(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(n_workers)) as pool:
            fitnesses = pool.map(_mujoco_hover_eval_worker, tasks)

    for ind, fit in zip(to_eval, fitnesses, strict=True):
        ind.fitness = float(fit)

    return population
