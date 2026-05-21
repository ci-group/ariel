"""End-to-end smoke tests for the e_drones_ec evaluator pipeline.

Drives one tiny generation of each encoding (spherical + CPPN-NEAT)
through ARIEL's EA pipeline using the new MuJoCo hover evaluator. The
tests don't check the *value* of fitness — only that the pipeline runs
without crashes and produces at least one individual with finite
fitness.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.timeout(120)
def test_spherical_evaluator_runs_end_to_end(tmp_path) -> None:
    from ariel.body_phenotypes.drone import (
        crossover_drones,
        evaluate_drones_hover_mujoco,
        initialize_drones,
        mutate_drones,
        parent_tag,
        truncation_select,
    )
    from ariel.ec import EA, Individual, Population
    from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
        SphericalAngularDroneGenomeHandler,
    )

    parameter_limits = np.array([
        [0.055, 0.17],
        [-np.pi, np.pi],
        [-np.pi / 2, np.pi / 2],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [0, 1],
    ])
    handler = SphericalAngularDroneGenomeHandler(
        min_max_narms=(4, 6),
        parameter_limits=parameter_limits,
        append_arm_chance=0.1,
        bilateral_plane_for_symmetry=None,
        repair=False,
        rnd=np.random.default_rng(0),
    )

    pop = Population([Individual() for _ in range(2)])
    init_op = initialize_drones(template_handler=handler)
    eval_op = evaluate_drones_hover_mujoco(
        decoder="spherical",
        decoder_kwargs={"propsize": 2},
        duration=0.4,
        warm_up=0.05,
        target_position=(0.0, 0.0, 1.0),
        n_workers=1,
    )
    pop = init_op(pop)
    pop = eval_op(pop)

    ops = [
        parent_tag(n=2),
        crossover_drones(template_handler=handler),
        mutate_drones(template_handler=handler),
        eval_op,
        truncation_select(n=2),
    ]
    ea = EA(
        population=pop,
        operations=ops,
        num_steps=1,
        db_file_path=tmp_path / "spherical.db",
        db_handling="delete",
    )
    ea.run()
    ea.fetch_population(only_alive=False, requires_eval=False)

    fits = [
        ind.fitness_ for ind in ea.population
        if ind.fitness_ is not None and np.isfinite(ind.fitness_)
    ]
    assert fits, "no individual produced a finite hover fitness"


@pytest.mark.timeout(120)
def test_cppn_evaluator_runs_end_to_end(tmp_path) -> None:
    from ariel.body_phenotypes.drone import (
        crossover_cppn_drones,
        evaluate_drones_hover_mujoco,
        initialize_cppn_drones,
        mutate_cppn_drones,
        parent_tag,
        truncation_select,
    )
    from ariel.ec import EA, Individual, Population
    from ariel.ec.drone.genome_handlers.cppn_neat_genome_handler import (
        CPPNNeatDroneGenomeHandler,
    )

    handler_kwargs = {
        "num_segments": 8,
        "min_max_narms": (4, 6),
        "parameter_limits": np.array([
            [0.055, 0.17],
            [-np.pi, np.pi],
            [-np.pi / 2, np.pi / 2],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [0, 1],
        ]),
        "repair": True,
    }
    template = CPPNNeatDroneGenomeHandler(rng=np.random.default_rng(0), **handler_kwargs)

    pop = Population([Individual() for _ in range(2)])
    init_op = initialize_cppn_drones(template_handler=template)
    eval_op = evaluate_drones_hover_mujoco(
        decoder="cppn",
        decoder_kwargs={"propsize": 2, "handler_kwargs": handler_kwargs},
        duration=0.4,
        warm_up=0.05,
        target_position=(0.0, 0.0, 1.0),
        n_workers=1,
    )
    pop = init_op(pop)
    pop = eval_op(pop)

    ops = [
        parent_tag(n=2),
        crossover_cppn_drones(template_handler=template),
        mutate_cppn_drones(template_handler=template),
        eval_op,
        truncation_select(n=2),
    ]
    ea = EA(
        population=pop,
        operations=ops,
        num_steps=1,
        db_file_path=tmp_path / "cppn.db",
        db_handling="delete",
    )
    ea.run()
    ea.fetch_population(only_alive=False, requires_eval=False)

    fits = [
        ind.fitness_ for ind in ea.population
        if ind.fitness_ is not None and np.isfinite(ind.fitness_)
    ]
    assert fits, "no CPPN individual produced a finite hover fitness"
