"""Saving and loading evolved controllers.

Genomes are written both as JSON, which is readable and round-trips through
:func:`genome_from_dict`, and as an NPZ archive of pickled objects for fast
reloading alongside the fitness and identity arrays.
"""

# Standard library
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party libraries
import numpy as np

# Local libraries
from ariel import log
from ariel.spatial_ea.hyperneat import CPPNConnection, CPPNGenome, CPPNNode

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.spatial_ea.config import SpatialEAConfig
    from ariel.spatial_ea.individual import SpatialIndividual


def genome_to_dict(genotype: CPPNGenome) -> dict[str, Any]:
    """Convert a CPPN genome into JSON-serialisable form.

    Parameters
    ----------
    genotype
        The genome to convert.

    Returns
    -------
        A mapping of plain lists and dictionaries.
    """
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "activation": node.activation,
                "layer": node.layer,
            }
            for node in genotype.get("nodes", [])
        ],
        "connections": [
            {
                "from_node": conn.from_node,
                "to_node": conn.to_node,
                "weight": float(conn.weight),
                "enabled": bool(conn.enabled),
            }
            for conn in genotype.get("connections", [])
        ],
    }


def genome_from_dict(payload: dict[str, Any]) -> CPPNGenome:
    """Rebuild a CPPN genome from its serialised form.

    Parameters
    ----------
    payload
        A mapping produced by :func:`genome_to_dict`.

    Returns
    -------
        The reconstructed genome.
    """
    return {
        "nodes": [
            CPPNNode(
                node_id=int(node["node_id"]),
                activation=str(node["activation"]),
                layer=int(node["layer"]),
            )
            for node in payload.get("nodes", [])
        ],
        "connections": [
            CPPNConnection(
                from_node=int(conn["from_node"]),
                to_node=int(conn["to_node"]),
                weight=float(conn["weight"]),
                enabled=bool(conn["enabled"]),
            )
            for conn in payload.get("connections", [])
        ],
    }


def _describe_best(
    best: SpatialIndividual,
    generation: int,
    num_joints: int,
) -> str:
    """Render a human-readable description of the best individual.

    Parameters
    ----------
    best
        The individual to describe.
    generation
        Generation the run finished on.
    num_joints
        Number of actuated joints per robot.

    Returns
    -------
        The description, ready to write to a text file.
    """
    nodes = best.genotype.get("nodes", [])
    connections = best.genotype.get("connections", [])

    lines = [
        "Best evolved controller",
        "=" * 60,
        f"Individual ID:   {best.unique_id}",
        f"Born generation: {best.generation}",
        f"Final generation:{generation}",
        f"Fitness:         {best.fitness:.6f}",
        f"Energy:          {best.energy:.2f}",
        f"Joints:          {num_joints}",
        "",
        f"Nodes:       {len(nodes)}",
        f"Connections: {len(connections)} "
        f"({sum(1 for c in connections if c.enabled)} enabled)",
        "",
        "Nodes",
        "-" * 60,
    ]
    lines.extend(
        f"  id={node.node_id:<4d} layer={node.layer:<3d} "
        f"activation={node.activation}"
        for node in nodes
    )
    lines.extend(["", "Connections", "-" * 60])
    lines.extend(
        f"  {conn.from_node:>4d} -> {conn.to_node:<4d} "
        f"weight={conn.weight:+.6f} enabled={conn.enabled}"
        for conn in connections
    )

    return "\n".join(lines) + "\n"


def save_final_controllers(
    population: list[SpatialIndividual],
    config: SpatialEAConfig,
    generation: int,
    num_joints: int,
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Write the final population's controllers to disk.

    Parameters
    ----------
    population
        The final population.
    config
        Run configuration; supplies the output folder and run metadata.
    generation
        Generation the run finished on.
    num_joints
        Number of actuated joints per robot.
    timestamp
        Filename timestamp. Defaults to the current time.

    Returns
    -------
        Paths of the written files, keyed ``json``, ``npz`` and ``best``. Empty
        when the population is empty.
    """
    if not population:
        log.warning("No survivors to save; skipping controller export")
        return {}

    stamp = timestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    folder = Path(config.result_folder)
    folder.mkdir(parents=True, exist_ok=True)

    json_path = folder / f"final_controllers_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "generation": generation,
        "num_joints": num_joints,
        "population_size": len(population),
        "config": {
            "selection_method": config.selection_method,
            "pairing_method": config.pairing_method,
            "movement_bias": config.movement_bias,
            "enable_energy": config.enable_energy,
            "mating_energy_effect": (
                config.mating_energy_effect if config.enable_energy else None
            ),
            "use_periodic_boundaries": config.use_periodic_boundaries,
            "use_directional_fitness": config.use_directional_fitness,
        },
        "controllers": [
            {
                "unique_id": individual.unique_id,
                "generation": individual.generation,
                "age": individual.age_at(generation),
                "fitness": individual.fitness,
                "energy": individual.energy,
                "parent_ids": individual.parent_ids,
                "spawn_position": (
                    individual.spawn_position.tolist()
                    if individual.spawn_position is not None
                    else None
                ),
                "genotype": genome_to_dict(individual.genotype),
            }
            for individual in population
        ],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    npz_path = folder / f"final_genotypes_{stamp}.npz"
    np.savez(
        npz_path,
        genotypes=np.array(
            [individual.genotype for individual in population],
            dtype=object,
        ),
        fitness=np.array(
            [individual.fitness for individual in population],
            dtype=float,
        ),
        ages=np.array(
            [individual.age_at(generation) for individual in population],
            dtype=int,
        ),
        ids=np.array(
            [
                individual.unique_id if individual.unique_id is not None else -1
                for individual in population
            ],
            dtype=int,
        ),
        energy=np.array(
            [individual.energy for individual in population],
            dtype=float,
        ),
        num_joints=num_joints,
        generation=generation,
        allow_pickle=True,
    )

    best = max(population, key=lambda individual: individual.fitness)
    best_path = folder / f"best_controller_{stamp}.txt"
    best_path.write_text(
        _describe_best(best, generation, num_joints),
        encoding="utf-8",
    )

    msg = (
        f"Saved {len(population)} controllers to {folder}; "
        f"best fitness {best.fitness:.6f} (id {best.unique_id})"
    )
    log.info(msg)

    return {"json": json_path, "npz": npz_path, "best": best_path}


def load_controllers_from_json(json_path: str | Path) -> dict[str, Any]:
    """Read back a saved controller export.

    Genomes are rebuilt into live :class:`CPPNNode` and
    :class:`CPPNConnection` objects, so the result can be fed straight back
    into a :class:`ariel.spatial_ea.hyperneat.CPPN`.

    Parameters
    ----------
    json_path
        Path to the JSON file.

    Returns
    -------
        The saved payload, with every ``genotype`` reconstructed.
    """
    with Path(json_path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    for controller in payload.get("controllers", []):
        controller["genotype"] = genome_from_dict(controller["genotype"])

    return payload


def load_genotypes_from_npz(npz_path: str | Path) -> dict[str, Any]:
    """Read back a saved genotype archive.

    Parameters
    ----------
    npz_path
        Path to the NPZ file.

    Returns
    -------
        The genotype, fitness, age, identity and energy arrays plus the joint
        count and final generation.
    """
    data = np.load(npz_path, allow_pickle=True)
    return {
        "genotypes": data["genotypes"],
        "fitness": data["fitness"],
        "ages": data["ages"],
        "ids": data["ids"],
        "energy": data.get("energy", None),
        "num_joints": int(data["num_joints"]),
        "generation": int(data["generation"]),
    }
