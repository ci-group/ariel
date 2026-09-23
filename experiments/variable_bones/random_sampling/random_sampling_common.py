"""Shared utilities for ARIEL random morphology-space sampling."""

from __future__ import annotations

import contextlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ariel.body_phenotypes.robogen_lite.collision_validation import is_physically_valid
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    MorphologyDecoderBestFirst,
)

NUM_CPPN_INPUTS = 6
NUM_CPPN_OUTPUTS = 1 + NUM_OF_TYPES_OF_MODULES + NUM_OF_ROTATIONS + 1


@dataclass(frozen=True)
class SamplingConfig:
    seed: int
    max_modules: int
    initial_structural_mutations: int
    require_actuated: bool
    require_brick: bool


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    return np.random.default_rng(seed)


def make_id_manager() -> IdManager:
    return IdManager(
        node_start=(NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS - 1),
        innov_start=(NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS) - 1,
    )


def create_random_cppn(
    id_manager: IdManager,
    initial_structural_mutations: int = 3,
) -> Genome:
    """Create one independent CPPN using the current body+brain initialization."""
    genome = Genome.random(
        num_inputs=NUM_CPPN_INPUTS,
        num_outputs=NUM_CPPN_OUTPUTS,
        next_node_id=(NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS),
        next_innov_id=0,
    )

    for _ in range(initial_structural_mutations):
        genome.mutate(
            1.0,
            1.0,
            id_manager.get_next_innov_id,
            id_manager.get_next_node_id,
        )

    return genome


def decode_graph(genome: Genome, max_modules: int):
    """Decode a CPPN while suppressing decoder console noise."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        decoder = MorphologyDecoderBestFirst(
            cppn_genome=genome,
            max_modules=max_modules,
        )
        return decoder.decode()


def get_actuator_count(graph) -> int:
    robot = construct_mjspec_from_graph(graph)
    model = robot.spec.compile()
    return int(model.nu)


def graph_statistics(graph) -> dict[str, float | int]:
    brick_lengths_m = [
        float(node_data["length"])
        for _, node_data in graph.nodes(data=True)
        if (
            node_data["type"] == ModuleType.BRICK.name
            and "length" in node_data
        )
    ]

    num_hinges = sum(
        1
        for _, node_data in graph.nodes(data=True)
        if node_data["type"] == ModuleType.HINGE.name
    )

    if brick_lengths_m:
        arr = np.asarray(brick_lengths_m, dtype=float)
        mean_mm = float(np.mean(arr) * 1000.0)
        median_mm = float(np.median(arr) * 1000.0)
        min_mm = float(np.min(arr) * 1000.0)
        max_mm = float(np.max(arr) * 1000.0)
        std_mm = float(np.std(arr) * 1000.0)
    else:
        mean_mm = float("nan")
        median_mm = float("nan")
        min_mm = float("nan")
        max_mm = float("nan")
        std_mm = float("nan")

    return {
        "num_modules": int(graph.number_of_nodes()),
        "num_bricks": int(len(brick_lengths_m)),
        "num_hinges": int(num_hinges),
        "mean_bone_length_mm": mean_mm,
        "median_bone_length_mm": median_mm,
        "min_bone_length_mm": min_mm,
        "max_bone_length_mm": max_mm,
        "std_bone_length_mm": std_mm,
    }


def iter_bones(graph):
    brick_index = 0
    for node_id, node_data in graph.nodes(data=True):
        if (
            node_data["type"] == ModuleType.BRICK.name
            and "length" in node_data
        ):
            yield {
                "bone_index": brick_index,
                "node_id": int(node_id),
                "length_mm": float(node_data["length"]) * 1000.0,
            }
            brick_index += 1


def graph_to_jsonable(graph) -> dict:
    return {
        "nodes": [
            {"id": int(node_id), **dict(node_data)}
            for node_id, node_data in graph.nodes(data=True)
        ],
        "edges": [
            {"parent": int(parent), "child": int(child), **dict(edge_data)}
            for parent, child, edge_data in graph.edges(data=True)
        ],
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
