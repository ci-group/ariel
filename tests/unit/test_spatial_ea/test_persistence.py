"""Test: saving and loading evolved controllers."""

# Standard library
import json
from pathlib import Path

# Third-party libraries
import numpy as np

# Local libraries
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.genetics import create_initial_hyperneat_genome
from ariel.spatial_ea.hyperneat import CPPN
from ariel.spatial_ea.individual import SpatialIndividual
from ariel.spatial_ea.persistence import (
    genome_from_dict,
    genome_to_dict,
    load_controllers_from_json,
    load_genotypes_from_npz,
    save_final_controllers,
)


def _population() -> list[SpatialIndividual]:
    """Build a small saved-ready population."""
    return [
        SpatialIndividual(
            unique_id=i,
            generation=i,
            genotype=create_initial_hyperneat_genome(),
            fitness=float(i),
            energy=90.0 - i,
            spawn_position=np.array([1.0 * i, 2.0 * i, 0.1]),
        )
        for i in range(3)
    ]


def test_genome_round_trips_through_json() -> None:
    """A serialised genome should rebuild into an equivalent live one."""
    genome = create_initial_hyperneat_genome()
    restored = genome_from_dict(
        json.loads(json.dumps(genome_to_dict(genome))),
    )

    assert len(restored["nodes"]) == len(genome["nodes"])
    assert len(restored["connections"]) == len(genome["connections"])

    for original, copy_ in zip(
        genome["connections"],
        restored["connections"],
        strict=True,
    ):
        assert original.from_node == copy_.from_node
        assert original.to_node == copy_.to_node
        assert original.weight == copy_.weight
        assert original.enabled == copy_.enabled

    # The rebuilt genome is usable, and gives the same answer.
    inputs = np.array([0.3, -0.2, 0.7, 0.1])
    assert CPPN(restored).activate(inputs) == CPPN(genome).activate(inputs)


def test_save_writes_all_three_files(tmp_path: Path) -> None:
    """Saving should produce the JSON, NPZ and readable best-controller."""
    config = SpatialEAConfig(result_folder=tmp_path, save_results=False)
    paths = save_final_controllers(
        _population(),
        config,
        generation=5,
        num_joints=8,
        timestamp="20260101_000000",
    )

    assert set(paths) == {"json", "npz", "best"}
    assert all(path.exists() for path in paths.values())
    assert paths["json"].name == "final_controllers_20260101_000000.json"
    assert "Best evolved controller" in paths["best"].read_text()


def test_saved_json_carries_the_fields_analysis_needs(tmp_path: Path) -> None:
    """Each controller record needs age, fitness, energy and its genome."""
    config = SpatialEAConfig(result_folder=tmp_path, save_results=False)
    paths = save_final_controllers(
        _population(),
        config,
        generation=5,
        num_joints=8,
    )

    payload = load_controllers_from_json(paths["json"])

    assert payload["num_joints"] == 8
    assert payload["generation"] == 5
    assert len(payload["controllers"]) == 3

    first = payload["controllers"][0]
    for key in ("unique_id", "age", "fitness", "energy", "genotype"):
        assert key in first
    # Born in generation 0, saved at generation 5.
    assert first["age"] == 5
    # The loader rebuilds live genome objects.
    assert CPPN(first["genotype"]).activate(np.zeros(4)).shape == (1,)


def test_npz_round_trip(tmp_path: Path) -> None:
    """The archive should preserve genomes and their metadata."""
    config = SpatialEAConfig(result_folder=tmp_path, save_results=False)
    population = _population()
    paths = save_final_controllers(
        population,
        config,
        generation=5,
        num_joints=8,
    )

    loaded = load_genotypes_from_npz(paths["npz"])

    assert loaded["num_joints"] == 8
    assert loaded["generation"] == 5
    assert list(loaded["ids"]) == [0, 1, 2]
    assert list(loaded["fitness"]) == [0.0, 1.0, 2.0]
    assert list(loaded["ages"]) == [5, 4, 3]
    assert loaded["energy"] is not None
    assert len(loaded["genotypes"]) == 3
    assert CPPN(loaded["genotypes"][0]).activate(np.zeros(4)).shape == (1,)


def test_saving_an_extinct_population_writes_nothing(tmp_path: Path) -> None:
    """There is nothing to save when no one survived."""
    config = SpatialEAConfig(result_folder=tmp_path, save_results=False)
    paths = save_final_controllers([], config, generation=3, num_joints=8)

    assert paths == {}
    assert list(tmp_path.iterdir()) == []


def test_a_run_stamps_every_file_identically(tmp_path: Path) -> None:
    """Analysis tooling pairs result files by an exact shared timestamp.

    Stamping each file at the moment it is written orphans them whenever a
    run crosses a second boundary mid-save.
    """
    from ariel.spatial_ea.engine import SpatialEA

    config = SpatialEAConfig(
        population_size=3,
        num_generations=1,
        simulation_time=0.15,
        world_size=(4.0, 4.0),
        result_folder=tmp_path,
        save_results=True,
        save_plots=False,
        print_generation_stats=False,
    )
    SpatialEA(config=config).run(generations=1)

    stamps = {
        path.stem.split("_", 2)[-1]
        for path in tmp_path.iterdir()
        if path.stem.startswith((
            "evolution_data_",
            "final_controllers_",
            "final_genotypes_",
            "best_controller_",
        ))
    }

    assert len(stamps) == 1, f"result files disagree on timestamp: {stamps}"
