"""Spatial evolutionary algorithm for ARIEL.

Robots with HyperNEAT-encoded controllers share one toroidal world and
reproduce by physical proximity, so spatial structure in the population is
emergent rather than imposed.

Notes
-----
    * Plotting lives in :mod:`ariel.spatial_ea.visualization` and is not
      re-exported here, so importing this package does not pull in
      ``matplotlib.pyplot``. Import that module directly when you need it, or
      set ``save_plots`` / ``save_generation_plots`` on the configuration and
      let the engine call it.

"""

from ariel.spatial_ea.bootstrap import SpatialBootstrap, build_default_bootstrap
from ariel.spatial_ea.clustering import (
    ClusteringResult,
    analyze_spatial_clustering,
    cluster_dbscan,
    cluster_hierarchical,
    cluster_kmeans,
    find_optimal_clusters,
    spatial_silhouette,
)
from ariel.spatial_ea.config import SpatialEAConfig, spatial_ea_config
from ariel.spatial_ea.data import EvolutionDataCollector
from ariel.spatial_ea.engine import SpatialEA
from ariel.spatial_ea.experiment import (
    AggregatedResults,
    ExperimentRunner,
    ExperimentSpec,
    RunResult,
    run_trial,
)
from ariel.spatial_ea.genetics import (
    clone_individual,
    create_initial_hyperneat_genome,
    crossover_genomes,
    crossover_hyperneat,
    mutate_genome,
    mutate_hyperneat,
)
from ariel.spatial_ea.genotype_distance import (
    compute_genotype_diversity,
    compute_pairwise_distance_matrix,
)
from ariel.spatial_ea.hyperneat import (
    CPPN,
    CPPNConnection,
    CPPNNode,
    SubstrateNetwork,
    create_substrate_for_gecko,
)
from ariel.spatial_ea.incubation import (
    IncubationEvolution,
    seed_spatial_population_from_incubation,
)
from ariel.spatial_ea.individual import SpatialIndividual
from ariel.spatial_ea.movement import (
    MatingController,
    run_mating_movement_phase,
)
from ariel.spatial_ea.persistence import (
    load_controllers_from_json,
    load_genotypes_from_npz,
    save_final_controllers,
)
from ariel.spatial_ea.recording import GenerationRecorder
from ariel.spatial_ea.selection import select_individuals
from ariel.spatial_ea.world import (
    SpawnedPopulation,
    generate_spawn_positions,
    spawn_population_in_world,
)

__all__: list[str] = [
    "CPPN",
    "AggregatedResults",
    "CPPNConnection",
    "CPPNNode",
    "ClusteringResult",
    "EvolutionDataCollector",
    "ExperimentRunner",
    "ExperimentSpec",
    "GenerationRecorder",
    "IncubationEvolution",
    "MatingController",
    "RunResult",
    "SpatialBootstrap",
    "SpatialEA",
    "SpatialEAConfig",
    "SpatialIndividual",
    "SpawnedPopulation",
    "SubstrateNetwork",
    "analyze_spatial_clustering",
    "build_default_bootstrap",
    "clone_individual",
    "cluster_dbscan",
    "cluster_hierarchical",
    "cluster_kmeans",
    "compute_genotype_diversity",
    "compute_pairwise_distance_matrix",
    "create_initial_hyperneat_genome",
    "create_substrate_for_gecko",
    "crossover_genomes",
    "crossover_hyperneat",
    "find_optimal_clusters",
    "generate_spawn_positions",
    "load_controllers_from_json",
    "load_genotypes_from_npz",
    "mutate_genome",
    "mutate_hyperneat",
    "run_mating_movement_phase",
    "run_trial",
    "save_final_controllers",
    "seed_spatial_population_from_incubation",
    "select_individuals",
    "spatial_ea_config",
    "spatial_silhouette",
    "spawn_population_in_world",
]
