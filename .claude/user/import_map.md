# Import map — symbol → import line

Fast lookup so an AI doesn't grep `src/` to find where something comes from.
Verified against `src/ariel/` `__init__.py`s and the `examples/`. If an import
ever fails, the source moved — check the package `__init__.py`.

## Evolutionary computation

| Symbol | Import |
|---|---|
| `EA`, `EAOperation`, `EASettings` | `from ariel.ec import EA, EAOperation, EASettings` |
| `Individual`, `Population`, `Archive` | `from ariel.ec import Individual, Population, Archive` |
| `Crossover`, `FloatMutator`, `IntegerMutator` | `from ariel.ec import Crossover, FloatMutator, IntegerMutator` |
| `FloatsGenerator`, `IntegersGenerator`, `SEED` | `from ariel.ec import FloatsGenerator, IntegersGenerator, SEED` |

## Bodies / genotypes

| Symbol | Import |
|---|---|
| `Genome` (CPPN-NEAT) | `from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome` |
| `IdManager` | `from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager` |
| `MorphologyDecoderBestFirst` | `from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import MorphologyDecoderBestFirst` |
| `HighProbabilityDecoder` | `from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder` |
| `VectorDecoder` | `from ariel.body_phenotypes.robogen_lite.decoders.vector_decoding import VectorDecoder` |
| `construct_mjspec_from_graph` | `from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph` |
| `NUM_OF_TYPES_OF_MODULES`, `NUM_OF_ROTATIONS` | `from ariel.body_phenotypes.robogen_lite.config import NUM_OF_TYPES_OF_MODULES, NUM_OF_ROTATIONS` |
| `body_spider45`, `body_spider` | `from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider_with_blocks import body_spider45` |
| `spider` | `from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider` |
| `gecko` | `from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko` |
| `random_spider20` | `from ariel.body_phenotypes.robogen_lite.prebuilt_robots.random_20 import random_spider20` |
| `TreeGenome` | `from ariel.ec.genotypes.tree.tree_genome import TreeGenome` |
| tree operators (`random_tree`, `add_node`, …) | `from ariel.ec.genotypes.tree.operators import random_tree, add_node` |
| `NeuralDevelopmentalEncoding` | `from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding` |

## Environments (worlds)

| Symbol | Import |
|---|---|
| `SimpleFlatWorld`, `RuggedTerrainWorld`, `OlympicArena`, `BaseWorld`, … | `from ariel.simulation.environments import SimpleFlatWorld, RuggedTerrainWorld, OlympicArena, BaseWorld` |
| `SimpleFlatWorldWithTarget` *(not in package `__init__`)* | `from ariel.simulation.environments._simple_flat_with_target import SimpleFlatWorldWithTarget` |

Full exported list: `AmphitheatreTerrainWorld`, `BaseWorld`, `CompoundWorld`,
`CraterTerrainWorld`, `OlympicArena`, `RuggedTerrainWorld`, `RuggedTiltedWorld`,
`SimpleFlatWorld`, `SimpleTiltedWorld`.

## Controllers

| Symbol | Import |
|---|---|
| `Controller` | `from ariel.simulation.controllers.controller import Controller` |
| `SimpleCPG`, `create_fully_connected_adjacency` | `from ariel.simulation.controllers.simple_cpg import SimpleCPG, create_fully_connected_adjacency` |
| `NaCPG` | `from ariel.simulation.controllers import NaCPG` |
| `get_state_from_data` | `from ariel.simulation.controllers.utils.data_get import get_state_from_data` |

## Tasks / fitness

| Symbol | Import |
|---|---|
| `distance_to_target`, `fitness_delta_distance`, `fitness_distance_and_efficiency`, `fitness_survival_and_locomotion`, `fitness_direct_path`, `fitness_speed_to_target` | `from ariel.simulation.tasks.targeted_locomotion import ...` |
| `xy_displacement`, `x_speed`, `y_speed` | `from ariel.simulation.tasks.gait_learning import ...` |
| `turning_in_place` | `from ariel.simulation.tasks.turning_in_place import turning_in_place` |

## Utils / I/O

| Symbol | Import |
|---|---|
| `Tracker` | `from ariel.utils.tracker import Tracker` |
| `VideoRecorder` | `from ariel.utils.video_recorder import VideoRecorder` *(also `from ariel.utils.renderers import VideoRecorder`)* |
| `video_renderer`, `single_frame_renderer`, `tracking_video_renderer` | `from ariel.utils.renderers import video_renderer, single_frame_renderer` |
| `MorphologicalMeasures` | `from ariel.utils.morphological_descriptor import MorphologicalMeasures` |
| `generate_save_path` | `from ariel.utils.file_ops import generate_save_path` |
| `simple_runner`, `thread_safe_runner` | `from ariel.utils.runners import simple_runner, thread_safe_runner` |
| `RevDE` | `from ariel.utils.optimizers.revde import RevDE` |
| type aliases (`Dimension`, `Position`, `FloatArray`) | `from ariel.parameters.ariel_types import Dimension, Position, FloatArray` |
