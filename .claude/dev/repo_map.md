# Repo map — where code lives

Developer-oriented tree of [src/ariel/](../../src/ariel/). Use it to decide
*where* new code belongs. Mirrors the module table in [../SCHEMA.md](../SCHEMA.md)
— keep the two consistent if you move things. For the *user-facing* view of the
same packages (public imports + snippets), see [../user/00_index.md](../user/00_index.md).

## `src/ariel/`

| Package | What it holds | Add new code here when… |
|---|---|---|
| [ec/](../../src/ariel/ec/) | EA engine: `EA`, `EAOperation`, `EASettings`, `Individual` (SQLModel), `Population`, `Archive`, `Crossover`, mutators/generators | adding evolution-loop machinery or data-layer logic |
| [ec/genotypes/tree/](../../src/ariel/ec/genotypes/tree/) | Tree genome + functional operators (`add_node`, `subtree_swap`, …) | new tree-genome operators (free functions) |
| [ec/genotypes/cppn/](../../src/ariel/ec/genotypes/cppn/) | CPPN decoding to morphology | CPPN→body decoding |
| [ec/genotypes/nde/](../../src/ariel/ec/genotypes/nde/) | `NeuralDevelopmentalEncoding` (PyTorch) | NDE encodings |
| [body_phenotypes/robogen_lite/](../../src/ariel/body_phenotypes/robogen_lite/) | Modular robot bodies: `modules/` (`Core`/`Brick`/`Hinge`, `Module(ABC)`), `cppn_neat/` (`Genome`, `IdManager`), `decoders/`, `prebuilt_robots/`, `constructor.py` | new module types, decoders, or prebuilt bodies |
| [body_phenotypes/lynx_mjspec/](../../src/ariel/body_phenotypes/lynx_mjspec/) | Lynx arm pipeline | Lynx-specific body work |
| [simulation/environments/](../../src/ariel/simulation/environments/) | `BaseWorld` + terrains (`SimpleFlatWorld`, `RuggedTerrainWorld`, `OlympicArena`, …) | a new world/terrain (subclass `BaseWorld`) |
| [simulation/controllers/](../../src/ariel/simulation/controllers/) | `Controller`, `Tracker` glue + CPGs (`SimpleCPG`, `NaCPG`, `nn.Module`) | new controllers |
| [simulation/tasks/](../../src/ariel/simulation/tasks/) | Fitness functions (`targeted_locomotion`, `gait_learning`, `turning_in_place`) | new fitness/evaluation metrics (free functions) |
| [parameters/](../../src/ariel/parameters/) | Type aliases (`ariel_types.py`), MuJoCo & module params (Pydantic) | shared types or config constants |
| [utils/](../../src/ariel/utils/) | `Tracker`, renderers, `VideoRecorder`, `MorphologicalMeasures`, `file_ops`, `mujoco_ops`, `optimizers/` (`RevDE`) | cross-cutting helpers |
| [visualisation/](../../src/ariel/visualisation/) | Dashboards / analysis (Plotly/Panel) | plotting & dashboards |

## Conventions when adding code

- Private/internal world files are underscore-prefixed (`_simple_flat.py`,
  `_base_world.py`) and re-exported via the package `__init__.py`. Follow the
  pattern: implement in `_name.py`, export the public class in `__init__.py`.
- Tests mirror this tree under [tests/](../../tests/). A new module needs a
  matching test (100% coverage — see [testing_and_tooling.md](testing_and_tooling.md)).
- Genotype *operators* are free functions taking the genome first; *genomes*
  themselves are dataclasses or stateful classes. See
  [functional_vs_oo.md](functional_vs_oo.md).
