# Spatial EA — upstream review notes

Proposed addition: `ariel.spatial_ea`, a spatial evolutionary algorithm in
which robots share one toroidal world and reproduce by physical proximity.
Spatial structure in the population is therefore emergent rather than imposed.

This document is written for reviewers of the upstream repository. It covers
what the package does, what it reuses from ARIEL and what it does not, the
decisions a reviewer might reasonably want to overturn, and several findings
about ARIEL itself that surfaced along the way.

---

## 1. Where it came from

The package is a port of a ~13,300-line research prototype into library form.
The prototype ran as a flat script directory with a module-level YAML
singleton, sibling imports, and only four `ariel.*` imports — it was not usable
as a library and could not be imported twice in one process.

The port targets **core EA parity**: the packaged version must reproduce the
prototype's experiments. That is now complete. The evolutionary core, batch
experiment tooling, per-generation recording, genotype clustering and post-hoc
visualisation are all packaged, and the prototype has been removed — its
history remains in git. Only its memory monitor was dropped outright, as
diagnostic scaffolding with no place in a library.

The prototype's own experiment configuration is kept at
`examples/spatial_ea/ea_config.yaml`. It is the worked example of the legacy
nested schema, and the regression case for `SpatialEAConfig.from_yaml`.

---

## 2. What the algorithm does

Per generation:

1. **Limit check** — stop if the population hit `max_population_limit` or fell
   below `min_population_limit`. Extinction and runaway growth are recorded
   outcomes, not errors.
2. **Zone update** — mating zones relocate under `static`,
   `generation_interval`, or `event_driven` strategies.
3. **Spawn** — the whole population goes into one compiled MuJoCo world.
4. **Evaluate once** — each individual is scored alone in a reused isolated
   world, so fitness measures locomotion, decoupled from social dynamics.
   Fitness is inherited thereafter and never recomputed.
5. **Record** — fitness, age and genome diversity.
6. **Select** — one of seven survivor policies (below).
7. **Reproduce** — energy depletion → **shared-world movement phase** → pairing
   → zone relocation → mating energy effect → crossover/clone → mutation →
   offspring placement.

The movement phase is the part that makes this "spatial": every robot is driven
by its own HyperNEAT substrate and receives a normalised direction vector as a
*neural input*. Nothing multiplies the motor outputs afterwards, so a controller
only benefits from the directional signal if evolution teaches it to use it.
Whether two robots reproduce is then decided by where physics actually put them.

**Selection policies** split into two families, which is load-bearing:

| Family | Policies | Population size |
|---|---|---|
| Truncating | `fitness_based`, `age_based`, `parents_die`, `zone_capacity` | held at target |
| Stochastic | `probabilistic_age`, `energy_based`, `density_based` | free to grow or collapse |

`density_based` is the research-relevant one: local crowding is a Gaussian
kernel sum over toroidal neighbour distances, and death probability is
`P_base + P_max·(1 − e^{−ρ/ρ_c})`, creating negative feedback between density
and survival.

---

## 3. What is reused from ARIEL, and what is not

This is the section most likely to draw objections, so it is stated plainly.

### Reused

| ARIEL API | Used for |
|---|---|
| `SimpleFlatWorld` + `BaseWorld.spawn` | multi-robot world construction |
| `prebuilt_robots.gecko` | the robot body |
| `utils.tracker.Tracker` | per-generation trajectory history |
| `utils.video_recorder.VideoRecorder` | frame sink for generation videos |
| `parameters.ariel_types` (`FloatArray`, `Position`, `Dimension`) | type aliases |
| `ariel.log` / `ariel.console` | logging and terminal output |
| `pydantic_settings.BaseSettings` config pattern | `SpatialEAConfig` |

That is a small surface for a ~10,650-line package, and reviewers should know it
is small **on purpose in some places and by necessity in others**.

### Not reused, with reasons

- **`ec.Individual`** is `SQLModel, table=True`. A second `table=True` subclass
  collides in the shared registry, so the spatial individual cannot extend it.
  It also carries per-generation position, orientation, energy and zone state
  that does not belong in a database row.
- **`ec.Population`** is annotated `list[Individual]`. It duck-types in
  practice — `best()`, `sort()`, `sample()`, `where()` all work on a foreign
  individual — but `alive`, `dead`, `evaluated` and `unevaluated` raise
  `AttributeError` because they read SQLModel columns. Half the API.
  **Making `Population` generic upstream would unblock this.**
- **`ec.EA`** is database-backed and reads `is_maximisation` /
  `target_population_size` from a module-level singleton rather than its
  constructor arguments. The spatial EA needs a variable population size and no
  DB.
- **`ec` selection operators** — there are none. No tournament, roulette or
  rank anywhere in `ariel.ec`. All 507 lines of `selection.py` and the
  tournament GA in `incubation.py` exist because of this gap.
- **`simulation.controllers.Controller`** clips to a hard-coded ±π/2 (ignoring
  any configured range), alpha-blends control outputs, and defaults to acting
  once every 50 steps — one second at `dt=0.02`. The spatial EA controls every
  step with no blending. Not a drop-in.
- **`utils.runners.simple_runner`** writes random noise to `data.ctrl`; it is a
  demo. `thread_safe_runner` is close, but leaves no hook for the per-step
  periodic wrapping the toroidal world needs.
- **`visualisation/`** is a NiceGUI/Panel dashboard playground with no
  figure-producing API.

### Acknowledged duplication

`body_phenotypes.robogen_lite.cppn_neat` is a working NEAT implementation
(`Genome.random/mutate/crossover/activate`, `Node`, `Connection`, `IdManager`
with real innovation numbers). It genuinely overlaps `spatial_ea/hyperneat.py`
and `spatial_ea/genetics.py`.

It was benchmarked, and it is *faster* than the port's CPPN (5.2 µs vs 8.0 µs
per `activate`), so performance is not the reason for the duplication. The
reasons are semantic:

- `Genome.mutate()` performs structural mutation only. The spatial EA's
  dominant operator is weight perturbation at rate 0.8.
- The activation set lacks `linear` and `abs`, and spells sine `"sin"`; the
  prototype uses `linear` on input nodes.
- `Genome.random()` builds a minimal fully-connected topology; the prototype
  deliberately seeds high structural and weight diversity, which materially
  affects early search.
- Innovation identity comes from a global `IdManager`; the port keys on
  `(from_node, to_node)`.

The HyperNEAT *substrate* — painting a large network's weights from a CPPN
queried on neuron coordinates — has no ARIEL equivalent and is needed either
way.

**Recommendation:** consolidating onto `cppn_neat` is worth doing, but it
changes experimental results and so should be a deliberate, re-baselined
change rather than a merge-time cleanup.

---

## 4. Deviations from the research prototype

The port is not bug-compatible. Seven defects were found in the prototype and
fixed rather than reproduced. Reviewers comparing against published results
should know about these.

| # | Prototype behaviour | Port behaviour |
|---|---|---|
| 1 | Spatial crossover kept one parent's node set while inheriting both parents' connections, leaving connections referencing non-existent nodes (silently dropped at evaluation) | Node sets are merged. The prototype's *own* incubation phase already did this correctly. |
| 2 | Mid-simulation periodic wrapping wrote to `data.geom_xpos`, which MuJoCo recomputes from `qpos` every step — so the toroidal world was effectively bounded | Wrapping is written to the free-joint `qpos`. |
| 3 | `parents_die` was called with an empty `paired_indices`, so it never retired any parent and silently degraded to fitness truncation | Paired indices carry across the generation boundary. |
| 4 | Nothing ever decremented energy, so `energy_based` selection could never kill anyone | Energy depletes per generation and on mating; a test drives a population to genuine extinction. |
| 5 | `add_node` mutation placed the new node on the output layer, producing dead connections | Layers shift to keep the output layer deepest. |
| 6 | `_genotype_to_dict` and `genotype_distance.py` referenced attributes that do not exist on the genome classes (`node.id`, `conn.innovation`, `conn.in_node`) — controller export would have raised | Both rewritten against the real schema and covered by tests. The distance module now identifies a connection by its `(from_node, to_node)` endpoints, since this encoding has no innovation numbers. |
| 7 | Spatial cluster statistics used Euclidean geometry on a toroidal world, so a cluster straddling a seam got a centroid where no member was and read as diffuse | Centroids are a circular mean and separations use the toroidal metric when wrapping is on. In a test case a genuinely tight cluster moves from spread 1.49 / silhouette 0.49 to spread 0.11 / silhouette 0.94. |

Two further deliberate divergences:

- **Control is driven inside the stepping loop** rather than through
  `mujoco.set_mjcb_control`. The global callback outlives the model it closes
  over, which is why the prototype needed explicit cleanup between generations.
- **Coordinate convention standardised.** The world spans `[0, W]` on both axes
  in both boundary modes. The prototype's non-periodic clipping used
  `[−W/2, +W/2]` while its zone sampling used `[0, W]`.

---

## 5. Public API

```python
from ariel.spatial_ea import SpatialEA, SpatialEAConfig

engine = SpatialEA(config=SpatialEAConfig(world_size=(4.0, 4.0)))
best = engine.run()
```

Command line, with every config field exposed as a flag generated from the
model (so the parser cannot drift out of step with the settings):

```bash
python -m ariel.spatial_ea --population-size 8 --num-generations 4 \
    --world-size 4 4 --save-generation-plots true
```

`SpatialEAConfig.from_yaml` reads both the flat native schema and the research
prototype's nested schema, so existing experiment configurations load unchanged.

Plotting lives in `ariel.spatial_ea.visualization` and is **not** re-exported
from the package, so importing `ariel.spatial_ea` does not pull in
`matplotlib.pyplot` (~0.29 s). Batch runs that never plot do not pay for it.

Three independent output switches: `save_results` (CSV/NPZ/controllers),
`save_plots` (summary figure), `save_generation_plots` (one trajectory figure
per generation).

---

## 6. Batch experiments and recording

Two capabilities sit on top of the engine.

### Batch experiments

An experiment is a named set of configuration overrides run several times with
different seeds. `ExperimentRunner` runs the trials, sequentially or across
worker processes, and pools them.

```python
from ariel.spatial_ea import ExperimentRunner, ExperimentSpec, SpatialEAConfig

runner = ExperimentRunner(base_config=SpatialEAConfig(), seed=0)
spec = ExperimentSpec(
    name="mating_zones",
    overrides={"pairing_method": "mating_zone", "num_mating_zones": 3},
    num_runs=5,
)
results, aggregated = runner.run_and_save(spec, parallel=True)
```

Two details are worth a reviewer's attention.

*Overrides are a plain mapping*, applied with `model_copy` and then
revalidated so that dependent fields stay consistent. The research prototype
mirrored every setting in a second ~60-field dataclass and round-tripped it
through a YAML file, because its configuration was a module-level singleton
that workers had to re-import. A flat settings model removes the need for both.
Unknown override names raise rather than being silently ignored.

*Aggregation has to state how it treats a finished run.* Population size is not
fixed and runs stop early, so trials of one experiment routinely differ in
length. Three strategies:

| Strategy | A run that has ended contributes |
|---|---|
| `nan` | nothing — later generations average only the live runs |
| `forward_fill` | the last value it recorded |
| `terminal_state` | its outcome: zero if extinct, its ceiling if it exploded |

`AggregatedResults.runs_active` records how many trials were genuinely still
running at each generation, and padding never inflates it. The comparison
figure plots it underneath the metric, because a mean over two surviving runs
should not be read like a mean over twenty.

Driver: `examples/spatial_ea/run_experiments.py`, which defines seven
experiments spanning the prototype's research questions (random vs. proximity
vs. zone pairing; density vs. energy death) and supports `--grid` sweeps.

### Recording

`GenerationRecorder` captures a video and/or a mid-phase snapshot of the shared
world, driven from inside the movement loop. Enable with
`--record-generation-videos true` or `--save-generation-snapshots true`.

It reuses ARIEL's `VideoRecorder` as the frame sink. ARIEL's `video_renderer`
and `tracking_video_renderer` could not be reused: they drive their own
simulation loops, and the movement phase already owns one that applies control
and boundary wrapping per step.

If no GL context is available the recorder disables itself and logs a warning
rather than aborting a run.

---

## 7. Module map

| Module | Lines | Notes |
|---|---:|---|
| `engine.py` | 956 | generation loop, selection and reproduction orchestration |
| `visualization.py` | 1451 | trajectory, statistics, zone, experiment and cluster figures |
| `experiment.py` | 867 | batch runs, parallelism, grid search, aggregation |
| `clustering.py` | 620 | genome clustering and its spatial correlation |
| `genotype_distance.py` | 521 | structural, weight, combined and behavioural distance |
| `interaction.py` | 764 | toroidal geometry, pairing strategies, mating zones |
| `genetics.py` | 589 | HyperNEAT genome operators |
| `hyperneat.py` | 545 | CPPN and substrate network |
| `data.py` | 617 | per-generation statistics and CSV/NPZ export |
| `selection.py` | 507 | seven survivor policies |
| `config.py` | 511 | settings, native and legacy YAML |
| `movement.py` | 454 | shared-world mating movement phase |
| `evaluation.py` | 388 | isolated fitness with optional directional target |
| `world.py` | 414 | multi-robot spawning and per-robot handles |
| `persistence.py` | 317 | controller save/load |
| `incubation.py` | 296 | pre-adaptation GA |
| `cli.py` | 164 | model-driven argument parser |
| `individual.py` | 139 | per-robot state |
| `recording.py` | 307 | per-generation video and snapshot capture |

Tests: `tests/unit/test_spatial_ea/`, 21 test modules, 270 tests.
Examples: `examples/spatial_ea/`, 7 scripts.

---

## 8. How to verify

```bash
uv pip install --python .venv pytest          # see §10
.venv/bin/python -m pytest tests/unit/test_spatial_ea -q
.venv/bin/python examples/spatial_ea/visual_experiment.py
```

`visual_experiment.py` runs the same seeded population twice — once with the
real MuJoCo movement phase, once with a cheap analytical nudge — writes a
figure per generation, and prints a pass/fail checklist of eight mechanisms
(evaluation, movement, pairing, birth, death, energy, diversity, recording).

**Parity check.** The package writes `evolution_data_*.csv` / `.npz` and
`final_controllers_*.json` with column names identical to the prototype's.
Before the prototype was removed, its `ExperimentVisualizer` was confirmed to
render this package's output unmodified — both the 12-panel comprehensive
figure and the six publication plots — which is what established parity.

Re-analysing a finished run no longer needs the prototype:

```bash
python examples/spatial_ea/visualize_run.py --results __results__
python examples/spatial_ea/analyze_clustering.py --results __results__
```

`EvolutionDataCollector.from_csv` reloads a saved run into the same object a
live run produces, so every figure works on either.

**A scale caveat for anyone running the demo.** Untrained HyperNEAT controllers
locomote at roughly one to two centimetres per simulated second. The examples
use a small world so that real movement is legible; at the prototype's 25 m
default, nothing visible happens in a short run. Robots that walk properly are
what the incubation phase is for, and it is too slow to demonstrate inline.

---

## 9. Known limitations

- Locomotion quality is the binding constraint on everything downstream. With
  short runs, pairing is driven more by spawn placement than by navigation.
- Nothing from the prototype remains unported except its memory monitor, which
  was dropped deliberately.
- Randomness goes through the `numpy.random` and `random` module-level
  generators rather than a private stream, so that seeding a worker process
  reproduces a run — the deferred parallel experiment runner depends on this.
  It diverges from ARIEL's `RNG = np.random.default_rng(SEED)` convention and
  accounts for most of the package's remaining ruff findings (34 × NPY002).
- The body is fixed to `gecko()`; only controllers evolve.
- Ruff reports 110 findings over ~10,650 lines against `select = ["ALL"]`, which
  compares favourably with the existing package (`ec/` 173, `simulation/` 243,
  `body_phenotypes/` 481) but is not zero.

---

## 10. Findings about ARIEL itself

These are independent of whether the spatial EA is merged.

1. **`pytest` cannot run the whole suite.** `tests/unit/test_config/` and the
   new `tests/unit/test_spatial_ea/` both contain a `test_config.py`, and the
   test subdirectories had no `__init__.py`, so collection failed with an
   import-file mismatch. This fork adds `__init__.py` to the five test
   subdirectories. `tests/` and `tests/unit/` already had them.

2. **A pre-existing test failure was hidden by that.**
   `tests/unit/test_simulation/test_environments/test_environments.py::test_all_heightmap_functions`
   fails with `KeyError: 'smooth_edges_heightmap'` — the test's argument
   dictionary was not updated when that heightmap function was added. Not
   touched by this work.

3. **`uv run pytest` does not work from a clean checkout.** It tries to build
   `labmaze` (via `dm-control`) from source and fails on a missing `bazel`.
   Installing pytest into an existing venv (`uv pip install --python .venv
   pytest`) avoids the re-resolve.

4. **`types-pyyaml` is declared but `pyyaml` itself is not.** `pyyaml` arrives
   transitively. This fork adds it to `dependencies`.

5. **`_base_world.py` has an uncommitted MuJoCo compatibility patch** reverting
   `proj=mj.mjtProjection.*` to `orthographic=bool`. Committed code assumes
   MuJoCo ≥ 3.6.0 while `pyproject.toml` pins `mujoco>=3.3.6`, so a compliant
   install can fail to build any world. **This needs a version guard or a
   tighter pin upstream, independently of this package.**

6. **`ec/generators copy.py`** — a stray duplicate with a literal space in the
   filename, imported by nothing.

### Suggested upstream changes that would shrink this package

- Make `ec.Population` generic over its element type. Would let the spatial EA
  drop its own container handling.
- Add selection operators to `ec` (tournament, roulette, rank). Would remove
  the largest genuinely-new module here.
- Give `Controller` a configurable clip range and an option to disable output
  blending. Would make the movement phase reusable.
- Promote `cppn_neat` out of `body_phenotypes` and add weight mutation. Would
  let this package drop ~1,100 lines.
