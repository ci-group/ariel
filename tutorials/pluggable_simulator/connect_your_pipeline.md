# Connecting your Isaac Lab + RL pipeline to ariel's EA loop

> Companion to [`README.md`](./README.md) §6. Read §6 first for the
> elevator-pitch + five-step recipe; this doc is the depth pass.

## 1. Why this doc

ariel's contribution to the consortium is the **evolutionary layer**:
genome representations, decoders, EA operators, repair, descriptors,
and a morphology IR (`DroneBlueprint`) that any simulator can
consume. It is **not** another simulator and **not** another RL
library.

If you already have an Isaac Lab + RL training pipeline that knows
how to train a hover (or any) policy for a fixed drone, you do
**not** need to switch tools to use ariel. You wrap your existing
trainer behind a one-line interface — "given this `DroneBlueprint`,
train a policy and report fitness" — and ariel's EA loop becomes
the outer scheduler that picks which morphologies to try.

This doc walks through the three roles you implement, the files
to look at, two integration patterns (in-process vs subprocess),
fitness-extraction options, and the common pitfalls we hit while
writing the shipped reference.

## 2. The three roles, in detail

### Role 1 — Genome → DroneBlueprint decoder

**What it is.** A function that takes whatever genotype your EA
operates on (numpy array, dataclass, CPPN, ...) and produces a
`DroneBlueprint`: the simulator-agnostic morphology IR.

**What you implement.** A function `genome_to_blueprint(g) →
DroneBlueprint`. In most cases this is one line: call a shipped
decoder.

**Which shipped file to copy from.**
[`src/ariel/body_phenotypes/drone/decoders.py`](../../src/ariel/body_phenotypes/drone/decoders.py)
— `spherical_angular_to_blueprint` and `cartesian_euler_to_blueprint`
are the production decoders. `spherical_angular` parameterises each
arm by (length, azimuth, pitch, motor-az, motor-pitch, spin-dir);
`cartesian_euler` uses (x, y, z, roll, pitch, yaw).

**Minimal sketch.**
```python
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint

def genome_to_blueprint(genome_matrix, propsize=5):
    return spherical_angular_to_blueprint(genome_matrix, propsize=propsize)
```

### Role 2 — An RL env that takes a DroneBlueprint

**What it is.** Your existing env, parameterised by a
`DroneBlueprint` at construction time. The env converts the
blueprint to whatever your simulator wants (URDF, USD, MjSpec,
propeller list) and spawns N parallel agents.

**What you implement.** A `from_blueprint(bp, ...)` classmethod on
your env's config, or an equivalent constructor.

**Which shipped file to copy from.**
[`src/ariel/simulation/tasks/isaaclab_hover_env.py`](../../src/ariel/simulation/tasks/isaaclab_hover_env.py)
is the reference. `make_blueprint_usd(bp, output_dir, robot_name)`
runs `blueprint_to_urdf` then Isaac Lab's `UrdfConverter` to write
the USD. `IsaacLabBlueprintHoverEnvCfg.from_blueprint(bp, num_envs)`
returns a config with `self.robot.spawn.usd_path` pointing at that
USD.

**Minimal sketch (DirectRLEnv shape).**
```python
@configclass
class MyEnvCfg(DirectRLEnvCfg):
    @classmethod
    def from_blueprint(cls, bp, num_envs=64, usd_output_dir=None):
        usd_path = make_blueprint_usd(bp, usd_output_dir or "/tmp")
        cfg = cls(num_envs=num_envs)
        cfg.robot.spawn.usd_path = str(usd_path)
        return cfg
```

### Role 3 — evaluate(genome) → fitness + outer EA loop

**What it is.** The glue: pick a morphology, train a policy on it,
score it, hand the scalar back to the EA. The EA then selects and
mutates the high-scoring morphologies.

**What you implement.** A loop that for each individual builds the
env, runs your trainer, parses out a fitness scalar, and feeds
populations through tournament selection + mutation.

**Which shipped file to copy from.**
[`evolve.py`](./evolve.py) is the reference. Its `_evaluate_in_subprocess`
is the swap point: replace its `subprocess.run([..., train.py, ...])`
with a call into your own trainer.

**Minimal sketch.**
```python
for gen in range(args.generations):
    fitnesses = [evaluate(g) for g in population]
    population = tournament_select_and_mutate(population, fitnesses)
```

## 3. Files to look at, by role

| Role | File | Why |
|---|---|---|
| Blueprint IR | [`src/ariel/body_phenotypes/drone/blueprint.py`](../../src/ariel/body_phenotypes/drone/blueprint.py) | The surface the decoder must populate; ships `to_dict`/`from_dict`/`save_json`/`load_json` for caching and subprocess hand-off |
| Decoder examples | [`src/ariel/body_phenotypes/drone/decoders.py`](../../src/ariel/body_phenotypes/drone/decoders.py) | `spherical_angular_to_blueprint`, `cartesian_euler_to_blueprint` |
| Backend conversion | [`src/ariel/body_phenotypes/drone/backends.py`](../../src/ariel/body_phenotypes/drone/backends.py) | `blueprint_to_urdf`, `blueprint_to_mjspec`, `blueprint_to_propellers` |
| Isaac Lab env (DirectRLEnv) | [`src/ariel/simulation/tasks/isaaclab_hover_env.py`](../../src/ariel/simulation/tasks/isaaclab_hover_env.py) | Copy-paste template: `from_blueprint` cfg + `make_blueprint_usd` helper + `_setup_scene` / `_pre_physics_step` / `_apply_action` / `_get_rewards` / `_reset_idx`. Also exports `make_rl_games_agent_cfg`. |
| gymnasium VecEnv | [`src/ariel/simulation/tasks/drone_gate_env.py`](../../src/ariel/simulation/tasks/drone_gate_env.py) + [`blueprint_gate_env.py`](../../src/ariel/simulation/tasks/blueprint_gate_env.py) | For sb3-style RL libraries — implements the `BlueprintGateEnv` Protocol |
| Subprocess EA + RL (reference) | [`tutorials/pluggable_simulator/evolve.py`](./evolve.py) + [`train.py`](./train.py) | The shipped loop: per-individual subprocess + checkpoint-filename fitness |
| Subprocess EA + RL (HPC) | [`src/ariel/ec/drone/evaluators/gate_evaluator.py`](../../src/ariel/ec/drone/evaluators/gate_evaluator.py) + [`gate_train.py`](../../src/ariel/ec/drone/evaluators/gate_train.py) | Pre-existing in-repo pattern; SLURM-friendly variant |
| Genome representations | [`src/ariel/ec/drone/genome_handlers/`](../../src/ariel/ec/drone/genome_handlers/) | `spherical_angular`, `cartesian_euler`, `cppn_neat`, `hybrid_cppn` with mutation/crossover |

## 4. Two integration patterns

### Pattern A — Subprocess per individual (recommended; shipped reference)

The parent process holds the EA state. For each individual:

1. Convert genome → `DroneBlueprint`; save as JSON.
2. `subprocess.run([python, train.py, --blueprint-json <path>,
   --experiment-prefix <unique>, ...])`.
3. After the subprocess exits, read fitness from
   `runs/<experiment-prefix>_<timestamp>/nn/last_*.pth`.

This is what [`evolve.py`](./evolve.py) does.

**Pros.**
- **Clean Isaac Sim state each individual.** No env-teardown reuse,
  no global-registry leakage between genomes.
- **Crash isolation.** A failed individual doesn't bring down the
  EA — parent just records `nan` and moves on.
- **HPC/SLURM-friendly.** Same shape works whether subprocesses run
  locally or get submitted to a job queue.

**Cons.**
- **~6 s Isaac Sim startup per individual.** Real cost on small
  populations or short training budgets. For a 3 gen × 4 ind smoke
  with 30 PPO epochs each, startup overhead is ~25% of wall time.

### Pattern B — In-process

One process; Isaac Sim launches once; the EA loops over individuals
inside the same Python session, building + tearing down `DirectRLEnv`
between genomes.

**Pros.**
- **No Isaac Sim startup per individual.** Once the launcher is up,
  individual eval is bounded only by PPO time.
- **Easier debugging.** Single-process stack traces; no IPC.

**Cons.**
- **Env teardown reuse is undertested.** Our first attempt at this
  pattern hung after individual 0 — `IsaacLabBlueprintHoverEnv.close()`
  didn't fully release scene state, and the second `__init__` blocked
  indefinitely. Fixing this requires deeper invasion of Isaac Lab's
  scene-spawn machinery than we wanted to ship in a tutorial.
- **rl_games' global registries are sticky.** `vecenv.register` and
  `env_configurations.register` bind names process-wide; you must
  re-register `env_creator` per individual to capture the new env.
- **One crash takes out the whole run.** A genome that produces an
  invalid URDF (degenerate inertia, etc.) crashes Isaac Sim's loader
  — and that's your whole EA gone.

If you want to attempt in-process anyway, structure it as:

```python
# Once at startup.
app_launcher = AppLauncher(args, multi_gpu=False)

for gen in range(generations):
    for ind in population:
        env = MyEnv(cfg=MyEnvCfg.from_blueprint(ind.blueprint, num_envs=N))
        train(env, ...)             # your trainer
        fitness = score(env, ...)   # your eval
        env.close()                 # may hang on Isaac Lab today
```

We recommend Pattern A until env teardown is hardened.

## 5. Fitness-extraction options

### Option 1 — Parse the rl_games checkpoint filename (simplest)

rl_games writes checkpoints to
`runs/<exp_name>_<timestamp>/nn/last_<exp_name>_ep_<E>_rew__<R>_.pth`.
The reward in the filename is the value the runner reported when it
wrote that checkpoint — good enough for a fitness scalar.

```python
import re
ckpt = max(runs_dir.glob(f"{exp}_*/nn/last_*.pth"),
           key=lambda p: p.stat().st_mtime)
fitness = float(re.search(r"rew_+(-?[\d.]+)_", ckpt.name).group(1))
```

Used by `_extract_reward_from_checkpoint` in [`evolve.py`](./evolve.py).

**Pros.** No code inside the trainer; works across processes.
**Cons.** Brittle if rl_games changes its filename schema (it has,
at least once, between major versions).

### Option 2 — Subclass IsaacAlgoObserver

rl_games' `IsaacAlgoObserver` (from `rl_games.common.algo_observer`)
gets called after each PPO epoch with the current rewards tensor.
Subclass it to capture the trajectory of rewards in-memory; return
the last-N-epoch mean as fitness.

**Pros.** Robust to filename changes; you control the aggregation
(last-N mean, max, percentile).
**Cons.** Requires running the trainer in-process so the observer
can be read after `runner.run` returns.

### Option 3 — Deterministic post-training eval pass

After `runner.run({"train": True, ...})` finishes, save the policy,
then run an eval pass with `play: True` and `deterministic: True`
for K episodes; average the cumulative reward.

**Pros.** Most accurate measure of policy quality.
**Cons.** Doubles per-individual wall time. Worth it when individual
training is long enough that the noise in option 1 (which scores
the last *training* episodes, mid-exploration) dominates the EA
signal.

For the tutorial-sized smoke (30 PPO epochs / individual) option 1
is fine. For real runs (hundreds of epochs) option 2 or 3 is
worth the setup cost.

## 6. Common pitfalls

### Orphan Isaac Sim processes after a failed eval

Isaac Sim's launcher can leave threads spinning at ~120% CPU after
an exception in env setup, even after the Python process "exits".
The check is in README §3c — run it after every failed iteration:

```bash
ps -u $USER -o pid,etime,pcpu,cmd \
    | grep -E "tutorials/pluggable_simulator/(train|evolve)\.py" \
    | grep -v grep
pkill -KILL -f "tutorials/pluggable_simulator/"  # if needed
```

In the subprocess pattern this matters less (each child cleans up
on its own exit), but a hung subprocess will still wedge the parent
EA until the parent's `subprocess.run` returns. Consider a
`timeout=` on the run call for long-running training.

### `simulation_app.close()` can hang at the end of training

A related symptom: after a successful PPO run the child writes its
checkpoint, logs "closing simulation app", then spins Isaac Sim's
threads at ~120% CPU indefinitely. The shipped `train.py` works
around this by hard-exiting with `os._exit(exit_code)` instead of
calling `simulation_app.close()` — the checkpoint is already on
disk by that point, so there's nothing to flush. If you wrap your
own trainer in a subprocess driven by an EA, do the same: skip the
graceful close and `os._exit` once your fitness signal is
persisted.

### rl_games global registries

`rl_games.common.vecenv.register("name", factory)` and
`rl_games.common.env_configurations.register("name", {"env_creator":
...})` bind names process-wide. In an in-process loop you must
re-register `env_creator` for every individual so the trainer picks
up the new env, not the previous one. In the subprocess pattern
this is automatic — each child has a fresh registry.

### Don't add simulator-owned binaries to ariel's deps

`pyproject.toml [project.dependencies]` must **not** name `torch`,
`gymnasium`, `numpy>=2`, or anything else Isaac Lab owns. ariel's
core deps are simulator-agnostic; binaries live in extras
(`rl-sb3`, `torch`). The guardrail step in README §3b (step 6)
catches accidental leaks by diffing `pip list` before/after the
ariel install and refusing to proceed if simulator binaries moved.

### Don't share blueprint JSON paths across overlapping subprocesses

`evolve.py` writes per-individual blueprint JSON files under a
`tempfile.TemporaryDirectory(prefix="ariel_evolve_")` with unique
names (`ariel_evolve_<eval_id>.json`). If you run subprocesses in
parallel, ensure each one's JSON has a distinct path — and that
the per-individual `--experiment-prefix` is unique so the
checkpoint-filename fitness extraction finds the right run.

### `DroneGateEnv` import path needs EA orchestration deps

In the Isaac Lab env, importing `DroneGateEnv` transitively pulls
in ariel's EA orchestration deps (sqlmodel, pydantic-settings) that
the Isaac Lab install path intentionally doesn't bring in. The §3b
recipe pip-installs them as a separate step. Importing only
`DroneBlueprint` + `decoders` + `backends` (the chain Isaac Lab
actually uses) works without those orchestration deps.

## 7. Calibration on a single machine

Measured wall times from one end-to-end run of the shipped
reference (`evolve.py` defaults: 3 gen × 4 ind × 30 PPO epochs ×
16 envs, single RTX-class GPU, warm caches):

| Phase | Wall time |
|---|---|
| rl_games PPO (30 epochs × 16 envs) | ~4-5 s |
| Isaac Sim startup + scene build (per subprocess) | ~8-10 s |
| Per-individual total | ~13-15 s |
| Per-generation total (population 4) | ~55 s |
| Full 3-gen run | ~160 s (~2.7 min) |

Notes:
- First individual is slightly slower (~15 s) than warm subsequent
  ones (~13 s) due to Isaac Sim's cold-cache USD pipeline.
- The very first ever invocation in a fresh checkout can take
  several minutes as Isaac Sim builds its asset cache. Subsequent
  runs in the same machine are at the rates above.
- Numbers scale linearly with population × generations × epochs;
  bigger envs (more parallel agents per individual) reduce
  per-individual variance more than they cost.

---

See also:
- [`README.md`](./README.md) — top-level tutorial.
- [`evolve.py`](./evolve.py) — shipped reference outer loop.
- [`train.py`](./train.py) — shipped per-individual inner loop.
- [`DRONE_BLUEPRINT_PLAN.md`](../../DRONE_BLUEPRINT_PLAN.md) §1
  "Value proposition" and §6 "Design decisions" for the deeper
  rationale.
