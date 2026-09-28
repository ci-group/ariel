# Connecting your Isaac Lab + RL pipeline to ariel's EA loop

> Companion to [`README.md`](./README.md) §6. Read §6 first for the
> elevator-pitch + five-step recipe; this doc is the depth pass.
>
> **Before you start:** everything here runs in the Isaac Lab conda env.
> Build it with [`README.md`](./README.md) §3b and finish that section's
> step 6 check before trying the code below.

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
is the reference. `make_blueprint_usd(bp, output_dir=..., robot_name=...)`
runs `blueprint_to_urdf` then Isaac Lab's `UrdfConverter` to write
the USD. `IsaacLabBlueprintHoverEnvCfg.from_blueprint(bp, num_envs=...)`
returns a config with `robot.spawn.usd_path` pointing at that USD.

**Minimal sketch (DirectRLEnv shape).** Like everything that touches
`isaaclab`, this only runs after `AppLauncher` has started Isaac Sim.
```python
import tempfile

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass

from ariel.simulation.tasks.isaaclab_hover_env import make_blueprint_usd


@configclass
class MyEnvCfg(DirectRLEnvCfg):
    # Your task's fields go here. The method below assumes the cfg has a
    # `scene: InteractiveSceneCfg` and a `robot: ArticulationCfg` whose
    # `spawn` is a `UsdFileCfg` (see IsaacLabBlueprintHoverEnvCfg).

    @classmethod
    def from_blueprint(cls, bp, *, num_envs=64, usd_output_dir=None):
        # A fresh directory per call, so parallel evaluations never
        # overwrite each other's drone.urdf / drone.usd.
        if usd_output_dir is None:
            usd_output_dir = tempfile.mkdtemp(prefix="my_blueprint_usd_")
        usd_path = make_blueprint_usd(bp, output_dir=usd_output_dir)
        cfg = cls()
        cfg.scene.num_envs = num_envs   # num_envs lives on the scene cfg
        cfg.robot.spawn.usd_path = usd_path
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
| Isaac Lab env (DirectRLEnv) | [`src/ariel/simulation/tasks/isaaclab_hover_env.py`](../../src/ariel/simulation/tasks/isaaclab_hover_env.py) | Copy-paste template: `from_blueprint` cfg + `make_blueprint_usd` helper + `_setup_scene` / `_pre_physics_step` / `_apply_action` / `_get_rewards` / `_reset_idx`. Its default `action_mode="mixer"` shows how to make actuation morphology-aware: `_init_mixer` builds the blueprint's allocation matrix, and `_init_rotors` holds the per-rotor model. Also exports `make_rl_games_agent_cfg`. |
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
  EA — parent records `nan`, ranks it below every real fitness in
  selection, leaves it out of the per-generation best/mean/worst, and
  reports it in a `failed=k/N` count.
- **HPC/SLURM-friendly.** Same shape works whether subprocesses run
  locally or get submitted to a job queue.

**Cons.**
- **~10 s of Isaac Sim startup + scene build per individual.** In the
  `evolve.py` defaults (3 gen × 4 ind, 30 PPO epochs each) that is about
  two-thirds of each individual's ~14–16 s; PPO itself takes ~4.5–5 s.
  The share shrinks as training gets longer. Breakdown in §7 below.

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
is never handed the rewards directly. Its hooks are
`after_init(algo)` (once, with the training algorithm),
`process_infos(infos, done_indices)` (every env step) and
`after_print_stats(frame, epoch_num, total_time)` (once per epoch).
The reward statistic rl_games itself reports lives on the algorithm:
`algo.game_rewards`, a running mean of the returns of the last 100
finished episodes (`games_to_track` in the agent config). A subclass
keeps the `algo` handle and samples that mean once per epoch:

```python
from rl_games.common.algo_observer import IsaacAlgoObserver


class FitnessObserver(IsaacAlgoObserver):
    """Record rl_games' mean episode return after every PPO epoch."""

    def after_init(self, algo):
        super().after_init(algo)
        self.history = []  # one value per epoch that had finished episodes

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        if self.algo.game_rewards.current_size > 0:
            self.history.append(float(self.algo.game_rewards.get_mean()[0]))

    def fitness(self, last_n=1):
        if not self.history:
            return float("nan")  # no episode finished during training
        tail = self.history[-last_n:]
        return sum(tail) / len(tail)
```

Pass `FitnessObserver()` where `train.py` passes `IsaacAlgoObserver()`
to `Runner(...)`, and keep a reference to it. After `runner.run(...)`
returns, `observer.fitness()` equals the reward in Option 1's
checkpoint filename (both read the same running mean in the final
epoch); `observer.fitness(last_n=5)` averages over the last five epochs.

**Pros.** No filename parsing; you choose the aggregation.
**Cons.** The observer lives in the process that runs the trainer. In
Pattern A that is the child, so the child must write the value
somewhere the parent can read it (e.g. a small JSON file) before it
exits. Like Option 1, it scores the *training* episodes, which were
played with exploration noise.

### Option 3 — Deterministic post-training eval pass

After training, load the final checkpoint into an rl_games *player*
and roll the policy out without exploration noise. `runner.run({"play":
True, ...})` does run a player, but it only prints `av reward:` and
returns nothing. To get a number back, write the loop yourself, as
Isaac Lab's `scripts/reinforcement_learning/rl_games/play.py` does:

```python
import torch


def deterministic_eval(runner, env, checkpoint_path, n_episodes=16):
    """Mean return of the deterministic policy over n_episodes.

    `runner` is the rl_games Runner that trained the policy (it holds the
    agent config); `env` is the RlGamesVecEnvWrapper it trained on.
    """
    player = runner.create_player()
    player.restore(checkpoint_path)
    player.reset()
    obs = env.reset()["obs"]
    _ = player.get_batch_size(obs, 1)
    returns = torch.zeros(env.num_envs, device=env.device)
    finished = []
    with torch.inference_mode():
        while len(finished) < n_episodes:
            actions = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
            obs, rew, dones, _ = env.step(actions)
            obs = obs["obs"]
            returns += rew
            done_ids = dones.nonzero(as_tuple=False).flatten()
            finished.extend(returns[done_ids].tolist())
            returns[done_ids] = 0.0
    return sum(finished[:n_episodes]) / n_episodes
```

**Pros.** Scores the policy you would actually deploy, not the
noisy training episodes.
**Cons.** Extra rollouts per individual: with `n_episodes` equal to
`num_envs`, at most one episode length (250 env steps for the hover
task) of batched stepping. That is small next to training: with 64
envs, it took 1.0 s against 29.2 s for 200 PPO epochs. Worth it when the
exploration noise in Options 1 and 2 dominates the EA signal.

**Which to use.** Option 1 is enough for the tutorial-sized smoke.
Option 2 gives the same number without parsing filenames and lets you
smooth over epochs. Only Option 3 removes exploration noise.

> **Why the hover reward is shaped the way it is.** A fitness is only as
> good as the reward behind it. Your task's reward must make *staying
> alive* pay, or the EA will select morphologies that fail fast. The
> shipped hover task learned this the hard way. Its original reward,
> `-distance_to_goal × step_dt` per step with an episode ending when
> the drone left the 0.1–3 m altitude band, made shorter episodes
> collect less penalty. A drone in free fall scored −0.59, against
> −6.09 for one holding hover thrust. In a 30-epoch test, training
> episodes averaged 49 of 250 steps. *(Fixed 2026-09-28: the task now
> uses Isaac Lab's quadcopter reward, `(15·(1 − tanh(d/0.8)) −
> 0.05·|v|² − 0.01·|ω|²) × step_dt`. Free fall now scores 0.54 against
> 9.95 for hover thrust. After 200 epochs × 64 envs, every deterministic
> eval episode ran the full 249 steps. These numbers were measured with
> the env's original `"wrench"` actions; in today's default `"mixer"`
> mode the same check gives 0.78 against 11.26.)* A cheap check for your own
> task: score a "do nothing useful but survive" policy and a "crash
> immediately" policy. The first must win.

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
(`rl-sb3`, `torch`). The guardrail at the end of `README.md` §3b
step 5 catches accidental leaks: it diffs `pip list` before and after
the ariel install and warns if a simulator binary moved.

### Don't share blueprint JSON paths across overlapping subprocesses

`evolve.py` writes per-individual blueprint JSON files under a
`tempfile.TemporaryDirectory(prefix="ariel_evolve_")` with unique
names (`ariel_evolve_<eval_id>.json`). If you run subprocesses in
parallel, ensure each one's JSON has a distinct path — and that
the per-individual `--experiment-prefix` is unique so the
checkpoint-filename fitness extraction finds the right run.

### Make the actuation depend on the morphology

If your env applies one abstract body force and torque, as Isaac Lab's
quadcopter example does, the evolved rotor layout never reaches the
control. The EA then has nothing to select on. We measured this in the
shipped hover task's `"wrench"` mode: 0.10 m vs 0.30 m arms changed
mean fitness by 1.6, while PPO's seed alone changed it by 8.6–20.0.
Route the policy's commands through the rotors instead. The shipped
default, `action_mode="mixer"`, converts collective thrust + 3 torques
into per-rotor thrusts with the blueprint's allocation matrix, clips
each rotor to its physical range, and applies the thrusts at the motor
links. Letting the policy command each rotor directly (`"rotor"`) is
also possible, but at the tutorial's budget PPO learned it in only 4
of 9 runs, against 7 of 9 for the mixer.

### Mass differs between the two shipped backends (open)

The same blueprint does not weigh the same in both backends. The
Isaac Lab backend takes mass and inertia from the blueprint's URDF: a
0.4 kg core plate plus arms and motors, about 0.5 kg for the tutorial
quad. The NumPy backend's `DroneConfiguration` builds its own mass
model from controller, battery, propeller and beam masses. Fitness
values are therefore not comparable across backends until they share
one mass model. This is recorded as an open item in
`IsaacLabBlueprintHoverEnvCfg.from_blueprint`.

### Isaac Sim not found in a new terminal

`ModuleNotFoundError: No module named 'isaacsim'` (or `'pxr'`) means
the env's activation hook is missing, so `conda activate` did not put
Isaac Sim on the Python path. This happens when the env was created
without `./isaaclab.sh --conda`, or before the `_isaac_sim` link
existed. The fix is in the troubleshooting list of `README.md` §3b.
In scripts and cron jobs, also
`source "$(conda info --base)/etc/profile.d/conda.sh"` before
`conda activate`.

*(Superseded 2026-09-28: this pitfall used to say that importing
`DroneGateEnv` in the Isaac Lab env needs ariel's EA orchestration
deps (sqlmodel, pydantic-settings). It doesn't: that import chain
loads neither package. What it needs is torch, gymnasium and
stable-baselines3, which `./isaaclab.sh --install` already provides.
Both the `DroneGateEnv` chain and the Blueprint chain import in the env
built by `README.md` §3b.)*

## 7. Calibration on a single machine

Measured wall times from an end-to-end run of the shipped reference
on 2026-09-28, with the hover env's default `action_mode="mixer"`
(`evolve.py` defaults: 3 gen × 4 ind × 30 PPO epochs × 16 envs; NVIDIA
RTX 2000 Ada laptop GPU with 8 GB; warm caches):

| Phase | Wall time |
|---|---|
| rl_games PPO (30 epochs × 16 envs) | 4.5–5.0 s |
| Isaac Sim startup + scene build (per subprocess) | 9.4–11.5 s |
| Per-individual total | 13.9–16.1 s |
| Per-generation total (population 4) | 56.8–58.3 s |
| Full 3-gen run | 172.3 s (~3 min) |

The hovering-check size in `README.md` §4 (`--epochs-per-eval 200
--num-envs 64`) took 38–46 s per individual, of which PPO was 29–36 s.
The full run took 498.6 s (~8 min). Earlier `"wrench"`-mode runs had
the same startup cost and slightly faster PPO (441.6 s in total).

Notes:
- The first individual is slower (16.1 s) than the ones after it
  (~14 s), due to Isaac Sim's cold-cache USD pipeline.
- The very first ever invocation in a fresh checkout can take
  several minutes as Isaac Sim builds its asset cache. Subsequent
  runs in the same machine are at the rates above.
- Wall time scales linearly with population × generations. Within an
  individual, only the PPO part grows with epochs; the ~10 s startup
  is fixed.

---

See also:
- [`README.md`](./README.md) — top-level tutorial.
- [`evolve.py`](./evolve.py) — shipped reference outer loop.
- [`train.py`](./train.py) — shipped per-individual inner loop.
- [`DRONE_BLUEPRINT_PLAN.md`](../../DRONE_BLUEPRINT_PLAN.md) §1
  "Value proposition" and §6 "Design decisions" for the deeper
  rationale.
