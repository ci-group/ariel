# Pluggable simulator backends for drone evolution + RL

This tutorial demonstrates how ariel decouples the **EA + RL learning loop**
from the **physics simulator** so that collaborators can plug in their
own simulators while reusing ariel's evolutionary and morphology-IR
infrastructure.

Two backends ship today:

| `--simulator` | Simulator (physics) | RL library / loop | Task |
|---------------|---------------------|---------------|----------------|
| `numpy`       | `DroneSimulator` (pure NumPy + SymPy) | **stable-baselines3 PPO** (trains end-to-end) | gate-passing |
| `isaaclab`    | Isaac Lab / Isaac Sim PhysX           | random-action env stepping (rl_games PPO wiring is Phase 2.5) | hover-to-goal |

Each backend brings its own simulator **and** its own RL library, because
that matches how real heterogeneous simulator ecosystems work — Isaac
Lab ships rsl_rl/rl_games/skrl as native trainers, the NumPy stack pairs
naturally with sb3, etc. The EA loop above never sees the simulator
choice; it just gets a fitness scalar back per individual.

**Phase 2 status:** the architectural seam is demonstrated for both
backends. The NumPy backend trains a policy with sb3 PPO end-to-end.
The Isaac Lab backend constructs the env, spawns N parallel drones,
and steps them with random actions to validate observations / rewards /
dones. Actual PPO training via `rl_games` is wired in
[`make_rl_games_agent_cfg()`](../../src/ariel/simulation/tasks/isaaclab_hover_env.py)
but currently blocked on an env-stack issue (Isaac Sim's bundled torch
tensorboard transitively imports an older TF/jax/numpy stack that
conflicts with the conda env's numpy 2). See
[DRONE_BLUEPRINT_PLAN.md §6 entry 17](../../DRONE_BLUEPRINT_PLAN.md) for
the full story and the Phase 2.5 path forward.

---

## 1. The architecture

```
ariel offers: EA loop  +  RL trainers  +  DroneBlueprint IR
─────────────────────────────────────────────────────────────
   genome handlers │ EA operators │ scripts/train.py dispatch
                              │
                              ▼
              ─── Plug point: simulator+trainer pair ───
                              │
   ┌──────────────────────────┼──────────────────────────┐
   ▼                          ▼                          ▼
NumpyBlueprintGateEnv   IsaacLabBlueprintHoverEnv   <YourBackendEnv>
 (gymnasium VecEnv)        (DirectRLEnv)             (whatever shape
       +                        +                      your simulator
sb3 PPO                  rsl_rl PPO                  needs)
       │                        │                          │
       ▼                        ▼                          ▼
blueprint_to_propellers   blueprint_to_urdf →       whatever conversion
       │                  UrdfConverter →                  your backend
       ▼                  Isaac Sim spawn                  needs
DroneSimulator
(pure NumPy)
```

**What ariel provides** (above the plug point):
- `DroneBlueprint` — the morphology IR every backend consumes.
- EA operators, genome handlers, repair, inspection, descriptors —
  all simulator-agnostic.
- A unified `train.py` dispatch that routes to backend-specific RL
  glue without exposing the choice to the EA.

**What a simulator backend provides** (below the plug point):
- A learning-ready env constructed from a `DroneBlueprint`.
- Hooks into its preferred RL library (sb3, rsl_rl, rl_games, skrl,
  or anything else).
- A task definition (reward, termination, observation/action spaces).

The seam is **deliberate**: the EA evolves the same morphology
variables either way; only the fitness function and the trained
policy are backend-specific.

---

## 2. The two contracts

ariel ships two complementary contracts, each matched to its
backend's native shape:

### 2a. `BlueprintGateEnv` Protocol (gymnasium VecEnv)

Lives in
[`src/ariel/simulation/tasks/blueprint_gate_env.py`](../../src/ariel/simulation/tasks/blueprint_gate_env.py).
Used by backends that train with **stable-baselines3** or any other
gymnasium-VecEnv-compatible RL library.

```python
@runtime_checkable
class BlueprintGateEnv(Protocol):
    blueprint: DroneBlueprint
    num_envs: int
    # ...plus the standard VecEnv methods inherited from
    # stable_baselines3.common.vec_env.VecEnv.
```

Conformance: subclass `stable_baselines3.common.vec_env.VecEnv` and
take a `DroneBlueprint` at construction. `NumpyBlueprintGateEnv` is
the shipped reference implementation.

### 2b. Isaac Lab's `DirectRLEnv` shape (lives in `isaaclab.envs`)

Used by backends that train with **Isaac Lab's native RL libraries**
(rsl_rl, rl_games, skrl). Isaac Lab provides this class hierarchy;
our `IsaacLabBlueprintHoverEnv` extends it and slots in a
Blueprint-derived USD at scene-setup time.

```python
class IsaacLabBlueprintHoverEnv(DirectRLEnv):
    cfg: IsaacLabBlueprintHoverEnvCfg
    def __init__(self, cfg, render_mode=None, **kwargs): ...
    def _setup_scene(self): ...
    def _pre_physics_step(self, actions): ...
    def _apply_action(self): ...
    def _get_observations(self): ...
    def _get_rewards(self): ...
    def _get_dones(self): ...
    def _reset_idx(self, env_ids): ...
```

Why two contracts and not one universal? Trying to force both
simulator ecosystems through a single shape (e.g., wrapping Isaac
Lab to gymnasium VecEnv via `isaaclab_rl.sb3`) drags in
stable-baselines3, which collides with the numpy-2 ABI in the
unified isaaclab conda env. Honest heterogeneity is cheaper than
forced uniformity. See [DRONE_BLUEPRINT_PLAN.md §6 entry 17](../../DRONE_BLUEPRINT_PLAN.md)
for the full rationale.

---

## 3. Running the shipped backends

### NumPy backend (gate task, sb3 PPO)

```bash
python tutorials/pluggable_simulator/train.py \
    --simulator numpy \
    --preset quad \
    --num-envs 8 \
    --total-timesteps 5000
```

Uses `DroneSimulator` (pure NumPy + SymPy) via `NumpyBlueprintGateEnv`
— a thin shim that calls `blueprint_to_propellers` and forwards to
the established `DroneGateEnv`. >100× real-time on CPU. ~6600 env-
steps/sec on 8 parallel envs.

### Isaac Lab backend (hover task, env-stepping smoke for v1)

```bash
python tutorials/pluggable_simulator/train.py \
    --simulator isaaclab \
    --headless \
    --num-envs 16 \
    --max-iterations 3
```

Requires the unified ariel+IsaacLab conda env (see
[DRONE_BLUEPRINT_PLAN.md §6 entry 15](../../DRONE_BLUEPRINT_PLAN.md)).
What v1 does:

1. Launches Isaac Sim via `AppLauncher`.
2. Runs `blueprint_to_urdf` to produce a URDF.
3. Calls `UrdfConverter` to produce a USD.
4. Spawns N parallel articulation instances in Isaac Sim.
5. Steps the env with random actions for `max_iterations × 24` steps,
   computing observations + rewards (`-distance_to_goal × step_dt`)
   + done flags per step. Verifies the whole physics + reward pipeline.

Phase 2.5 will replace step 5 with a real PPO training run via
`rl_games.torch_runner.Runner` — the config helper
[`make_rl_games_agent_cfg`](../../src/ariel/simulation/tasks/isaaclab_hover_env.py)
already returns a working agent config; wiring it through is gated on
resolving the bundled-package-vs-conda-env stack issue described
above.

---

## 4. Adding your own simulator

Five steps. The exact contract you implement depends on your RL
library of choice:

### If you're using stable-baselines3 (gymnasium VecEnv)

**1. Create `src/ariel/simulation/tasks/<your_backend>_gate_env.py`.**

```python
from stable_baselines3.common.vec_env import VecEnv
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint

class YourBackendBlueprintGateEnv(VecEnv):
    def __init__(self, *, blueprint: DroneBlueprint, num_envs: int, **kwargs):
        # Convert the Blueprint into whatever your simulator needs.
        # Helpers available in ariel.body_phenotypes.drone.backends:
        #   - blueprint_to_propellers(bp)   → list[dict] motor positions/dirs
        #   - blueprint_to_mjspec(bp)       → mujoco.MjSpec
        #   - blueprint_to_urdf(bp, path)   → URDF file
        ...
        self.blueprint = blueprint
        self.num_envs = num_envs
        super().__init__(num_envs=num_envs,
                         observation_space=...,
                         action_space=...)

    # Implement the VecEnv abstract methods:
    def reset(self):       ...
    def step_async(self, actions): ...
    def step_wait(self):   ...
    def close(self):       ...
    def get_attr(self, attr_name, indices=None): ...
    def set_attr(self, attr_name, value, indices=None): ...
    def env_method(self, method_name, *args, indices=None, **kwargs): ...
    def env_is_wrapped(self, wrapper_class, indices=None): ...
```

### If you're using Isaac Lab's native RL (rsl_rl, rl_games, skrl)

**1. Create `src/ariel/simulation/tasks/<your_backend>_<task>_env.py`.**

```python
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint

@configclass
class YourBackendEnvCfg(DirectRLEnvCfg):
    # episode_length_s, decimation, action_space, observation_space, scene, robot ...
    @classmethod
    def from_blueprint(cls, blueprint: DroneBlueprint, **kwargs):
        # Generate a USD asset from the Blueprint, slot path into self.robot.spawn.
        ...

class YourBackendEnv(DirectRLEnv):
    cfg: YourBackendEnvCfg
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        ...
    def _setup_scene(self): ...
    def _pre_physics_step(self, actions): ...
    def _apply_action(self): ...
    def _get_observations(self): ...
    def _get_rewards(self): ...
    def _get_dones(self): ...
    def _reset_idx(self, env_ids): ...
```

### Both paths: register your backend in `train.py`

**2. Add a dispatch branch in `train.py`** (a `main_<your_backend>`
function that imports your env + your RL library and runs training).

**3. Add `--simulator <your_backend>` to the choices list** so the
peek-parser routes correctly.

**4. Verify it runs:**

```bash
python tutorials/pluggable_simulator/train.py --simulator <your_backend> \
    --num-envs 2 --total-timesteps 1000   # or --max-iterations 2
```

**5. Plug into the EA evaluator.** Once training works, point
`ariel.ec.drone.evaluators.gate_evaluator.GateEvaluator` (or your own
evaluator) at the new backend. The EA loop never sees the simulator
choice — it just gets fitness numbers back per individual.

---

## 5. Why this matters

The ARIEL consortium's collaborators bring their own simulators:
MuJoCo, Aerial Gym, Isaac Lab, IsaacGym, custom in-house stacks.
Each group has a preferred RL library too. The two-contract seam
means each can keep their preferred simulator and trainer while
sharing ariel's evolutionary and morphology infrastructure —
decoders, EA operators, repair, descriptors, the morphology IR.
One IR (`DroneBlueprint`), one EA loop, many backends, many
trainers.

See [DRONE_BLUEPRINT_PLAN.md](../../DRONE_BLUEPRINT_PLAN.md) §1
("Value proposition") and §6 entry 17 for the broader rationale.
