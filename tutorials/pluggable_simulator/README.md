# Pluggable simulator backends for drone evolution + RL

This tutorial demonstrates how ariel decouples the **EA + PPO learning loop**
from the **physics simulator** so that collaborators can plug in their own
simulators while reusing ariel's evolutionary and learning infrastructure.

The same PPO trainer drives any backend that implements the
`BlueprintGateEnv` Protocol; today two backends ship (NumPy and a stub for
Isaac Lab), and a third — yours — can be added without changing the trainer.

---

## 1. The architecture

```
ariel offers: EA loop + PPO trainer + DroneBlueprint IR
─────────────────────────────────────────────────────────
       genome handlers │ EA operators │ gate-train PPO
                              │
                              ▼
          ─── Plug point: BlueprintGateEnv Protocol ───
                              │
   ┌──────────────────────────┼──────────────────────────┐
   ▼                          ▼                          ▼
 NumpyBlueprint        IsaacLabBlueprint           <Your simulator>
   GateEnv               GateEnv  (stub)             GateEnv
   │                          │                          │
   ▼                          ▼                          ▼
blueprint_to_propellers  blueprint_to_urdf →      whatever conversion
   │                     UrdfConverter →                 your backend
   ▼                     parallel envs                   needs
DroneSimulator
(pure NumPy)
```

**What ariel provides** (above the plug point):
- `DroneBlueprint` — the morphology IR every backend consumes.
- The PPO loop in `gate_train.py` / this tutorial's `train.py`.
- EA operators, genome handlers, repair, inspection — all simulator-agnostic.

**What a simulator backend provides** (below the plug point):
- A gymnasium `VecEnv` constructed from a `DroneBlueprint`.
- Per-step physics, reward, termination, and observation packaging.
- That's it.

---

## 2. The contract: `BlueprintGateEnv`

The Protocol lives in
[`src/ariel/simulation/tasks/blueprint_gate_env.py`](../../src/ariel/simulation/tasks/blueprint_gate_env.py)
and is intentionally minimal:

```python
@runtime_checkable
class BlueprintGateEnv(Protocol):
    blueprint: DroneBlueprint
    num_envs: int
    # ...plus the standard VecEnv methods inherited from
    # stable_baselines3.common.vec_env.VecEnv.
```

A conforming class:

1. Accepts a `DroneBlueprint` at construction.
2. Exposes `.blueprint` and `.num_envs`.
3. Implements the standard `VecEnv` interface (`reset`, `step_async`,
   `step_wait`, `close`, `get_attr`, `set_attr`, `env_method`,
   `env_is_wrapped`) — usually by inheriting from
   `stable_baselines3.common.vec_env.VecEnv` and doing the obvious thing.

Because the Protocol is `@runtime_checkable`, `isinstance(env, BlueprintGateEnv)`
works as a sanity assertion (`train.py` does this before handing the env to PPO).

---

## 3. Running the shipped backends

### NumPy backend (works today)

```bash
python tutorials/pluggable_simulator/train.py \
    --simulator numpy \
    --preset quad \
    --num-envs 8 \
    --total-timesteps 5000
```

This wraps the existing `DroneSimulator` (pure NumPy + SymPy) via
`NumpyBlueprintGateEnv` — a thin shim that calls `blueprint_to_propellers`
and forwards to the established `DroneGateEnv`. >100× real-time on CPU.

### Isaac Lab backend (stub; Phase 2)

```bash
python tutorials/pluggable_simulator/train.py \
    --simulator isaaclab \
    --preset quad \
    --num-envs 64 \
    --total-timesteps 50000
```

Currently raises `NotImplementedError` with a pointer to the Phase 2 plan.
When implemented, it will:

1. Run `blueprint_to_urdf` in-process to produce a URDF.
2. Call Isaac Lab's `UrdfConverter` to produce a USD.
3. Spawn N parallel articulation instances in Isaac Sim.
4. Apply per-motor thrust each step via a first-order motor model
   (lifted from soft_airframe's `morphy_simulator.py`).

This is unblocked because ariel and Isaac Lab now share one conda env
(see [DRONE_BLUEPRINT_PLAN.md §6 entry 15](../../DRONE_BLUEPRINT_PLAN.md)).

---

## 4. Adding your own simulator

Five steps:

**1. Create `src/ariel/simulation/tasks/<your_backend>_gate_env.py`.**

```python
from stable_baselines3.common.vec_env import VecEnv
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint

class YourBackendBlueprintGateEnv(VecEnv):
    def __init__(self, *, blueprint: DroneBlueprint, num_envs: int, **kwargs):
        # Convert the Blueprint into whatever your simulator needs.
        # Available helpers in ariel.body_phenotypes.drone.backends:
        #   - blueprint_to_propellers(bp)   → list[dict] (motor positions/dirs)
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

**2. Define the gate-passing task** in your env — gate positions, reward
shaping, observation/action spaces. The shipped envs use the gate layout in
[`src/ariel/simulation/tasks/drone_gate_env.py:22-34`](../../src/ariel/simulation/tasks/drone_gate_env.py)
as the reference; matching that layout means trained policies can be
compared backend-to-backend.

**3. Register your backend in `train.py`** by adding one branch to
`make_env()`:

```python
if simulator == "your_backend":
    from ariel.simulation.tasks.your_backend_gate_env import YourBackendBlueprintGateEnv
    return YourBackendBlueprintGateEnv(blueprint=blueprint, num_envs=num_envs, **kwargs)
```

**4. Verify Protocol conformance** by running:

```bash
python tutorials/pluggable_simulator/train.py --simulator your_backend \
    --total-timesteps 1000 --num-envs 2
```

If the assert at the top of `main()` fires, the env doesn't satisfy the
Protocol — typically because `self.blueprint` or `self.num_envs` wasn't
set, or one of the VecEnv methods is missing.

**5. Plug into the EA evaluator.** Once the env trains a policy, point
`ariel.ec.drone.evaluators.gate_evaluator.GateEvaluator` (or your own
evaluator) at the new backend. The EA loop never sees the simulator
choice — it just gets fitness numbers back per individual.

---

## 5. Why this matters

The ARIEL consortium's collaborators bring their own simulators: MuJoCo,
Aerial Gym, Isaac Lab, IsaacGym, custom in-house stacks. The Protocol
seam means each group keeps their preferred simulator while sharing
ariel's evolutionary and learning infrastructure — decoders, EA
operators, repair, descriptors, training boilerplate. One IR
(`DroneBlueprint`), one EA loop, many backends.

See [DRONE_BLUEPRINT_PLAN.md](../../DRONE_BLUEPRINT_PLAN.md) §1 ("Value
proposition") and §6 entry 17 for the broader rationale.
