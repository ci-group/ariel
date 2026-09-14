---
type: ariel_reference
tags: [ariel, residual-policy, drone, reinforcement-learning, environment]
source: ariel project source code
date_ingested: 2026-07-10
---

# ResidualDroneEnv

Gymnasium-compatible environment wrapper that applies an analytical hover prior on top of a learned residual action for morphology-conditional multi-task drone control.

## Location

`examples/spear/library/envs/residual_drone_env.py`

## Purpose

Wraps `TorchDroneGateEnv` to implement the Stage-2/3 residual learning architecture:

```
action_total = clamp( prior(state; cmaes_params) + α · residual_action )
```

The PPO actor in Stage 3 training only ever sees/produces `residual_action`. The prior is owned by the env, allowing a single multi-task residual policy to cover 5 tasks (hover, figure8, slalom, shuttle-run, circle) across 100 morphologies. The observation space is extended with task one-hot encoding and morphology features, with critic-only access to prior trustworthiness (cmaes_params + median_score).

## Signature

```python
class ResidualDroneEnv(TorchDroneGateEnv):
    def __init__(
        self,
        morph: dict,
        *,
        task: str = "hover",
        alpha: float | None = None,
        num_envs: int = 1,
        max_steps: int = 600,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        ...

    def step_async(self, residual_actions: np.ndarray) -> None:
        ...

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        ...

    def reset(self) -> np.ndarray:
        ...
```

## Parameters (Constructor)

| Name        | Type            | Default   | Description                                                                                                                                                                                                                                                                      |
| ----------- | --------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `morph`     | `dict`          | —         | Morphology descriptor with keys: `propellers` (list of dicts), `mass` (float), `inertia` (3×3 array), `prop_size` (int), `cmaes_params` (N+5 array), `morph_features` (22d array), `median_score` (float, optional). Typically one row of `__data__/hex_library/v1/library.npz`. |
| `task`      | `str`           | `"hover"` | Task name from `TASK_NAMES` = ("hover", "figure8", "slalom", "shuttle-run", "circle"). Determines gate configuration and reward shaping.                                                                                                                                         |
| `alpha`     | `float \| None` | `None`    | Residual scaling factor. If `None`, uses `TASK_ALPHA[task]`. Hover uses 0.10 (prior is near-perfect); trajectory tasks use 0.40.                                                                                                                                                 |
| `num_envs`  | `int`           | `1`       | Per-worker batch size (one VecEnv instance; Stage 3 runs multiple).                                                                                                                                                                                                              |
| `max_steps` | `int`           | `600`     | Episode length before auto-termination.                                                                                                                                                                                                                                          |
| `device`    | `str`           | `"cpu"`   | PyTorch device for tensors (`"cpu"`, `"cuda:0"`, etc.).                                                                                                                                                                                                                          |
| `seed`      | `int \| None`   | `None`    | Random seed for gate initialization.                                                                                                                                                                                                                                             |

## Class Attributes

### Task Configuration

| Attribute | Type | Description |
|-----------|------|-------------|
| `TASK_NAMES` | tuple[str] | `("hover", "figure8", "slalom", "shuttle-run", "circle")` |
| `NUM_TASKS` | int | 5 |
| `MORPH_FEAT_DIM` | int | 22 (hand-crafted morphology descriptor dimension) |
| `PRIOR_PARAM_DIM` | int | 11 (N + 5 for hex: N trims + 5 gains) |
| `PRIOR_TAIL_DIM` | int | 12 (PRIOR_PARAM_DIM + 1 for median_score) |

### Per-Task Residual Scaling

```python
TASK_ALPHA = {
    "hover":       0.10,     # Prior is near-perfect; residual has small budget
    "figure8":     0.40,     # Residual has full swing for banking/tracking
    "slalom":      0.40,
    "shuttle-run": 0.40,
    "circle":      0.40,
}
```

The per-task alpha allows hover to remain tightly controlled while trajectory tasks can learn significant departures from the prior.

### Per-Task Prior Gain Scaling

```python
TASK_PRIOR_GAIN_SCALE = {
    "hover":       np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "figure8":     np.array([1.0, 1.0, 0.3, 0.5, 0.5]),
    "slalom":      np.array([1.0, 1.0, 0.3, 0.5, 0.5]),
    "shuttle-run": np.array([1.0, 1.0, 0.3, 0.5, 0.5]),
    "circle":      np.array([1.0, 1.0, 0.3, 0.5, 0.5]),
}
```

Scales the 5 CMA-ES gains `[k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate]` per task. For trajectory tasks, `k_tilt` is scaled to 0.3 (weakening attitude leveling) so the residual can produce sustained banking without fighting the prior. Rate damping and altitude tracking remain active for safety.

### Reward Shaping Constants

| Attribute | Value | Description |
|-----------|-------|-------------|
| `UPRIGHT_BONUS` | 0.002 | Bonus per step for staying upright (0.0 for hover, UPRIGHT_BONUS for trajectory). v5: cut from 0.01 — the unconditional bonus paid ~+10/1000 steps, making "level drifting" a local optimum. |
| `TILT_TERMINATE_COS` | 0.0 | Cosine of angle for tilt termination (disabled; altitude floor + NaN checks remain) |
| `EXTRA_YAW_RATE_PEN` | 0.005 | Extra penalty on yaw rate |
| `VELOCITY_REWARD_COEF` | 0.03 | Velocity reward magnitude (0.0 for hover, VELOCITY_REWARD_COEF for trajectory). v5: raised from 0.005 (6×) so velocity-toward-gate dominates the dense signal. Prior-only drift on figure8 now earns +1.56/1000 steps (was ~+20). |
| `ALTITUDE_FLOOR_Z` | -0.5 | Floor altitude in NED meters |
| `ALTITUDE_FLOOR_COEF` | 0.5 | Penalty weight below floor (0.0 for hover, ALTITUDE_FLOOR_COEF for trajectory) |
| `RESIDUAL_L2_PENALTY` | 0.0 | Removed in v4; prior no longer fights residual |

## Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `task_name` | `str` | The task (e.g., "hover") |
| `task_id` | `int` | Index into `TASK_NAMES` |
| `alpha` | `float` | Effective residual scaling for this task |
| `prior` | `HoverPrior` | Analytical controller instance |
| `observation_space` | `gymnasium.spaces.Box` | Extended to `(base_dim + NUM_TASKS + MORPH_FEAT_DIM + PRIOR_TAIL_DIM,)` |

## Observation Space

Extends parent observation with static per-env features:

```
obs = [
    gate_drone_obs (26d)        # from TorchDroneGateEnv
    task_one_hot (5d)           # one-hot encoding of task_id
    morph_features (22d)        # hand-crafted morphology descriptor
    cmaes_params (11d)          # ← critic-only (NOT fed to actor)
    median_score (1d, /100)     # ← critic-only (prior trustworthiness)
]
# total: 26 + 5 + 22 + 11 + 1 = 65d
```

The policy's `_split` in `37_train_residual_mtrl.py` routes the trailing 12d (cmaes_params + median_score) to the critic input exclusively, preventing the actor from learning to selectively undo the prior.

## Action Space

Input to `step_async()` is a residual action in `[-1, 1]^N`. The env composes:

```python
effort = prior.prior_effort(state, scaled_cmaes_params)
total_action = prior.effort_to_action(effort + alpha * residual_action)
```

The resulting `total_action` is clamped to `[-1, 1]` and passed to the drone dynamics.

## Methods

### `step_async(residual_actions: np.ndarray) -> None`

Asynchronously queue action(s) by composing residual with prior.

**Parameters:**
- `residual_actions`: `(num_envs, N)` where N = number of motors

**Behavior:**
- Converts residual to torch tensor on device
- Calls `prior.prior_effort()` to get analytical prior effort
- Adds scaled residual: `total = effort + alpha * residual`
- Applies prior's clamp/scale via `effort_to_action()`
- Stores in `self.actions_t` for downstream dynamics

### `step_wait() -> tuple[obs, rewards, dones, infos]`

Retrieve step results and expand observations.

**Hover task override:**
- Parent's gate-distance reward would be ~0 (prior already at target)
- Instead, applies shaped hover reward from `_hover_reward()` function
- Reward: `≈ 0.0125/step → ≈ +15 per 1200 steps at best`

**Trajectory tasks:**
- Inherits parent's gate-distance + velocity + upright + altitude shaping
- Prior provides altitude tracking + light rate damping
- Attitude leveling is weakened per task so residual can bank freely

**Returns:**
- `obs`: `(num_envs, full_obs_dim)` expanded observations
- `rewards`: `(num_envs,)` shaped reward per task
- `dones`: `(num_envs,)` episode termination flags
- `infos`: list of dicts with episode metadata

### `reset() -> np.ndarray`

Reset environment state and return initial observation.

**Key detail:**
- Overrides parent's motor initialization from `w=0` (mid-throttle) to `w_hover_norm` (hover-equivalent normalized speed)
- High-TWR morphs would otherwise explode at t=0 if initialized at zero RPM
- Re-pulls observation after motor-state edit so gate obs reflects correct motor speeds

**Returns:**
- `obs`: `(num_envs, full_obs_dim)` initial observation

## Notes

### Design Decisions

1. **Per-task α and gain scaling:** Allows a single residual policy to cover diverse task types. Hover's tight prior budget prevents destabilization; trajectory tasks can learn banking.

2. **Critic-only prior descriptor:** The actor never sees `cmaes_params` or `median_score`, preventing it from learning to ignore the prior on difficult morphologies. The critic uses this signal to estimate value more tightly.

3. **Motor-w reset fix:** Initializing to `w=0` (mid-throttle normalization) caused high-TWR hexes to immediately crash. Solution: pre-compute hover-equivalent normalized motor speed and apply at reset. This single fix increased library build pass rate from ~40% to ~100%.

4. **Tilt termination disabled:** Early versions terminated at ≈85° tilt. Diagnostics showed ~97.5% of episode terminations across all tasks were tilt-induced — the threshold was the binding safety net even on racing tasks where banking past 90° is desirable. Removed in favor of NaN-divergence, altitude floor, and OOB bounds.

### Relationship to [[Residual_Policy_Learning]]

This implementation follows the [[Residual_Policy_Learning]] framework, with two extensions:
- **Per-task residual scaling:** Each task has its own α, allowing task-specific authority budgets
- **Multi-morphology conditioning:** Observation includes morphology features so one policy generalizes across 100+ morphologies

### Common Pitfalls

- **Forgetting to scale gains per task:** Without `TASK_PRIOR_GAIN_SCALE`, the hover-tuned attitude leveler actively fights trajectory residual learning. Set `k_tilt=0.3` for racing tasks.
- **Prior initialization:** Verify step-0 reward is high without residual. If not, `HoverPrior` is mis-wired.
- **High-TWR morphs crashing at reset:** Use the `w_hover_norm` initialization, not `w=0`.

### Stage 3 Integration

Called by `37_train_residual_mtrl.py` via `MorphRotatingVecEnv`, which:
- Instantiates one `ResidualDroneEnv` per worker
- Rotates (morph, task) pairs across workers each episode
- Feeds observations to a shared PPO actor (with `_split` to hide prior descriptor from actor)

## See Also

- [[HoverPrior]] — the analytical controller class
- [[Residual_Policy_Learning]] — the algorithm framework
- `prior_controller.py` — HoverPrior implementation
- `37_train_residual_mtrl.py` — Stage-3 MTRL trainer using this env
- `morphology_conditioned_control.md` — the conceptual context
