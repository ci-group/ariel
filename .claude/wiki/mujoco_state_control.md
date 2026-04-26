---
type: api_reference
tags: [mujoco, python, simulation, mjdata, physics]
source: https://mujoco.readthedocs.io/en/stable/programming/simulation.html
date_ingested: 2026-04-13
---
# mujoco_state_control

MuJoCo's state model: what constitutes the simulation state, how it maps to `mjData` fields, and how to get/set/copy it using the `mjtState` bitfield.

## State Components (mjtState)

The state is partitioned into named components addressable via the `mjtState` bitfield enum. Components can be OR'd to form custom bitfields.

### Physics state (`mjSTATE_PHYSICS`)
Time-integrated quantities advanced during stepping:

| Field | Description |
|-------|-------------|
| `data.qpos` | Generalized positions (nq,) — configuration in joint coordinates |
| `data.qvel` | Generalized velocities (nv,) — not a simple time-derivative when quaternion joints exist |
| `data.act` | Actuator activations (na,) — for stateful actuators (e.g. muscles) |
| `data.history` | Timestamped buffer for actuators/sensors with `nsample > 0` (delay support) |

### Full physics state (`mjSTATE_FULLPHYSICS`)
Physics state plus:

| Field | Description |
|-------|-------------|
| `data.time` | Simulation time; `dt/dt = 1`; needed for time-indexed control laws |
| `data.plugin_state` | States declared by engine plugins |

### User inputs (`mjSTATE_USER`)
Set by user, untouched by the simulator:

| Field | Description |
|-------|-------------|
| `data.ctrl` | Actuator control signals (nu,) — either direct force or activation target |
| `data.qfrc_applied` | Directly applied generalized forces (nv,) |
| `data.xfrc_applied` | Cartesian wrenches on body CoMs (nbody, 6) |
| `data.mocap_pos` | Positions of mocap bodies (nmocap, 3) |
| `data.mocap_quat` | Orientations of mocap bodies (nmocap, 4) |
| `data.eq_active` | Per-constraint equality-constraint toggle (byte array) |
| `data.userdata` | User-defined memory block; engine never writes here |

### Warmstarts (`mjSTATE_WARMSTART`)
| Field | Description |
|-------|-------------|
| `data.qacc_warmstart` | Previous step accelerations; warmstarts the constraint solver |

> Warmstarts have negligible effect with the Newton solver (2–3 iters to convergence) but matter for PGS and for exact reproducibility when loading non-initial states.

### Integration state (`mjSTATE_INTEGRATION`)
Union of all the above. Two `mjData` instances with identical integration state will produce identical pipeline output. This is the full input to `mj_forward`.

## State Manipulation Functions

```python
import mujoco

# Size of state vector for a given component bitfield
size = mujoco.mj_stateSize(model, spec)          # spec is int (mjtState bitfield)

# Copy state to/from a flat numpy array
state = np.zeros(size)
mujoco.mj_getState(model, data, state, spec)      # data → array
mujoco.mj_setState(model, data, state, spec)      # array → data

# Copy state between two mjData objects
mujoco.mj_copyState(model, src, dst, spec)        # src → dst for selected components

# Extract state into a new array (returns array)
state = mujoco.mj_extractState(model, data, spec)
```

### Example: save/restore integration state

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('model.xml')
data  = mujoco.MjData(model)

spec = mujoco.mjtState.mjSTATE_INTEGRATION
size = mujoco.mj_stateSize(model, spec)
saved = np.zeros(size)

# Save
mujoco.mj_getState(model, data, saved, spec)

# ... run simulation, do rollouts ...

# Restore
mujoco.mj_setState(model, data, saved, spec)
```

### Example: copy state between rollout workers

```python
# Thread-safe: copy physics state from main data to worker
worker_data = mujoco.MjData(model)
mujoco.mj_copyState(model, data_main, worker_data, mujoco.mjtState.mjSTATE_INTEGRATION)
```

## Reset Functions

```python
mujoco.mj_resetData(model, data)
# Sets qpos = model.qpos0, mocap_pos/quat = fixed body poses,
# all other state/control vars to 0.

mujoco.mj_resetDataKeyframe(model, data, key_id)
# Reset to a specific keyframe (int index).
```

## Notes

- `mjSTATE_INTEGRATION` is **maximalist** — includes fields often unused. For compact state storage, explicitly select only needed components (e.g. `mjSTATE_PHYSICS | mjSTATE_USER`).
- `xfrc_applied` is (nbody, 6) and often unused — omitting it saves significant storage.
- Copying `mjData` fully with `mj_copyData` is much slower than `mj_copyState` with a targeted spec — prefer the latter for rollout-based EA evaluation loops.
- For reproducibility when loading non-initial states, always include `mjSTATE_WARMSTART` in the spec passed to `mj_copyState`.

## See Also

[[MjData]], [[MjData_State_Variables]], [[MjData_Actuator_Variables]], [[mujoco_simulation_functions]]
