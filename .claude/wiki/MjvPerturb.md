---
type: api_reference
tags: [mujoco, python, class, visualization, interaction]
source: https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjvisualize.h
date_ingested: 2026-04-13
---
# MjvPerturb

Object selection and perturbation handle used in interactive visualization. Tracks which body the user has selected and stores the reference pose used to drag or apply forces to it.

## Signature

```python
pert = mujoco.MjvPerturb()
mujoco.mjv_defaultPerturb(pert)   # initialize to defaults
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `select` | int | Selected body id; ≤ 0 means nothing selected |
| `flexselect` | int | Selected flex id; -1 means none |
| `skinselect` | int | Selected skin id; -1 means none |
| `active` | int | Perturbation bitmask (`mjtPertBit`) — what is currently active |
| `active2` | int | Secondary perturbation bitmask |
| `refpos` | `(3,)` float | Reference position for the selected object |
| `refquat` | `(4,)` float | Reference orientation (quaternion) for the selected object |
| `refselpos` | `(3,)` float | Selection point position in world coordinates |
| `localpos` | `(3,)` float | Selection point in object-local coordinates |
| `localmass` | float | Spatial inertia at selection point |
| `scale` | float | Mouse motion to model-space scaling factor |

## `mjtPertBit` — Perturbation Modes

| Value | Name | Description |
|-------|------|-------------|
| 1 | `mjPERT_TRANSLATE` | Translate selected body |
| 2 | `mjPERT_ROTATE` | Rotate selected body |

Set `pert.active` to one of these (or a bitwise OR of both) to activate perturbation.

## Key Perturbation Functions

| Function | Description |
|----------|-------------|
| `mjv_defaultPerturb(pert)` | Initialize to defaults (nothing selected, active=0) |
| `mjv_initPerturb(m, d, scn, pert)` | Set refpos/refquat to selected body's current pose |
| `mjv_movePerturb(m, d, action, reldx, reldy, scn, pert)` | Mouse hook — update refpos/refquat from mouse delta |
| `mjv_applyPerturbPose(m, d, pert, flg_paused)` | Write pert pose into `d.mocap` or `d.qpos` |
| `mjv_applyPerturbForce(m, d, pert)` | Write pert force/torque into `d.xfrc_applied` |
| `mjv_select(m, d, vopt, aspect, relx, rely, scn, selpnt, geomid, flexid, skinid)` | Ray-cast to select object under cursor |

## Usage Pattern

```python
import mujoco

pert = mujoco.MjvPerturb()
mujoco.mjv_defaultPerturb(pert)

# After user clicks on a body (relx, rely are normalized window coords):
selpnt = np.zeros(3)
geomid = np.array([-1])
flexid = np.array([-1])
skinid = np.array([-1])
bodyid = mujoco.mjv_select(
    model, data, opt,
    viewport.width / viewport.height,   # aspect ratio
    relx, rely,
    scene,
    selpnt, geomid, flexid, skinid
)

if bodyid > 0:
    pert.select = bodyid
    pert.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
    mujoco.mjv_initPerturb(model, data, scene, pert)

# Each frame while dragging (reldx, reldy are mouse delta / window size):
mujoco.mjv_movePerturb(model, data, action, reldx, reldy, scene, pert)
mujoco.mjv_applyPerturbPose(model, data, pert, flg_paused=0)
mujoco.mjv_applyPerturbForce(model, data, pert)
```

## Notes

- `mjv_applyPerturbPose` writes to `d.mocap` when the body is a mocap body, and to `d.qpos` only if `flg_paused=1` and the subtree root has a free joint.
- `mjv_applyPerturbForce` writes to `d.xfrc_applied` only if the selected body is dynamic (non-world, non-mocap).
- Pass `pert` (or `None`) as the 4th argument to `mjv_updateScene`; if `None`, no perturbation arrows are drawn.
- The passive viewer exposes `handle.pert` as an `MjvPerturb` instance.

See [[mujoco_visualization_functions]], [[mujoco_viewer]], [[MjvScene]].
