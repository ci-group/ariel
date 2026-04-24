---
type: api_reference
tags: [mujoco, python, simulation, kinematics, jacobian]
source: https://mujoco.readthedocs.io/en/stable/programming/simulation.html
date_ingested: 2026-04-13
---
# mujoco_jacobians

MuJoCo Jacobian computation API. Jacobians map joint velocities to end-effector velocities and (transposed) end-effector forces to joint forces. Computing them is essentially free in MuJoCo after kinematics have been run.

## Overview

MuJoCo can differentiate analytically:
- Tendon lengths → `data.ten_moment` (moment arms)
- Actuator transmission lengths → `data.actuator_moment`
- All scalar constraint violations → `data.efc_J`

For user-defined points, use the `mj_jac` family of functions.

> **Size note:** The Jacobian is N × `model.nv`, not N × `model.nq`. When quaternion joints are present, `nq > nv`.

## mj_jac — Main Jacobian Function

```python
mujoco.mj_jac(model, data, jacp, jacr, point, body)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `MjModel` | Compiled model |
| `data` | `MjData` | Simulation data (kinematics must be current) |
| `jacp` | `ndarray (3, nv)` or `None` | Output: translational Jacobian |
| `jacr` | `ndarray (3, nv)` or `None` | Output: rotational Jacobian |
| `point` | `ndarray (3,)` | 3D point in world frame |
| `body` | `int` | ID of body the point is attached to |

Pass `None` for `jacp` or `jacr` to skip computing that component.

## Convenience Functions

| Function | Point used |
|----------|-----------|
| `mj_jacBody(m, d, jacp, jacr, body)` | Body origin |
| `mj_jacBodyCom(m, d, jacp, jacr, body)` | Body center of mass |
| `mj_jacGeom(m, d, jacp, jacr, geom)` | Geom center |
| `mj_jacSite(m, d, jacp, jacr, site)` | Site position |
| `mj_jacSubtreeCom(m, d, jacp, body)` | Subtree center of mass (translational only) |

All functions have the same `(model, data, jacp, jacr, ...)` signature pattern.

## Example: End-effector Jacobian

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('model.xml')
data  = mujoco.MjData(model)

# Run kinematics to update positions
mujoco.mj_forward(model, data)

# Get end-effector site ID and compute Jacobian
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

# Jacobian-based control: map end-effector error to joint torques
ee_error = target_pos - data.site('end_effector').xpos
joint_torques = jacp.T @ ee_error
```

## Automatically Computed Jacobians in mjData

| Field | Shape | Meaning |
|-------|-------|---------|
| `data.ten_moment` | `(ntendon, nv)` | Tendon moment arms (Jacobian of tendon lengths w.r.t. qpos) |
| `data.actuator_moment` | `(nu, nv)` | Actuator moment arms (Jacobian of transmission lengths) |
| `data.efc_J` | `(nefc, nv)` | Constraint Jacobian (dense or sparse) |
| `data.efc_JT` | `(nv, nefc)` | Transposed constraint Jacobian (only when sparse format is on) |

## Notes

- Must call at least `mj_kinematics(model, data)` before calling any `mj_jac*` function.
- In the full pipeline `mj_forward` or `mj_step` calls kinematics automatically.
- Jacobian computation is O(nv) per row — essentially free for typical robot sizes.
- `data.efc_J` sparsity depends on `mjModel.opt` settings; use `mujoco.mj_isSparse(model)` to check.

## See Also

[[mujoco_kinematics_functions]], [[mujoco_simulation_functions]], [[MjData_Force_Variables]]
