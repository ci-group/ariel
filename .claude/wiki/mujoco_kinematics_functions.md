---
type: api_reference
tags: [mujoco, python, functions, kinematics]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MuJoCo Kinematics Functions

Forward kinematics substage functions. These compute body positions, orientations, tendon/actuator lengths, and camera/light positions from `data.qpos`. All modify `data` in place and return `None`.

## Functions

| Function | Description |
|----------|-------------|
| `mj_kinematics(m, d)` | Forward kinematics: computes `xpos`, `xquat`, `xmat` for bodies, `geom_xpos`, `geom_xmat`, `site_xpos`, `site_xmat` |
| `mj_comPos(m, d)` | CoM positions and composite inertias: computes `xipos`, `ximat`, `cinert` |
| `mj_tendon(m, d)` | Tendon lengths and moment arms |
| `mj_transmission(m, d)` | Actuator lengths (`actuator_length`) and moment arms (`actuator_moment`) |
| `mj_camlight(m, d)` | Update camera and light positions/orientations in world frame |

## Computed Quantities

After `mj_kinematics(model, data)`:

| `data` field | Shape | Description |
|--------------|-------|-------------|
| `xpos` | `(nbody, 3)` | Cartesian body positions (world frame) |
| `xquat` | `(nbody, 4)` | Cartesian body orientations (quaternion) |
| `xmat` | `(nbody, 9)` | Cartesian body orientations (3×3 row-major) |
| `xipos` | `(nbody, 3)` | Cartesian body CoM positions |
| `ximat` | `(nbody, 9)` | Cartesian body CoM orientations |
| `geom_xpos` | `(ngeom, 3)` | Cartesian geom positions |
| `geom_xmat` | `(ngeom, 9)` | Cartesian geom orientations |
| `site_xpos` | `(nsite, 3)` | Cartesian site positions |
| `site_xmat` | `(nsite, 9)` | Cartesian site orientations |
| `cam_xpos` | `(ncam, 3)` | Cartesian camera positions |
| `cam_xmat` | `(ncam, 9)` | Cartesian camera orientations |

## Example

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('model.xml')
data  = mujoco.MjData(model)

# Set joint angles
data.qpos[7] = 0.5   # some joint angle

# Compute only kinematics (no dynamics)
mujoco.mj_kinematics(model, data)
mujoco.mj_comPos(model, data)

# Read results
body_pos = data.body('hand').xpos.copy()   # (3,) world position
site_pos = data.site('fingertip').xpos.copy()
```

## Note on Ordering

`mj_fwdPosition` (called inside `mj_forward` / `mj_step`) runs:
1. `mj_kinematics`
2. `mj_comPos`
3. `mj_tendon`
4. `mj_transmission`
5. `mj_camlight`

If you only need kinematics (not full dynamics), call these individually.

See [[mujoco_simulation_functions]], [[mujoco_inertia_functions]], [[MjData]].
