---
type: api_reference
tags: [mujoco, python, mjdata, spatial, transforms, ariel]
source: https://ci-group.github.io/ariel/source/Mujoco_docs/mujoco_docs.html
---
# MjData — Spatial Variables

World-frame position, orientation, and applied Cartesian forces for bodies, geometries, sites, and cameras in [[MjData]]. All are recomputed each `mj_step` from `qpos`.

---

## Body Transforms

### `xpos`
World-frame Cartesian position of each body's origin.
- **Shape:** `(nbody, 3)`

### `xmat`
World-frame orientation of each body as a flattened 3×3 rotation matrix.
- **Shape:** `(nbody, 9)` — row-major, reshape to `(3,3)` if needed

```python
import numpy as np
body_id = model.body('torso').id
rot = data.xmat[body_id].reshape(3, 3)
pos = data.xpos[body_id]
```

### `xquat`
World-frame orientation of each body as a quaternion `[qw, qx, qy, qz]`.
- **Shape:** `(nbody, 4)`

---

## Geometry Transforms

### `geom_xpos`
World-frame position of each geometry's center.
- **Shape:** `(ngeom, 3)`

### `geom_xmat`
World-frame orientation of each geometry as a flattened 3×3 rotation matrix.
- **Shape:** `(ngeom, 9)`

---

## Site Transforms

### `site_xpos`
World-frame position of each site.
- **Shape:** `(nsite, 3)`

### `site_xmat`
World-frame orientation of each site as a flattened 3×3 rotation matrix.
- **Shape:** `(nsite, 9)`

---

## Camera Transforms

### `cam_xpos`
World-frame position of each camera.
- **Shape:** `(ncam, 3)`

### `cam_xmat`
World-frame orientation of each camera as a flattened 3×3 rotation matrix.
- **Shape:** `(ncam, 9)`

---

## Applied Cartesian Forces

### `xfrc_applied`
User-specified external forces applied to bodies in world (Cartesian) coordinates.
- **Shape:** `(nbody, 6)` — `[fx, fy, fz, tx, ty, tz]` (force + torque)

```python
# Apply a 5 N upward force on body 1
data.xfrc_applied[1, 2] = 5.0
```

---

## See Also

- [[MjData]]
- [[MjData_State_Variables]]
- [[MjData_Force_Variables]]
