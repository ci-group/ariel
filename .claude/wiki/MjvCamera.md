---
type: api_reference
tags: [mujoco, python, class, rendering, visualization]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MjvCamera

Abstract visualizer camera. Used with [[mujoco_rendering_pipeline]] to define the viewpoint.

## Signature

```python
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)   # initialize to defaults
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `type` | `mjtCamera` int | camera mode (FREE=0, TRACKING=1, FIXED=2, USER=3) |
| `fixedcamid` | int | model camera id (used when `type=FIXED`) |
| `trackbodyid` | int | body id to track (used when `type=TRACKING`) |
| `lookat` | `(3,)` numpy | 3D point being viewed |
| `distance` | float | distance from camera to lookat point |
| `azimuth` | float | horizontal angle in degrees |
| `elevation` | float | vertical angle in degrees |

## Camera Types (`mjtCamera`)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `mjCAMERA_FREE` | free-floating camera controlled by azimuth/elevation/distance |
| 1 | `mjCAMERA_TRACKING` | follows a body (set `trackbodyid`) |
| 2 | `mjCAMERA_FIXED` | uses a named model camera (set `fixedcamid`) |
| 3 | `mjCAMERA_USER` | user-controlled |

## Example

```python
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)

cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.0, 0.0, 0.5]
cam.distance  = 3.0
cam.azimuth   = 90.0
cam.elevation = -30.0
```

## Related Functions

```python
mujoco.mjv_defaultCamera(cam)              # initialize to defaults
mujoco.mjv_defaultFreeCamera(model, cam)   # initialize free camera for model extent
mujoco.mjv_moveCamera(model, action, reldx, reldy, scene, cam)  # interactive move
```

See [[MjvOption]], [[MjvScene]], [[mujoco_rendering_pipeline]].
