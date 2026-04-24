---
type: api_reference
tags: [mujoco, python, class, ariel]
source: https://ci-group.github.io/ariel/source/Mujoco_docs/mujoco_docs.html
---
# MjData

`MjData` is MuJoCo's runtime simulation state object. It holds all quantities that change at every timestep — positions, velocities, accelerations, forces, sensor readings, and spatial transforms. It is updated in-place by `mj_step()` and related functions.

## Signature

```python
mujoco.MjData(model: MjModel)
```

## Variable Categories

| Category | Variables | Page |
|---|---|---|
| State | `qpos`, `qvel`, `qacc` | [[MjData_State_Variables]] |
| Actuators | `act`, `act_dot`, `actuator_force`, `actuator_length`, `actuator_velocity`, `ctrl` | [[MjData_Actuator_Variables]] |
| Forces | `qfrc_actuator`, `qfrc_applied`, `qfrc_bias`, `qfrc_constraint`, `qfrc_passive`, `qfrc_spring`, `cfrc_ext`, `cfrc_int` | [[MjData_Force_Variables]] |
| Spatial | `xpos`, `xmat`, `xquat`, `geom_xpos`, `geom_xmat`, `site_xpos`, `site_xmat`, `cam_xpos`, `cam_xmat`, `xfrc_applied` | [[MjData_Spatial_Variables]] |
| Sensors / Other | `sensordata`, `mocap_pos`, `mocap_quat`, `energy`, `M`, `bind` | [[MjData_Sensor_Variables]] |

## Notes

- All arrays are NumPy views into contiguous MuJoCo-managed memory.
- Writing to these arrays (e.g. `data.qpos[0] = x`) directly affects the simulation state.
- `bind` provides a step-by-step data access helper for tracking specific geometry objects.

## See Also

- [[MjData_State_Variables]]
- [[MjData_Actuator_Variables]]
- [[MjData_Force_Variables]]
- [[MjData_Spatial_Variables]]
- [[MjData_Sensor_Variables]]
