---
type: source_summary
tags: [mujoco, ariel, mjdata, simulation]
source: https://ci-group.github.io/ariel/source/Mujoco_docs/mujoco_docs.html
---
# MuJoCo Data Class Variables (ARIEL)

Reference documentation for the MuJoCo `MjData` class variables as used in the ARIEL framework. Covers all runtime state variables updated each simulation step — positions, velocities, forces, sensor data, and spatial transforms.

## Pages Created

- [[MjData]] — Overview of the MjData class and its role in simulation
- [[MjData_State_Variables]] — `qpos`, `qvel`, `qacc` (generalized position, velocity, acceleration)
- [[MjData_Actuator_Variables]] — `act`, `act_dot`, `actuator_force`, `actuator_length`, `actuator_velocity`, `ctrl`
- [[MjData_Force_Variables]] — `qfrc_*` and `cfrc_*` force arrays
- [[MjData_Spatial_Variables]] — `xpos`, `xmat`, `xquat`, `geom_xpos`, `site_xpos`, `cam_xpos`, `xfrc_applied`
- [[MjData_Sensor_Variables]] — `sensordata`, `mocap_pos`, `mocap_quat`, `energy`, `M`, `bind`
