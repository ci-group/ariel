---
type: api_reference
tags: [mujoco, python, mjdata, state, ariel]
source: https://ci-group.github.io/ariel/source/Mujoco_docs/mujoco_docs.html
---
# MjData — State Variables

Generalized position, velocity, and acceleration vectors that form the core simulation state in [[MjData]].

---

## `qpos` — Generalized Position

**Definition:** Stores the complete position state of all joints and free bodies in the model.

**Layout for a free body + hinge joints:**

```
qpos = [x, y, z, qw, qx, qy, qz, angle_joint1, angle_joint2, ...]
        ^-- free joint (7 values) --^  ^-- hinge joints (1 value each) --^
```

- **Core (free joint):** `[x, y, z]` — 3D position in world frame; `[qw, qx, qy, qz]` — unit quaternion orientation
- **Hinge joints:** one angle (radians) per joint

**Type:** `numpy.ndarray`, shape `(nq,)` where `nq = model.nq`

**Example:**

```python
# Read position and orientation of the root free body
x, y, z = data.qpos[0:3]
qw, qx, qy, qz = data.qpos[3:7]

# Set a hinge joint angle
data.qpos[7] = 1.57  # ~90 degrees
```

---

## `qvel` — Generalized Velocity

**Definition:** Velocity state vector. For free bodies this is a 6-DOF spatial velocity; for hinge joints it is angular velocity in radians/second.

**Type:** `numpy.ndarray`, shape `(nv,)` where `nv = model.nv`

**Example:**

```python
# Linear velocity of free body
vx, vy, vz = data.qvel[0:3]
# Angular velocity of free body
wx, wy, wz = data.qvel[3:6]
```

---

## `qacc` — Generalized Acceleration

**Definition:** Acceleration state vector, computed from forces after each `mj_step`. Same layout as `qvel`.

**Type:** `numpy.ndarray`, shape `(nv,)`

---

## See Also

- [[MjData]]
- [[MjData_Actuator_Variables]]
- [[MjData_Force_Variables]]
