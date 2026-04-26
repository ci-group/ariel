---
type: api_reference
tags: [mujoco, python, mjdata, forces, ariel]
source: https://ci-group.github.io/ariel/source/Mujoco_docs/mujoco_docs.html
---
# MjData — Force Variables

Generalized and contact force arrays in [[MjData]]. All `qfrc_*` arrays are in generalized (joint-space) coordinates with shape `(nv,)`. Contact forces `cfrc_*` are per-body.

---

## Generalized Force Arrays (`qfrc_*`)

All have shape `(nv,)` where `nv = model.nv`.

### `qfrc_actuator`
Forces contributed by actuators, projected into joint space.

### `qfrc_applied`
Externally applied generalized forces (set by user or controller).

```python
# Apply a generalized force to joint 0
data.qfrc_applied[0] = 10.0
```

### `qfrc_bias`
Bias forces: Coriolis, centripetal, and gravitational terms combined (`C(q,v) + g(q)`).

### `qfrc_constraint`
Forces due to active constraints (contacts, limits, tendons).

### `qfrc_passive`
Passive forces: joint damping and spring stiffness.

### `qfrc_spring`
Spring forces from spring/damper actuators or tendons.

---

## Contact Force Arrays (`cfrc_*`)

Shape: `(nbody, 6)` — one 6D spatial force per body.

### `cfrc_ext`
External contact forces acting on each body (wrench in world frame).

### `cfrc_int`
Internal contact forces (reaction forces within the body subtree).

---

## See Also

- [[MjData]]
- [[MjData_State_Variables]]
- [[MjData_Actuator_Variables]]
- [[MjData_Spatial_Variables]]
