---
type: api_reference
tags: [mujoco, python, mjdata, actuator, ariel]
source: https://ci-group.github.io/ariel/source/Mujoco_docs/mujoco_docs.html
---
# MjData — Actuator Variables

Arrays in [[MjData]] that track the state and output of actuators, plus the control input vector.

---

## `ctrl` — Control Signals

**Definition:** Input control vector. Write desired actuator commands here before calling `mj_step`.

**Type:** `numpy.ndarray`, shape `(nu,)` where `nu = model.nu`

**Example:**

```python
data.ctrl[0] = 1.0   # set first actuator command
data.ctrl[:] = 0.0   # zero all controls
```

---

## `act` — Actuator Activation States

**Definition:** Internal activation states for stateful actuators (e.g. muscles). For stateless actuators this is unused.

**Type:** `numpy.ndarray`, shape `(na,)` where `na = model.na`

---

## `act_dot` — Activation Time-Derivative

**Definition:** Time-derivative of `act`. Computed during `mj_step`.

**Type:** `numpy.ndarray`, shape `(na,)`

---

## `actuator_force` — Actuator Forces

**Definition:** Scalar force/torque produced by each actuator along its transmission direction.

**Type:** `numpy.ndarray`, shape `(nu,)`

---

## `actuator_length` — Actuator Length

**Definition:** Current length of each actuator (used by muscles and length-dependent actuators).

**Type:** `numpy.ndarray`, shape `(nu,)`

---

## `actuator_velocity` — Actuator Velocity

**Definition:** Time-derivative of `actuator_length`; rate of change of each actuator's length.

**Type:** `numpy.ndarray`, shape `(nu,)`

---

## See Also

- [[MjData]]
- [[MjData_Force_Variables]]
- [[MjData_State_Variables]]
