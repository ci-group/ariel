---
type: api_reference
tags: [mujoco, python, functions, simulation, dynamics]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MuJoCo Simulation Functions

Core simulation stepping and forward/inverse dynamics functions. All take `(model, data)` and return `None`, modifying `data` in place.

## Top-Level Stepping

```python
mujoco.mj_step(model, data)
# Full step: forward dynamics + numerical integration.
# Equivalent to mj_step1 + mj_step2.

mujoco.mj_step1(model, data)
# First half-step: position/velocity/control (before integration).
# Internals: mj_checkPos, mj_checkVel, mj_fwdPosition, mj_sensorPos,
#            mj_energyPos, mj_fwdVelocity, mj_sensorVel, mj_energyVel,
#            then fires mjcb_control.

mujoco.mj_step2(model, data)
# Second half-step: actuation/acceleration/constraints + integrate.
# Internals: mj_fwdActuation, mj_fwdAcceleration, mj_fwdConstraint,
#            mj_sensorAcc, mj_checkAcc, then Euler or implicit integration.

mujoco.mj_forward(model, data)
# Full forward dynamics without integration (computes accelerations).
```

## Forward Dynamics Substages

All have signature `(model: MjModel, data: MjData) -> None`.

| Function | Description |
|----------|-------------|
| `mj_fwdPosition(m, d)` | kinematics, flex, tendon, transmission |
| `mj_fwdVelocity(m, d)` | velocity-dependent forces (Coriolis, damping) |
| `mj_fwdActuation(m, d)` | actuator forces and activations |
| `mj_fwdAcceleration(m, d)` | smooth (unconstrained) acceleration |
| `mj_fwdConstraint(m, d)` | constraint solver (PGS/CG/Newton), updates `qacc` |
| `mj_collision(m, d)` | collision detection, populates `data.contact` |
| `mj_makeConstraint(m, d)` | build constraint Jacobians and residuals |
| `mj_projectConstraint(m, d)` | project constraint Jacobians into constraint space |
| `mj_referenceConstraint(m, d)` | compute constraint reference accelerations |
| `mj_constraintUpdate(m, d, jar, cost, flg_coneHessian)` | one iteration of constraint solver |

## Inverse Dynamics

```python
mujoco.mj_inverse(model, data)
# Inverse dynamics: computes qfrc_inverse from data.qacc.

mujoco.mj_rne(model, data, flg_acc, result)
# Recursive Newton-Euler: computes M*qacc + C(q, qvel).

mujoco.mj_rnePostConstraint(model, data)
# RNE with final computed constraint forces.
```

## State Management

```python
mujoco.mj_resetData(model, data)
# Reset data to default initial state.

mujoco.mj_resetDataKeyframe(model, data, key_id)
# Reset data to a specific keyframe (int index).

mujoco.mj_getState(model, data, state, spec)
# Copy state vector out (into numpy array `state`).

mujoco.mj_setState(model, data, state, spec)
# Set state from vector.

size = mujoco.mj_stateSize(model, spec)
# Return size of state vector for given spec.
```

## Name / ID Lookup

```python
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'my_body')
name    = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
```

## Examples

```python
import mujoco

model = mujoco.MjModel.from_xml_path('model.xml')
data  = mujoco.MjData(model)

# Run 1000 steps
for _ in range(1000):
    mujoco.mj_step(model, data)

# Use step1/step2 to insert control between halves
def control_loop(model, data):
    mujoco.mj_step1(model, data)
    data.ctrl[0] = -1.0 * data.qpos[0]   # custom control
    mujoco.mj_step2(model, data)

# Compute kinematics only (no dynamics)
mujoco.mj_kinematics(model, data)
```

See also [[mujoco_kinematics_functions]], [[mujoco_inertia_functions]], [[mujoco_callbacks]].
