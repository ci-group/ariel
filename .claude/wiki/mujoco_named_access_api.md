---
type: api_reference
tags: [mujoco, python, api, named-access]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MuJoCo Named Access API

Both [[MjModel]] and [[MjData]] provide O(1) named element access. Call accessor methods with a string name; the returned object exposes all struct fields with the entity-type prefix stripped, as NumPy views into the underlying C memory.

## Accessor Methods

Available on both `model` and `data`:

`body`, `xbody`, `joint`, `dof`, `geom`, `site`, `camera`, `light`, `mesh`, `skin`, `hfield`, `tex`, `mat`, `pair`, `exclude`, `eq`, `tendon`, `actuator`, `sensor`, `numeric`, `text`, `tuple`, `key`, `plugin`

## Accessor Object Properties

| Property | Description |
|----------|-------------|
| `.id` | integer id (equivalent to `mj_name2id(...)`) |
| `.name` | string name (equivalent to `mj_id2name(...)`) |
| all struct fields | NumPy views, prefix stripped |

## Examples — Model Access

```python
# geom attributes (views into model arrays)
model.geom('floor').rgba     # shape (4,), view into model.geom_rgba[4*i:4*i+4]
model.geom('floor').id       # integer index
model.geom('floor').type     # int (mjtGeom)
model.geom('floor').size     # (3,) shape/size

model.body('upper_arm').mass     # float
model.body('upper_arm').pos      # (3,) position relative to parent

model.joint('elbow').type        # int (mjtJoint)
model.joint('elbow').qposadr     # int — address in qpos

model.actuator('motor1').id
model.camera('overhead').fovy
```

## Examples — Data Access

```python
# After mj_forward / mj_kinematics:
data.body('pelvis').xpos         # (3,) world position — view into data.xpos
data.body('pelvis').xquat        # (4,) orientation quaternion
data.body('pelvis').xmat         # (9,) rotation matrix (row-major)

data.geom('floor').xpos          # (3,) geom world position
data.geom('floor').xmat          # (9,)

data.joint('knee').qpos          # slice of data.qpos (length depends on joint type)
data.joint('knee').qvel          # slice of data.qvel
data.joint('knee').qacc          # slice of data.qacc

data.site('end_effector').xpos   # (3,) world position
data.site('end_effector').xmat   # (9,)

data.sensor('force_sensor').data # sensor output slice of data.sensordata
data.actuator('motor1').force    # scalar actuator force
```

## NumPy View Semantics

All returned arrays are **live views** into MuJoCo's C memory. Copy if you need a snapshot:

```python
pos = data.body('hand').xpos.copy()  # safe snapshot
```

## Name / ID Lookup (Alternative)

```python
# Equivalent to named access .id
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
name    = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
```

See [[MjModel]], [[MjData]], [[mujoco_enumerations]].
