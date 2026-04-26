---
type: api_reference
tags: [mujoco, python, enumerations, constants]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MuJoCo Enumerations and Constants

All C enums are Python enum classes. Access as `mujoco.EnumClass.VALUE`. All C constants are accessible as `mujoco.CONSTANT_NAME`.

## `mjtObj` — Object Types

Used with `mj_name2id` / `mj_id2name`:

`mjOBJ_UNKNOWN`, `mjOBJ_BODY`, `mjOBJ_XBODY`, `mjOBJ_JOINT`, `mjOBJ_DOF`, `mjOBJ_GEOM`, `mjOBJ_SITE`, `mjOBJ_CAMERA`, `mjOBJ_LIGHT`, `mjOBJ_MESH`, `mjOBJ_SKIN`, `mjOBJ_HFIELD`, `mjOBJ_TEXTURE`, `mjOBJ_MATERIAL`, `mjOBJ_PAIR`, `mjOBJ_EXCLUDE`, `mjOBJ_EQUALITY`, `mjOBJ_TENDON`, `mjOBJ_ACTUATOR`, `mjOBJ_SENSOR`, `mjOBJ_NUMERIC`, `mjOBJ_TEXT`, `mjOBJ_TUPLE`, `mjOBJ_KEY`, `mjOBJ_PLUGIN`

```python
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'my_body')
```

## `mjtJoint` — Joint Types

| Value | Name | DoFs | Description |
|-------|------|------|-------------|
| 0 | `mjJNT_FREE` | 7 | global position + quaternion orientation |
| 1 | `mjJNT_BALL` | 4 | orientation relative to parent (quaternion) |
| 2 | `mjJNT_SLIDE` | 1 | translation along axis |
| 3 | `mjJNT_HINGE` | 1 | rotation about axis |

## `mjtGeom` — Geom Types

`mjGEOM_PLANE`, `mjGEOM_HFIELD`, `mjGEOM_SPHERE`, `mjGEOM_CAPSULE`, `mjGEOM_ELLIPSOID`, `mjGEOM_CYLINDER`, `mjGEOM_BOX`, `mjGEOM_MESH`, `mjGEOM_ARROW`, `mjGEOM_ARROW1`, `mjGEOM_ARROW2`, `mjGEOM_LINE`, `mjGEOM_SKIN`, `mjGEOM_LABEL`, `mjGEOM_NONE`

## `mjtCamera` — Camera Types

| Value | Name | Description |
|-------|------|-------------|
| 0 | `mjCAMERA_FREE` | free-floating camera |
| 1 | `mjCAMERA_TRACKING` | tracks a body |
| 2 | `mjCAMERA_FIXED` | uses named model camera |
| 3 | `mjCAMERA_USER` | user-controlled |

## `mjtCatBit` — Scene Category Bits

| Name | Description |
|------|-------------|
| `mjCAT_STATIC` | body 0 (world frame) elements |
| `mjCAT_DYNAMIC` | other body elements |
| `mjCAT_DECOR` | decorative elements |
| `mjCAT_ALL` | all categories |

## `mjtFontScale` — Font Scales (for MjrContext)

`mjFONTSCALE_50`, `mjFONTSCALE_100`, `mjFONTSCALE_150`, `mjFONTSCALE_200`, `mjFONTSCALE_250`, `mjFONTSCALE_300`

## `mjtRndFlag` — Rendering Flags (index into `scene.flags`)

`mjRND_SHADOW`, `mjRND_WIREFRAME`, `mjRND_REFLECTION`, `mjRND_ADDITIVE`, `mjRND_SKYBOX`, `mjRND_FOG`, `mjRND_HAZE`, `mjRND_SEGMENT`, `mjRND_IDCOLOR`

## `mjtVisFlag` — Visualization Flags (index into `opt.flags`)

`mjVIS_CONVEXHULL`, `mjVIS_TEXTURE`, `mjVIS_JOINT`, `mjVIS_CAMERA`, `mjVIS_ACTUATOR`, `mjVIS_ACTIVATION`, `mjVIS_LIGHT`, `mjVIS_TENDON`, `mjVIS_RANGEFINDER`, `mjVIS_CONSTRAINT`, `mjVIS_INERTIA`, `mjVIS_SCLINERTIA`, `mjVIS_PERTFORCE`, `mjVIS_PERTOBJ`, `mjVIS_CONTACTPOINT`, `mjVIS_ISLAND`, `mjVIS_CONTACTFORCE`, `mjVIS_TRANSPARENT`, `mjVIS_AUTOCONNECT`, `mjVIS_COM`, `mjVIS_SELECT`, `mjVIS_STATIC`, `mjVIS_SKIN`, `mjVIS_FLEXVERT`, `mjVIS_FLEXEDGE`, `mjVIS_FLEXFACE`, `mjVIS_FLEXSKIN`

## `mjtIntegrator` — Integrators

| Value | Name |
|-------|------|
| 0 | `mjINT_EULER` |
| 1 | `mjINT_RK4` |
| 2 | `mjINT_IMPLICIT` |
| 3 | `mjINT_IMPLICITFAST` |

## Notable Constants

```python
mujoco.mjPI             # 3.14159...
mujoco.mjMAXVAL         # maximum value (1e10)
mujoco.mjMINMU          # minimum friction coefficient
mujoco.mjNREF           # size of solref array (2)
mujoco.mjNIMP           # size of solimp array (5)
mujoco.mjNEQDATA        # constraint equality data size
mujoco.mjVISSTRING      # visualization string array
mujoco.mjDISABLESTRING  # disable flags string array
mujoco.mjENABLESTRING   # enable flags string array
```

## Usage Example

```python
# Check joint type
if model.joint('knee').type == mujoco.mjtJoint.mjJNT_HINGE:
    print("hinge joint")

# Set geom type in MjSpec
geom.type = mujoco.mjtGeom.mjGEOM_BOX

# Visualization flag
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
```

See [[mujoco_named_access_api]], [[mujoco_simulation_functions]], [[MjvOption]].
