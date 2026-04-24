---
type: api_reference
tags: [mujoco, mjspec, model-building, c-api]
source: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html
date_ingested: 2026-04-13
---
# mjsJoint

C struct representing a joint element in an [[MjSpec]] model specification. Corresponds to the MJCF `<joint>` element.

## Signature

```c
typedef struct mjsJoint_ {
  mjsElement* element;             // element type (do not modify)
  mjtJoint type;                   // joint type

  // kinematics
  double pos[3];                   // anchor position in body frame
  double axis[3];                  // joint axis (hinge/slide)
  double ref;                      // reference configuration value (qpos0)
  int align;                       // align free joint with body com (mjtAlignFree)

  // stiffness
  double stiffness;                // stiffness coefficient
  double springref;                // spring reference value: qpos_spring
  double springdamper[2];          // [timeconst, dampratio]

  // limits
  int limited;                     // joint has limits (mjtLimited)
  double range[2];                 // joint limits [min, max]
  double margin;                   // margin for limit detection
  mjtNum solref_limit[mjNREF];     // solver reference: joint limits
  mjtNum solimp_limit[mjNIMP];     // solver impedance: joint limits
  int actfrclimited;               // actuator force limits active (mjtLimited)
  double actfrcrange[2];           // actuator force limits [min, max]

  // dof properties
  double armature;                 // armature inertia (mass for slider)
  double damping;                  // damping coefficient
  double frictionloss;             // friction loss
  mjtNum solref_friction[mjNREF];  // solver reference: dof friction
  mjtNum solimp_friction[mjNIMP];  // solver impedance: dof friction

  // other
  int group;                       // group
  mjtByte actgravcomp;             // apply gravcomp via actuators
  mjDoubleVec* userdata;           // user data
  mjString* info;                  // message appended to compiler errors
} mjsJoint;
```

## Parameters

| Field | Type | Description |
|---|---|---|
| `type` | `mjtJoint` | `mjJNT_FREE`, `mjJNT_BALL`, `mjJNT_SLIDE`, `mjJNT_HINGE` |
| `pos` | `double[3]` | Anchor point position in body frame |
| `axis` | `double[3]` | Joint axis (for hinge/slide; ignored for free/ball) |
| `ref` | `double` | Reference configuration value used as `qpos0` |
| `align` | `int` | Align free joint with body CoM (`mjtAlignFree`) |
| `stiffness` | `double` | Passive stiffness coefficient |
| `springref` | `double` | Spring rest position (`qpos_spring`) |
| `springdamper` | `double[2]` | `[timeconst, dampratio]` for springdamper shorthand |
| `limited` | `int` | Whether limits apply (`mjtLimited`: `mjLIMITED_FALSE/TRUE/AUTO`) |
| `range` | `double[2]` | Joint limits `[min, max]` in radians or meters |
| `margin` | `double` | Distance from limit where solver activates |
| `armature` | `double` | Rotor inertia (slider: effective mass) |
| `damping` | `double` | Viscous damping coefficient |
| `frictionloss` | `double` | Dry friction coefficient |
| `actfrclimited` | `int` | Whether actuator force limits apply |
| `actfrcrange` | `double[2]` | Actuator force limits `[min, max]` |
| `actgravcomp` | `mjtByte` | Apply gravity compensation torque via actuators |
| `group` | `int` | Visualization group |

## mjtLimited Enum

```c
typedef enum mjtLimited_ {
  mjLIMITED_FALSE = 0,   // no limits
  mjLIMITED_TRUE,        // limits active
  mjLIMITED_AUTO,        // infer from presence of range attribute
} mjtLimited;
```

## mjtAlignFree Enum

```c
typedef enum mjtAlignFree_ {
  mjALIGNFREE_FALSE = 0,  // don't align
  mjALIGNFREE_TRUE,       // align
  mjALIGNFREE_AUTO,       // respect global compiler flag
} mjtAlignFree;
```

## Examples

```c
// C: hinge joint with limits
mjsJoint* joint = mjs_addJoint(body, NULL);
joint->type = mjJNT_HINGE;
joint->axis[2] = 1.0;          // rotate around z-axis
joint->limited = mjLIMITED_TRUE;
joint->range[0] = -1.5708;     // -90 degrees
joint->range[1] =  1.5708;     // +90 degrees
joint->damping = 0.1;
joint->armature = 0.01;

// Python:
joint = body.add_joint()
joint.type = mujoco.mjtJoint.mjJNT_HINGE
joint.axis = [0, 0, 1]
joint.range = [-1.5708, 1.5708]
joint.limited = True
joint.damping = 0.1

// Free joint shorthand (Python):
body.add_freejoint()
```

## Notes

- `pos` defines the joint anchor in the body's local frame, not the world frame.
- For free joints use `mjs_addFreeJoint(body)` / `body.add_freejoint()` — no need to set axis or limits.
- `damping` value of 0.1 can cause underdamped oscillation in arm joints; 2.0 is safer for Lynx-style arms (see feedback memory).
- See [[mjsBody]] for the parent body struct. See [[mujoco_model_editing_c_api]] for constructor functions.
