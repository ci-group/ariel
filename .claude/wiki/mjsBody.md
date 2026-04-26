---
type: api_reference
tags: [mujoco, mjspec, model-building, c-api]
source: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html
date_ingested: 2026-04-13
---
# mjsBody

C struct representing a body element in an [[MjSpec]] model specification. Corresponds to the MJCF `<body>` element.

## Signature

```c
typedef struct mjsBody_ {
  mjsElement* element;             // element type (do not modify)
  mjString* childclass;            // childclass name

  // body frame
  double pos[3];                   // frame position
  double quat[4];                  // frame orientation
  mjsOrientation alt;              // frame alternative orientation

  // inertial frame
  double mass;                     // mass
  double ipos[3];                  // inertial frame position
  double iquat[4];                 // inertial frame orientation
  double inertia[3];               // diagonal inertia (in i-frame)
  mjsOrientation ialt;             // inertial frame alternative orientation
  double fullinertia[6];           // non-axis-aligned inertia matrix

  // other
  mjtByte mocap;                   // is this a mocap body
  double gravcomp;                 // gravity compensation
  mjDoubleVec* userdata;           // user data
  mjtByte explicitinertial;        // whether to save with explicit inertial clause
  mjsPlugin plugin;                // passive force plugin
  mjString* info;                  // message appended to compiler errors
} mjsBody;
```

## Parameters

| Field | Type | Description |
|---|---|---|
| `element` | `mjsElement*` | Element type header — do not modify |
| `childclass` | `mjString*` | Default class name applied to child elements |
| `pos` | `double[3]` | Body frame position in parent frame |
| `quat` | `double[4]` | Body frame orientation (quaternion, w-first) |
| `alt` | `mjsOrientation` | Alternative orientation (Euler, axis-angle, etc.) |
| `mass` | `double` | Body mass (used if inertia not specified) |
| `ipos` | `double[3]` | Inertial frame position relative to body frame |
| `iquat` | `double[4]` | Inertial frame orientation |
| `inertia` | `double[3]` | Diagonal inertia components in the inertial frame |
| `ialt` | `mjsOrientation` | Alternative orientation for inertial frame |
| `fullinertia` | `double[6]` | Full inertia matrix (non-axis-aligned): `[I11, I22, I33, I12, I13, I23]` |
| `mocap` | `mjtByte` | Is this a mocap body (kinematically driven) |
| `gravcomp` | `double` | Gravity compensation factor (0=none, 1=full) |
| `userdata` | `mjDoubleVec*` | User-defined numeric data |
| `explicitinertial` | `mjtByte` | Force explicit `<inertial>` in saved XML |
| `plugin` | `mjsPlugin` | Passive force plugin attached to this body |
| `info` | `mjString*` | Error message appended on compilation failure |

## mjsFrame

Frame struct (attach point for subtrees, no inertia):

```c
typedef struct mjsFrame_ {
  mjsElement* element;
  mjString* childclass;
  double pos[3];                   // position
  double quat[4];                  // orientation
  mjsOrientation alt;              // alternative orientation
  mjString* info;
} mjsFrame;
```

## mjsOrientation

Alternative orientation specifier (only one type active at a time):

```c
typedef struct mjsOrientation_ {
  mjtOrientation type;             // active type (see mjtOrientation enum)
  double axisangle[4];             // axis (xyz) and angle
  double xyaxes[6];                // x-axis (xyz) and y-axis (xyz)
  double zaxis[3];                 // z axis (minimal rotation)
  double euler[3];                 // Euler angles (in sequence from mjsCompiler.eulerseq)
} mjsOrientation;
```

## Examples

```c
// Python (via MjSpec bindings):
body = spec.worldbody.add_body()
body.name = "torso"
body.pos = [0, 0, 1.0]
body.mass = 5.0

// C:
mjsBody* body = mjs_addBody(mjs_findBody(spec, "world"), NULL);
mjs_setName(body->element, "torso");
body->pos[2] = 1.0;
body->mass = 5.0;
```

## Notes

- `pos` and `quat` define the body frame relative to the parent body's frame.
- `ipos`/`iquat`/`inertia` define the inertial frame relative to the body frame.
- If `inertia` is all zero and `mass` is set, MuJoCo infers inertia from child geoms (if `compiler.inertiafromgeom` permits).
- See [[mjsGeom]], [[mjsJoint]] for child element types. See [[mujoco_model_editing_c_api]] for constructor functions.
