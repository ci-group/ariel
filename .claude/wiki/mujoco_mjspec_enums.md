---
type: api_reference
tags: [mujoco, mjspec, model-building, enumerations]
source: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html
date_ingested: 2026-04-13
---
# mujoco_mjspec_enums

Enumeration types defined in `mujoco/mjspec.h` used when building models programmatically via [[MjSpec]] / [[mujoco_model_editing_c_api]].

## mjtGeomInertia

How inertia is inferred from geom geometry:

```c
typedef enum mjtGeomInertia_ {
  mjINERTIA_VOLUME = 0,   // mass distributed throughout volume
  mjINERTIA_SHELL,        // mass distributed on surface only
} mjtGeomInertia;
```

## mjtMeshInertia

How mesh inertia is computed:

```c
typedef enum mjtMeshInertia_ {
  mjMESH_INERTIA_CONVEX = 0,  // convex hull inertia
  mjMESH_INERTIA_EXACT,       // exact inertia
  mjMESH_INERTIA_LEGACY,      // legacy (older MuJoCo) inertia
  mjMESH_INERTIA_SHELL,       // shell (surface only) inertia
} mjtMeshInertia;
```

## mjtMeshBuiltin

Procedurally generated mesh primitives:

```c
typedef enum mjtMeshBuiltin_ {
  mjMESH_BUILTIN_NONE = 0,
  mjMESH_BUILTIN_SPHERE,
  mjMESH_BUILTIN_HEMISPHERE,
  mjMESH_BUILTIN_CONE,
  mjMESH_BUILTIN_SUPERSPHERE,
  mjMESH_BUILTIN_SUPERTORUS,
  mjMESH_BUILTIN_WEDGE,
  mjMESH_BUILTIN_PLATE,
} mjtMeshBuiltin;
```

## mjtBuiltin

Procedurally generated textures:

```c
typedef enum mjtBuiltin_ {
  mjBUILTIN_NONE = 0,
  mjBUILTIN_GRADIENT,   // gradient: rgb1 → rgb2
  mjBUILTIN_CHECKER,    // checker pattern: rgb1, rgb2
  mjBUILTIN_FLAT,       // 2d: rgb1; cube: rgb1-up, rgb2-side, rgb3-down
} mjtBuiltin;
```

## mjtMark

Mark types for procedural textures:

```c
typedef enum mjtMark_ {
  mjMARK_NONE = 0,
  mjMARK_EDGE,    // edges
  mjMARK_CROSS,   // cross
  mjMARK_RANDOM,  // random dots
} mjtMark;
```

## mjtLimited

Whether an element has limits:

```c
typedef enum mjtLimited_ {
  mjLIMITED_FALSE = 0,  // not limited
  mjLIMITED_TRUE,       // limited
  mjLIMITED_AUTO,       // infer from presence of range attribute
} mjtLimited;
```

Used by `mjsJoint.limited`, `mjsJoint.actfrclimited`, `mjsTendon.limited`, `mjsActuator.ctrllimited`, etc.

## mjtAlignFree

Whether to align a free joint with the body's inertial frame:

```c
typedef enum mjtAlignFree_ {
  mjALIGNFREE_FALSE = 0,
  mjALIGNFREE_TRUE,
  mjALIGNFREE_AUTO,  // respect global compiler.alignfree flag
} mjtAlignFree;
```

## mjtInertiaFromGeom

Whether body inertia is inferred from child geoms:

```c
typedef enum mjtInertiaFromGeom_ {
  mjINERTIAFROMGEOM_FALSE = 0,  // do not use; explicit inertial element required
  mjINERTIAFROMGEOM_TRUE,       // always use; overwrite inertial element
  mjINERTIAFROMGEOM_AUTO,       // use only if inertial element is missing
} mjtInertiaFromGeom;
```

Set via `mjsCompiler.inertiafromgeom` (or `spec.compiler.inertiafromgeom` in Python).

## mjtOrientation

Which alternative orientation format is active in an `mjsOrientation` struct:

```c
typedef enum mjtOrientation_ {
  mjORIENTATION_QUAT = 0,     // quaternion (default: pos/quat fields)
  mjORIENTATION_AXISANGLE,    // axis (xyz) and angle
  mjORIENTATION_XYAXES,       // x and y axis vectors
  mjORIENTATION_ZAXIS,        // z axis (minimal rotation from world-z)
  mjORIENTATION_EULER,        // Euler angles (sequence from compiler.eulerseq)
} mjtOrientation;
```

## Notes

- `mjtLimited_AUTO` is convenient: if you set `range[2]`, limits are automatically enabled.
- `mjtInertiaFromGeom_AUTO` (default in practice) means explicit `<inertial>` in the MJCF takes priority; if absent, geom inertias are summed.
- See [[mjsJoint]], [[mjsGeom]], [[mjsActuator]] for where these enums appear. See [[mujoco_enumerations]] for enums defined in `mjmodel.h`.
