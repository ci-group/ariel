---
type: api_reference
tags: [mujoco, mjspec, model-building, c-api]
source: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html
date_ingested: 2026-04-13
---
# mjsGeom

C struct representing a geom element in an [[MjSpec]] model specification. Corresponds to the MJCF `<geom>` element.

## Signature

```c
typedef struct mjsGeom_ {
  mjsElement* element;             // element type (do not modify)
  mjtGeom type;                    // geom type

  // frame, size
  double pos[3];                   // position
  double quat[4];                  // orientation
  mjsOrientation alt;              // alternative orientation
  double fromto[6];                // alternative for capsule, cylinder, box, ellipsoid
  double size[3];                  // type-specific size

  // contact
  int contype;                     // contact type bitmask
  int conaffinity;                 // contact affinity bitmask
  int condim;                      // contact dimensionality
  int priority;                    // contact priority
  double friction[3];              // friction coefficients: slide, roll, spin
  double solmix;                   // solver mixing for contact pairs
  mjtNum solref[mjNREF];           // solver reference
  mjtNum solimp[mjNIMP];           // solver impedance
  double margin;                   // margin for contact detection
  double gap;                      // include in solver if dist < margin-gap

  // inertia inference
  double mass;                     // used to compute density
  double density;                  // used to compute mass and inertia from volume or surface
  mjtGeomInertia typeinertia;      // surface (mjINERTIA_SHELL) or volume (mjINERTIA_VOLUME)

  // fluid forces
  mjtNum fluid_ellipsoid;          // whether ellipsoid-fluid model is active
  mjtNum fluid_coefs[5];           // ellipsoid-fluid interaction coefs

  // visual
  mjString* material;              // name of material
  float rgba[4];                   // rgba color when no material
  int group;                       // group for visualization

  // mesh / heightfield
  mjString* hfieldname;            // heightfield name (for mjGEOM_HFIELD)
  mjString* meshname;              // mesh name (for mjGEOM_MESH)
  double fitscale;                 // uniform scale for mesh fitting

  // other
  mjDoubleVec* userdata;           // user data
  mjsPlugin plugin;                // SDF plugin
  mjString* info;                  // message appended to compiler errors
} mjsGeom;
```

## Parameters

| Field | Type | Description |
|---|---|---|
| `type` | `mjtGeom` | Geom shape: `mjGEOM_SPHERE`, `mjGEOM_CAPSULE`, `mjGEOM_BOX`, `mjGEOM_CYLINDER`, `mjGEOM_PLANE`, `mjGEOM_MESH`, `mjGEOM_HFIELD`, etc. |
| `pos` | `double[3]` | Position in body frame |
| `quat` | `double[4]` | Orientation in body frame (quaternion, w-first) |
| `alt` | `mjsOrientation` | Alternative orientation |
| `fromto` | `double[6]` | Alternative size spec: `[x1,y1,z1, x2,y2,z2]` for capsule/cylinder/box/ellipsoid |
| `size` | `double[3]` | **Always 3 elements.** Unused dims must be 0. See size convention below. |
| `contype` | `int` | Contact type bitmask (collides when `contype & other.conaffinity != 0`) |
| `conaffinity` | `int` | Contact affinity bitmask |
| `condim` | `int` | Contact dimensionality (1, 3, 4, or 6) |
| `priority` | `int` | Contact priority for solver mixing |
| `friction` | `double[3]` | Friction: `[slide, roll, spin]` |
| `mass` | `double` | If >0, overrides density for inertia inference |
| `density` | `double` | Mass per unit volume (or area for shell inertia) |
| `typeinertia` | `mjtGeomInertia` | `mjINERTIA_VOLUME` or `mjINERTIA_SHELL` |
| `material` | `mjString*` | Material name for rendering |
| `rgba` | `float[4]` | Color (r,g,b,a) when no material |
| `group` | `int` | Group for visualization toggling |
| `meshname` | `mjString*` | Name of mesh asset (for `mjGEOM_MESH`) |
| `hfieldname` | `mjString*` | Name of heightfield asset (for `mjGEOM_HFIELD`) |

## Size Convention (always 3 elements)

| Type | size[0] | size[1] | size[2] |
|---|---|---|---|
| sphere | radius | 0 | 0 |
| capsule | radius | half-length | 0 |
| cylinder | radius | half-length | 0 |
| box | x half-size | y half-size | z half-size |
| ellipsoid | x radius | y radius | z radius |
| plane | x half-size | y half-size | 0 |
| mesh / hfield | fitscale | 0 | 0 (size ignored — mesh defines geometry) |

⚠️ The Python binding enforces `geom.size` shape `[3,1]`. Always pad with `0`, never omit.

## mjsSite

Site struct (visual marker / reference point, no collision):

```c
typedef struct mjsSite_ {
  mjsElement* element;
  double pos[3];                   // position
  double quat[4];                  // orientation
  mjsOrientation alt;
  double fromto[6];
  double size[3];                  // geom size (same convention as mjsGeom)
  mjtGeom type;                    // geom type for visualization
  mjString* material;
  int group;
  float rgba[4];
  mjDoubleVec* userdata;
  mjString* info;
} mjsSite;
```

## Examples

```c
// C:
mjsGeom* geom = mjs_addGeom(body, NULL);
geom->type = mjGEOM_CAPSULE;
geom->size[0] = 0.02;   // radius
geom->size[1] = 0.15;   // half-length
geom->size[2] = 0.0;    // unused
geom->mass = 0.1;

// Python:
geom = body.add_geom()
geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
geom.size = [0.02, 0.15, 0]
geom.mass = 0.1
```

## Notes

- `size` must always have 3 elements — see the size convention table above.
- `fromto` provides an alternative to `pos`/`quat`/`size` for elongated shapes; set `size[0]` to radius only.
- Setting both `mass` and `density` on the same geom is an error.
- See [[mjsBody]] for parent struct. See [[mujoco_model_editing_c_api]] for add/find functions.
