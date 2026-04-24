---
type: api_reference
tags: [mujoco, python, class, visualization, rendering]
source: https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjvisualize.h
date_ingested: 2026-04-13
---
# MjvGeom

Abstract geometric element used to inject custom geometry into the MuJoCo scene before rendering. Populated by `mjv_updateScene` for simulation objects; also created manually to add debug overlays, arrows, connectors, or annotations.

## Signature

```python
geom = mujoco.MjvGeom()
# Initialize with mjv_initGeom:
mujoco.mjv_initGeom(geom, type, size, pos, mat, rgba)
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `type` | int | Geometry type (`mjtGeom` enum: `mjGEOM_SPHERE`, `mjGEOM_CAPSULE`, `mjGEOM_BOX`, `mjGEOM_ARROW`, `mjGEOM_LINE`, etc.) |
| `dataid` | int | Mesh / hfield / plane id; -1 if not applicable |
| `objtype` | int | MuJoCo object type (`mjtObj`) of the parent object |
| `objid` | int | Id of the parent object |
| `category` | int | Visual category (`mjtCatBit`): `mjCAT_STATIC`, `mjCAT_DYNAMIC`, `mjCAT_DECOR` |
| `matid` | int | Material id; -1 if none |
| `texcoord` | int | Flag: texture coordinates available |
| `segid` | int | Segmentation id for `mjRND_SEGMENT` / `mjRND_IDCOLOR` rendering |
| `size` | `(3,)` float | Size parameters (geometry-type-dependent) |
| `pos` | `(3,)` float | Position in model space |
| `mat` | `(9,)` float | Orientation as 3×3 rotation matrix (row-major) |
| `rgba` | `(4,)` float | Color: [R, G, B, A] in [0, 1] |
| `emission` | float | Emission coefficient (0 = no emission) |
| `specular` | float | Specular coefficient |
| `shininess` | float | Shininess coefficient |
| `reflectance` | float | Reflectance coefficient |
| `label` | str (char[100]) | Text label drawn near the geom |
| `camdist` | float | Distance to camera (set by renderer) |
| `modelrbound` | float | Model bounding radius |
| `transparent` | bool | Transparency flag |

## Key Functions

### `mjv_initGeom`

```c
void mjv_initGeom(mjvGeom* geom, int type, const mjtNum size[3],
                  const mjtNum pos[3], const mjtNum mat[9], const float rgba[4])
```

Initialize a geom: set `type`, `size`, `pos`, `mat`, `rgba` — leave others at defaults. Pass `None` for any field to keep it at the default value.

### `mjv_connector`

```c
void mjv_connector(mjvGeom* geom, int type, mjtNum width,
                   const mjtNum from[3], const mjtNum to[3])
```

Initialize a connector geom (cylinder or line) between two 3D points. `type` should be `mjGEOM_CAPSULE`, `mjGEOM_CYLINDER`, or `mjGEOM_LINE`.

## Usage — Adding Custom Arrows/Debug Geoms

```python
import mujoco
import numpy as np

# scene.ngeom tracks the count; manually increment to inject custom geoms
if scene.ngeom < scene.maxgeom:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([0.01, 0.01, 0.1]),   # size: radius, radius, length
        np.array([0.0, 0.0, 1.0]),     # pos
        np.eye(3).flatten(),           # mat (identity)
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # rgba: red
    )
    geom.label = "force"
    scene.ngeom += 1
```

## Usage — Line Between Two Points

```python
from_pt = np.array([0.0, 0.0, 0.0])
to_pt   = np.array([1.0, 0.0, 0.0])

if scene.ngeom < scene.maxgeom:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        0.005,     # width
        from_pt,
        to_pt
    )
    geom.rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    scene.ngeom += 1
```

## Notes

- Custom geoms must be added **after** `mjv_updateScene` (which resets `scene.ngeom`) and **before** `mjr_render`.
- `scene.maxgeom` is set at `MjvScene` construction; choose a value large enough for simulation geoms plus custom ones.
- `category = mjCAT_DECOR` is appropriate for debug/custom overlays.

See [[MjvScene]], [[mujoco_visualization_functions]], [[mujoco_rendering_pipeline]].
