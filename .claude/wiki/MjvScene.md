---
type: api_reference
tags: [mujoco, python, class, rendering]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MjvScene

Holds abstract geometric objects for rendering. Constructor calls `mjv_makeScene` internally.

## Signature

```python
scene = mujoco.MjvScene(model=model, maxgeom=10000)
```

## Parameters

- `model` — `MjModel` instance
- `maxgeom` — maximum number of geoms the scene can hold (default 10000)

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ngeom` | int | current number of geoms in the scene |
| `maxgeom` | int | maximum geom capacity |
| `geoms` | array | `mjvGeom` array |
| `flags` | `(mjNRNDFLAG,)` | rendering flags indexed by `mjtRndFlag` |
| `lights` | array | scene lights |
| `nlights` | int | number of lights |

## Rendering Flags (`mjtRndFlag`)

Toggle via `scene.flags[mujoco.mjtRndFlag.mjRND_XXX]`:

| Flag | Description |
|------|-------------|
| `mjRND_SHADOW` | shadows |
| `mjRND_WIREFRAME` | wireframe mode |
| `mjRND_REFLECTION` | reflections |
| `mjRND_ADDITIVE` | additive blending |
| `mjRND_SKYBOX` | skybox |
| `mjRND_FOG` | atmospheric fog |
| `mjRND_HAZE` | haze effect |
| `mjRND_SEGMENT` | segmentation rendering |
| `mjRND_IDCOLOR` | object ID color rendering |

## Usage

Updated each frame by `mjv_updateScene`:

```python
mujoco.mjv_updateScene(
    model, data, opt,
    None,                           # perturb object (or MjvPerturb)
    cam,
    mujoco.mjtCatBit.mjCAT_ALL,
    scene
)
```

Then rendered by [[MjrContext]]:

```python
mujoco.mjr_render(viewport, scene, ctx)
```

See [[mujoco_rendering_pipeline]].
