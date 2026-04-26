---
type: api_reference
tags: [mujoco, python, class, rendering]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MjrRect

Viewport rectangle used with rendering functions.

## Signature

```python
viewport = mujoco.MjrRect(left=0, bottom=0, width=1280, height=720)
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `left` | int | left edge in pixels |
| `bottom` | int | bottom edge in pixels (OpenGL origin is bottom-left) |
| `width` | int | viewport width in pixels |
| `height` | int | viewport height in pixels |

## Usage

```python
viewport = mujoco.MjrRect(0, 0, 1280, 720)
mujoco.mjr_render(viewport, scene, ctx)
mujoco.mjr_readPixels(rgb, depth, viewport, ctx)
```

See [[MjrContext]], [[mujoco_rendering_pipeline]].
