---
type: api_reference
tags: [mujoco, python, class, rendering, opengl]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MjrContext

OpenGL rendering context. Wraps the MuJoCo `mjrContext` C struct. Constructor calls `mjr_makeContext` internally.

## Signature

```python
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
```

## Parameters

- `model` — [[MjModel]] instance
- font scale — `mjtFontScale` enum value controlling UI text size

## Font Scale Values (`mjtFontScale`)

| Value | Description |
|-------|-------------|
| `mjFONTSCALE_50` | 50% |
| `mjFONTSCALE_100` | 100% (normal) |
| `mjFONTSCALE_150` | 150% (recommended for HD) |
| `mjFONTSCALE_200` | 200% |
| `mjFONTSCALE_250` | 250% |
| `mjFONTSCALE_300` | 300% |

## Notes

- Do NOT modify `MjrContext` attributes directly; MuJoCo manages it internally.
- Must be created after an OpenGL context is current (see [[mujoco_rendering_pipeline]] for EGL/GLFW setup).

## Usage

```python
# Render scene to active framebuffer
mujoco.mjr_render(viewport, scene, ctx)

# Read pixels
rgb   = np.zeros((H, W, 3), dtype=np.uint8)
depth = np.zeros((H, W),    dtype=np.float32)
mujoco.mjr_readPixels(rgb, depth, viewport, ctx)

# Overlay text
mujoco.mjr_overlay(font, gridpos, viewport, text1, text2, ctx)
```

See [[MjvScene]], [[MjrRect]], [[mujoco_rendering_pipeline]].
