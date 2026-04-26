---
type: api_reference
tags: [mujoco, python, rendering, opengl, visualization]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MuJoCo Rendering Pipeline

MuJoCo supports both a high-level `Renderer` class and a low-level OpenGL pipeline using `mjv`/`mjr` functions.

## High-Level: `mujoco.Renderer`

```python
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data)                    # default free camera
renderer.update_scene(data, camera='cam_name') # or camera id (int)
pixels = renderer.render()                     # returns (H, W, 3) uint8 numpy array
depth  = renderer.render_depth()              # returns (H, W) float32
renderer.close()
```

**As a context manager:**
```python
with mujoco.Renderer(model, height=480, width=640) as renderer:
    renderer.update_scene(data)
    img = renderer.render()
```

## Low-Level OpenGL Pipeline

### Setup Objects

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('model.xml')
data  = mujoco.MjData(model)

cam   = mujoco.MjvCamera()
opt   = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)

mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)

cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.0, 0.0, 0.5]
cam.distance  = 3.0
cam.azimuth   = 90.0
cam.elevation = -30.0
```

### Create OpenGL Context and Rendering Context

See [[mujoco_rendering_pipeline#OpenGL Backends]] below for EGL/GLFW/OSMesa setup.

```python
ctx      = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
viewport = mujoco.MjrRect(0, 0, 1280, 720)
```

### Render Loop

```python
mujoco.mj_step(model, data)

mujoco.mjv_updateScene(
    model, data, opt,
    None,                          # perturb (or mujoco.MjvPerturb())
    cam,
    mujoco.mjtCatBit.mjCAT_ALL,
    scene
)
mujoco.mjr_render(viewport, scene, ctx)

# Read pixels
rgb   = np.zeros((720, 1280, 3), dtype=np.uint8)
depth = np.zeros((720, 1280),    dtype=np.float32)
mujoco.mjr_readPixels(rgb, depth, viewport, ctx)
```

## Key Rendering Functions

| Function | Description |
|----------|-------------|
| `mjv_defaultCamera(cam)` | initialize camera to defaults |
| `mjv_defaultFreeCamera(model, cam)` | initialize free camera sized to model |
| `mjv_defaultOption(opt)` | initialize visualization options |
| `mjv_updateScene(m, d, opt, pert, cam, catmask, scene)` | update scene from model/data |
| `mjv_moveCamera(model, action, reldx, reldy, scene, cam)` | interactive camera move |
| `mjr_render(viewport, scene, ctx)` | render scene to active framebuffer |
| `mjr_readPixels(rgb, depth, viewport, ctx)` | read pixel buffer to numpy arrays |
| `mjr_overlay(font, gridpos, viewport, text1, text2, ctx)` | overlay text |
| `mjr_figure(viewport, figure, ctx)` | render 2D figure |

## `mjtCatBit` — Category Bits for `mjv_updateScene`

| Value | Description |
|-------|-------------|
| `mjCAT_STATIC` | body 0 (world) elements |
| `mjCAT_DYNAMIC` | non-world body elements |
| `mjCAT_DECOR` | decorative elements |
| `mjCAT_ALL` | all categories |

## OpenGL Backends

Select via environment variable `MUJOCO_GL`:

| Backend | Module | Use case |
|---------|--------|----------|
| `glfw` | `mujoco.glfw` | on-screen windowed, not headless |
| `egl` | `mujoco.egl` | headless, GPU-accelerated (recommended) |
| `osmesa` | `mujoco.osmesa` | headless, software (CPU only) |

### EGL (Headless, Recommended)

```python
import os
os.environ['MUJOCO_GL'] = 'egl'

from mujoco.egl import GLContext

gl_ctx = GLContext(max_width=1920, max_height=1080)
gl_ctx.make_current()
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
```

### OSMesa (CPU Headless)

```python
from mujoco.osmesa import GLContext
gl_ctx = GLContext(max_width=1920, max_height=1080)
gl_ctx.make_current()
```

### GLFW (On-screen)

```python
from mujoco.glfw import GLContext
gl_ctx = GLContext(max_width=1920, max_height=1080)
gl_ctx.make_current()
```

All three `GLContext` classes share the same interface: `__init__(max_width, max_height)`, `make_current()`, `free()`.

## Full Offscreen Example (EGL)

```python
import os
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('model.xml')
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

with mujoco.Renderer(model, height=480, width=640) as renderer:
    renderer.update_scene(data)
    pixels = renderer.render()   # (480, 640, 3) uint8
```

See [[MjvCamera]], [[MjvOption]], [[MjvScene]], [[MjrContext]], [[MjrRect]], [[mujoco_viewer]].
