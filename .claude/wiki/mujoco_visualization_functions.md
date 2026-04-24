---
type: api_reference
tags: [mujoco, python, visualization, rendering, opengl]
source: https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mujoco.h
date_ingested: 2026-04-13
---
# MuJoCo Visualization Functions

Complete reference for all `mjv_*` (abstract visualizer) and `mjr_*` (OpenGL renderer) functions. The visualizer generates abstract scene geometry from model/data; the renderer converts it to pixels via OpenGL.

---

## `mjv_*` — Abstract Visualizer

### Initialization

| Function | Signature | Description |
|----------|-----------|-------------|
| `mjv_defaultCamera` | `(cam: MjvCamera) → None` | Initialize camera to defaults |
| `mjv_defaultFreeCamera` | `(m: MjModel, cam: MjvCamera) → None` | Initialize free camera sized to model extent |
| `mjv_defaultPerturb` | `(pert: MjvPerturb) → None` | Initialize perturbation to defaults |
| `mjv_defaultOption` | `(opt: MjvOption) → None` | Initialize visualization options to defaults |
| `mjv_defaultFigure` | `(fig: MjvFigure) → None` | Initialize 2D figure to defaults |
| `mjv_defaultScene` | `(scn: MjvScene) → None` | Initialize scene (zero all fields) |
| `mjv_makeScene` | `(m: MjModel, scn: MjvScene, maxgeom: int) → None` | Allocate scene geometry buffer |
| `mjv_freeScene` | `(scn: MjvScene) → None` | Free scene memory |

### Scene Update

| Function | Signature | Description |
|----------|-----------|-------------|
| `mjv_updateScene` | `(m, d, opt, pert, cam, catmask, scn) → None` | Full scene update: clears geom list then adds geoms for all selected categories |
| `mjv_addGeoms` | `(m, d, opt, pert, catmask, scn) → None` | Add geoms without updating camera/lights |
| `mjv_makeLights` | `(m, d, scn) → None` | Build light list from model lights |
| `mjv_updateCamera` | `(m, d, cam, scn) → None` | Recompute camera frustum from cam params |
| `mjv_updateSkin` | `(m, d, scn) → None` | Update skin vertex positions |

```python
# Standard render-loop call
mujoco.mjv_updateScene(
    model, data,
    opt,                              # MjvOption
    pert,                             # MjvPerturb or None
    cam,                              # MjvCamera
    mujoco.mjtCatBit.mjCAT_ALL,       # catmask
    scene                             # MjvScene
)
```

### Custom Geometry

| Function | Signature | Description |
|----------|-----------|-------------|
| `mjv_initGeom` | `(geom, type, size, pos, mat, rgba) → None` | Initialize a geom; pass `None` for any field to keep default |
| `mjv_connector` | `(geom, type, width, from, to) → None` | Create cylinder/line connector between two 3D points |

### Camera Interaction

| Function | Signature | Description |
|----------|-----------|-------------|
| `mjv_moveCamera` | `(m, action, reldx, reldy, scn, cam) → None` | Mouse hook for camera orbit/pan/zoom |
| `mjv_moveModel` | `(m, action, reldx, reldy, roomup, scn) → None` | Mouse hook for model transform in room space |
| `mjv_cameraInModel` | `(headpos, forward, up, scn) → None` | Get camera position/orientation in model frame |
| `mjv_cameraInRoom` | `(headpos, forward, up, scn) → None` | Get camera position/orientation in room frame |
| `mjv_frustumHeight` | `(scn) → float` | Return frustum height at model center |
| `mjv_alignToCamera` | `(res, vec, forward) → None` | Project 3D vec onto camera plane |
| `mjv_averageCamera` | `(cam1, cam2) → MjvGLCamera` | Average two GL cameras (for stereo) |
| `mjv_cameraFrame` | `(headpos, forward, up, right, d, cam) → None` | Get camera frame vectors |
| `mjv_cameraFrustum` | `(zver, zhor, zclip, m, cam) → None` | Get frustum extents |

### Coordinate Conversion

| Function | Signature | Description |
|----------|-----------|-------------|
| `mjv_room2model` | `(modelpos, modelquat, roompos, roomquat, scn) → None` | Convert room → model coordinates |
| `mjv_model2room` | `(roompos, roomquat, modelpos, modelquat, scn) → None` | Convert model → room coordinates |

### Perturbation

| Function | Signature | Description |
|----------|-----------|-------------|
| `mjv_initPerturb` | `(m, d, scn, pert) → None` | Set pert refpos/refquat to current body pose |
| `mjv_movePerturb` | `(m, d, action, reldx, reldy, scn, pert) → None` | Mouse hook for moving perturbation reference |
| `mjv_applyPerturbPose` | `(m, d, pert, flg_paused: int) → None` | Write pose into `d.mocap` or `d.qpos` |
| `mjv_applyPerturbForce` | `(m, d, pert) → None` | Write force/torque into `d.xfrc_applied` |

### Selection

```c
int mjv_select(const mjModel* m, const mjData* d, const mjvOption* vopt,
               mjtNum aspectratio, mjtNum relx, mjtNum rely,
               const mjvScene* scn, mjtNum selpnt[3],
               int geomid[1], int flexid[1], int skinid[1])
```

Ray-cast from cursor position `(relx, rely)` (normalized: 0=left/bottom, 1=right/top) into the scene. Returns the selected **body id** (> 0), or -1 if nothing hit. Writes world-space selection point to `selpnt`, and ids to `geomid`, `flexid`, `skinid`.

---

## `mjr_*` — OpenGL Renderer

### Context Management

| Function | Signature | Description |
|----------|-----------|-------------|
| `mjr_defaultContext` | `(con) → None` | Zero-initialize context |
| `mjr_makeContext` | `(m, con, fontscale) → None` | Create OpenGL context for model (called by `MjrContext.__init__`) |
| `mjr_changeFont` | `(fontscale, con) → None` | Change font scale on existing context |
| `mjr_freeContext` | `(con) → None` | Free OpenGL resources |
| `mjr_resizeOffscreen` | `(width, height, con) → None` | Resize offscreen buffer |

### Resource Upload

| Function | Signature | Description |
|----------|-----------|-------------|
| `mjr_uploadTexture` | `(m, con, texid) → None` | Upload texture to GPU |
| `mjr_uploadMesh` | `(m, con, meshid) → None` | Upload mesh to GPU |
| `mjr_uploadHField` | `(m, con, hfieldid) → None` | Upload height-field to GPU |

### Framebuffer Management

| Function | Signature | Description |
|----------|-----------|-------------|
| `mjr_setBuffer` | `(framebuffer, con) → None` | Switch between on-screen and offscreen framebuffers |
| `mjr_restoreBuffer` | `(con) → None` | Restore default framebuffer |
| `mjr_readPixels` | `(rgb, depth, viewport, con) → None` | Read pixel buffer into numpy arrays |
| `mjr_drawPixels` | `(rgb, depth, viewport, con) → None` | Draw pixels from numpy arrays into framebuffer |
| `mjr_blitBuffer` | `(src, dst, flg_color, flg_depth, con) → None` | Blit (copy) region between framebuffers |
| `mjr_setAux` | `(index, con) → None` | Set auxiliary buffer as current |
| `mjr_blitAux` | `(index, src, left, bottom, con) → None` | Blit from auxiliary to default buffer |
| `mjr_addAux` | `(index, width, height, samples, con) → None` | Create auxiliary framebuffer |
| `mjr_maxViewport` | `(con) → MjrRect` | Return max viewport dimensions |

### Rendering

| Function | Signature | Description |
|----------|-----------|-------------|
| `mjr_render` | `(viewport, scn, con) → None` | Render 3D scene to active framebuffer |
| `mjr_finish` | `() → None` | Call `glFinish()` — wait for GPU to complete |
| `mjr_getError` | `() → int` | Return OpenGL error code (0 = no error) |
| `mjr_figure` | `(viewport, fig, con) → None` | Render 2D figure overlay |
| `mjr_rectangle` | `(viewport, r, g, b, a) → None` | Draw filled rectangle |
| `mjr_label` | `(viewport, font, txt, r, g, b, a, rt, gt, bt, con) → None` | Draw text label with background |
| `mjr_overlay` | `(font, gridpos, viewport, overlay, overlay2, con) → None` | Overlay two text strings at grid position |
| `mjr_text` | `(font, txt, con, x, y, r, g, b) → None` | Draw text at pixel (x, y) |
| `mjr_findRect` | `(x, y, nrect, rects) → int` | Find which rectangle contains pixel (x, y) |

```python
# Standard per-frame sequence
mujoco.mjr_render(viewport, scene, ctx)            # 3D scene
mujoco.mjr_overlay(                                # text overlay
    mujoco.mjtFont.mjFONT_NORMAL,
    mujoco.mjtGridPos.mjGRID_TOPLEFT,
    viewport,
    f"Step {step}",
    "",
    ctx
)
mujoco.mjr_figure(fig_viewport, fig, ctx)          # 2D plot overlay
rgb = np.zeros((H, W, 3), dtype=np.uint8)
mujoco.mjr_readPixels(rgb, None, viewport, ctx)    # read to numpy
```

---

## `mjtCatBit` — Category Mask for `mjv_updateScene`

| Name | Value | Description |
|------|-------|-------------|
| `mjCAT_STATIC` | 1 | Body 0 (world) elements |
| `mjCAT_DYNAMIC` | 2 | All other body elements |
| `mjCAT_DECOR` | 4 | Decorative / custom geoms |
| `mjCAT_ALL` | 7 | All categories |

## `mjtPertBit` — Perturbation Flags

| Name | Value | Description |
|------|-------|-------------|
| `mjPERT_TRANSLATE` | 1 | Translation perturbation |
| `mjPERT_ROTATE` | 2 | Rotation perturbation |

See [[MjvPerturb]], [[MjvGeom]], [[MjvFigure]], [[MjvScene]], [[MjvCamera]], [[MjvOption]], [[MjrContext]], [[MjrRect]], [[mujoco_rendering_pipeline]].
