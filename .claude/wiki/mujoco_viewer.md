---
type: api_reference
tags: [mujoco, python, viewer, visualization, interactive]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MuJoCo Interactive Viewer

The `mujoco.viewer` submodule provides an interactive GUI viewer for simulation. Supports both blocking and non-blocking (passive) modes.

## Import

```python
import mujoco
import mujoco.viewer
```

## Blocking Launch

```python
mujoco.viewer.launch(model, data)
```

Opens the viewer and blocks until the window is closed. Useful for static inspection.

## Non-Blocking (Passive) Launch

```python
with mujoco.viewer.launch_passive(model, data) as handle:
    while handle.is_running():
        mujoco.mj_step(model, data)
        handle.sync()          # push model/data changes to GUI, pull user inputs
```

Returns a handle that lets you run your own physics loop while the viewer updates.

## Handle API

| Property/Method | Description |
|-----------------|-------------|
| `handle.is_running()` | returns `True` while viewer window is open |
| `handle.sync()` | synchronize data between Python and viewer GUI |
| `handle.cam` | `MjvCamera` — current camera state |
| `handle.opt` | `MjvOption` — visualization options |
| `handle.pert` | `MjvPerturb` — perturbation state |
| `handle.close()` | close the viewer (also called on context manager exit) |

## Real-Time Loop Example

```python
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('model.xml')
data  = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as handle:
    while handle.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        handle.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))
```

## macOS Note

On macOS, `launch_passive` requires using the `mjpython` launcher instead of `python`:

```bash
mjpython my_script.py
```

See [[mujoco_rendering_pipeline]] for headless/offscreen rendering.
