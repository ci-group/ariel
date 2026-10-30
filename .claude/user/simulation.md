# Simulation — worlds, spawn, run, render

ARIEL wraps MuJoCo. The shape of every sim: **pick a world → spawn a body →
compile → `MjData` → step**, optionally rendering.

## Worlds

Import from the package (re-exported in
[environments/__init__.py](../../src/ariel/simulation/environments/__init__.py)):

```python
from ariel.simulation.environments import (
    SimpleFlatWorld, RuggedTerrainWorld, RuggedTiltedWorld, SimpleTiltedWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, CompoundWorld, OlympicArena,
    BaseWorld,
)
```

`SimpleFlatWorldWithTarget` is **not** in the package `__init__` — import it
directly:

```python
from ariel.simulation.environments._simple_flat_with_target import (
    SimpleFlatWorldWithTarget,
)
```

All worlds subclass `BaseWorld` and expose `.spec` (a `mujoco.MjSpec`) and
`.spawn(spec, position=...)`.

## The core loop

```python
import mujoco
from ariel.simulation.environments import SimpleFlatWorld

mujoco.set_mjcb_control(None)        # clear any previous control callback
world = SimpleFlatWorld()
world.spawn(core.spec, position=[0, 0, 0.1])   # core from bodies.md
model = world.spec.compile()
data = mujoco.MjData(model)

while data.time < duration:
    mujoco.mj_step(model, data)
```

`model.nu` = number of actuators (joints) — you'll need it to size controllers.
Add cameras / target bodies via `world.spec.worldbody.add_camera(...)` /
`.add_body(...)` before `compile()` (see
[examples/re_book/1_brain_evolution.py](../../examples/re_book/1_brain_evolution.py)).

## Rendering & recording

From [ariel.utils.renderers](../../src/ariel/utils/renderers.py):

| Function | Use |
|---|---|
| `video_renderer(model, data, duration=10.0, video_recorder=None, ...)` | render a run to video |
| `single_frame_renderer(model, data, ...) -> PIL.Image` | one frame / screenshot |
| `tracking_video_renderer(...)` | camera follows the robot |

`VideoRecorder` is re-exported from `renderers` for convenience:

```python
from ariel.utils.renderers import VideoRecorder, video_renderer

rec = VideoRecorder(file_name="run", output_folder=str(out_dir))
video_renderer(model, data, duration=30, video_recorder=rec)
```

Interactive viewer: `from mujoco import viewer; viewer.launch(model=model,
data=data)`. Basic MuJoCo setup demo:
[examples/a_mujoco/0_mujoco_launcher.py](../../examples/a_mujoco/0_mujoco_launcher.py).

## Driving the joints

Step-only loops above just simulate physics. To produce motion, attach a
controller and bind it with `mujoco.set_mjcb_control(...)` — see
[controllers.md](controllers.md).

## Deep MuJoCo facts

For exact MuJoCo API details (`MjModel`, `MjData` fields, `MjSpec` editing),
use [../../.claude/wiki/](../wiki/) via [`/query`](../commands/query.md) rather
than guessing array shapes.
