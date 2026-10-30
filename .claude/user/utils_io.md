# Utils & I/O — tracking, video, results DB

Helpers around a run: record robot state, save video/frames, measure
morphology, and read back the results database.

## Tracker — record state during a run

```python
from ariel.utils.tracker import Tracker
import mujoco

tracker = Tracker(
    mujoco.mjtObj.mjOBJ_BODY,        # what kind of object
    "core",                          # name substring to bind
    ["xpos"],                        # attributes to record
)
tracker.setup(world.spec, data)      # call before stepping
# ... run sim ...
traj = tracker.history["xpos"]       # {body_id: [pos, pos, ...]}
```

Signature: `Tracker(mujoco_obj_to_find=None, name_to_bind=None,
observable_attributes=None, quiet=False)` — defaults to all `core` geoms /
`xpos` if omitted. [tracker.py](../../src/ariel/utils/tracker.py). `Tracker`
also plugs into `Controller` (see [controllers.md](controllers.md)).

## Video & frames

```python
from ariel.utils.video_recorder import VideoRecorder      # or: from ariel.utils.renderers import VideoRecorder
from ariel.utils.renderers import video_renderer, single_frame_renderer

rec = VideoRecorder(file_name="run", output_folder=str(out), width=640,
                    height=480, fps=30)
video_renderer(model, data, duration=30, video_recorder=rec)
```

`VideoRecorder` also has the manual API used in custom render loops:
`rec.write(frame=...)` then `rec.release()`
([video_recorder.py](../../src/ariel/utils/video_recorder.py)). Example:
[examples/a_mujoco/2_video_recorder.py](../../examples/a_mujoco/2_video_recorder.py).

## Morphological measures

`from ariel.utils.morphological_descriptor import MorphologicalMeasures` —
descriptors of a body (proportions, branching, etc.)
([morphological_descriptor.py](../../src/ariel/utils/morphological_descriptor.py)).
Used in [examples/c_genotypes/1_body_evolution_tree.py](../../examples/c_genotypes/1_body_evolution_tree.py).

## Other utils

| Import | Use |
|---|---|
| `from ariel.utils.file_ops import generate_save_path` | build timestamped output paths |
| `from ariel.utils.mujoco_ops import euler_to_quat_conversion, has_self_collision` | geometry/collision helpers |
| `from ariel.utils.runners import simple_runner, thread_safe_runner` | drive a sim loop |
| `from ariel.utils.optimizers.revde import RevDE` | reversible differential evolution operator |

## Reading the results DB

An `EA` run writes SQLite (default `__data__/database.db`). Load with sqlite3 /
pandas / polars:

```python
import pandas as pd, sqlite3
df = pd.read_sql("SELECT * FROM individual", sqlite3.connect("__data__/database.db"))
```

Schema, the birth/death temporal model, and pandas/polars access patterns are
documented in [../wiki/ariel_database.md](../wiki/ariel_database.md) and
[../../wiki/ariel_db_pandas.md](../../wiki/ariel_db_pandas.md). Replay a saved
best individual: [examples/c_genotypes/runner_best_individual.py](../../examples/c_genotypes/runner_best_individual.py),
[simulate_from_db.py](../../examples/c_genotypes/simulate_from_db.py).

## Parallel evaluation

Multiprocessing / Ray patterns for evaluating a population in parallel:
[docs/source/parallel_robot_eval/](../../docs/source/parallel_robot_eval/) and
[examples/re_book/1_brain_evolution_multiprocessing.py](../../examples/re_book/1_brain_evolution_multiprocessing.py).
