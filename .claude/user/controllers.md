# Controllers — driving the joints

A controller turns the sim state into actuator commands (`data.ctrl`). Two
common routes: a **CPG** (central pattern generator, good for locomotion) or a
**neural network** (e.g. with vision). Both are wired in via
`mujoco.set_mjcb_control(...)` and the `Controller` glue object.

## CPG controllers

`SimpleCPG` and `NaCPG` are `nn.Module`s built from an adjacency dict.

```python
from ariel.simulation.controllers.simple_cpg import (
    SimpleCPG, create_fully_connected_adjacency,
)

adj = create_fully_connected_adjacency(model.nu)   # nu = #actuators
cpg = SimpleCPG(adj)            # SimpleCPG(adjacency_dict, mu=1.0, dt=0.01, ...)
action = cpg.forward(data.time)  # -> per-joint targets
```

- `SimpleCPG(adjacency_dict, mu=1.0, dt=0.01, hard_bounds=(-pi/2, pi/2), *,
  angle_tracking=False, seed=None)` —
  [simple_cpg.py](../../src/ariel/simulation/controllers/simple_cpg.py).
- `NaCPG` (neural-adaptive variant), exported as the package default
  (`from ariel.simulation.controllers import NaCPG`) —
  [na_cpg.py](../../src/ariel/simulation/controllers/na_cpg.py). Variants:
  `na_cpg_norm.py`, `na_cpg_beta.py`.
- `create_fully_connected_adjacency(num_nodes) -> dict[int, list[int]]` exists in
  each CPG module; import it from the one you use.

CPG parameters (`phase`, `w`, `amplitudes`, `ha`, `b`) are tensors you can set
from a genome vector — see `map_genotype_to_brain` in
[examples/re_book/2_body_brain_evolution.py](../../examples/re_book/2_body_brain_evolution.py).

## The `Controller` glue + binding

```python
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker

tracker = Tracker(mujoco.mjtObj.mjOBJ_BODY, "core", ["xpos"])
ctrl = Controller(
    controller_callback_function=lambda m, d, *a, **k: cpg.forward(d.time),
    tracker=tracker,
)
ctrl.tracker.setup(world.spec, data)
mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, duration=duration))
```

`Controller` (a `@dataclass`,
[controller.py](../../src/ariel/simulation/controllers/controller.py)) fields:
`controller_callback_function`, `time_steps_per_ctrl_step=50`,
`time_steps_per_save=500`, `alpha=0.5`, `tracker`. It throttles control vs.
save frequency, blends old/new control by `alpha`, and clips to servo bounds.

## Custom / neural controller (the callback pattern)

Any callable `(model, data, *args) -> array of length model.nu` works as the
controller. For a vision NN that reads camera frames + robot state each control
step, see the full `run_vision_simulation` / `Network` in
[examples/re_book/1_brain_evolution.py](../../examples/re_book/1_brain_evolution.py).
Robot proprioceptive state helper:
`from ariel.simulation.controllers.utils.data_get import get_state_from_data`.

## Where this fits

Body → [bodies.md](bodies.md); world/run loop → [simulation.md](simulation.md);
scoring the resulting motion → [tasks_fitness.md](tasks_fitness.md).
