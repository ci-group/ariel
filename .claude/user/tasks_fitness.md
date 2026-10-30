# Tasks & fitness

Fitness functions are **plain functions** that score behaviour. You compute the
score after a simulation run and assign it to `ind.fitness` (or return it to your
optimizer). All live in [simulation/tasks/](../../src/ariel/simulation/tasks/).

## Targeted locomotion

`from ariel.simulation.tasks.targeted_locomotion import (...)`
([targeted_locomotion.py](../../src/ariel/simulation/tasks/targeted_locomotion.py)):

| Function | Signature (args are `np.ndarray` positions unless noted) |
|---|---|
| `distance_to_target` | `(final_position, target_position) -> float` |
| `fitness_delta_distance` | `(initial_pos, final_pos, target_pos) -> float` |
| `fitness_distance_and_efficiency` | `(initial_pos, final_pos, target_pos, total_control_effort) -> float` |
| `fitness_survival_and_locomotion` | `(initial_pos, final_pos, target_pos, min_z_height) -> float` |
| `fitness_direct_path` | `(initial_pos, final_pos, target_pos, total_path_length) -> float` |
| `fitness_speed_to_target` | `(time_to_target, duration, min_distance_to_target) -> float` |

```python
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
score = distance_to_target(final_pos, target_pos)   # minimise this
```

These are **minimised** in the examples (smaller distance = better) — set
`EASettings(is_maximisation=False)` accordingly. Full routing of all six by a
CLI flag: [examples/re_book/1_brain_evolution.py](../../examples/re_book/1_brain_evolution.py).

## Gait learning (open-field locomotion)

`from ariel.simulation.tasks.gait_learning import (...)`
([gait_learning.py](../../src/ariel/simulation/tasks/gait_learning.py)):
`xy_displacement(xy1, xy2)`, `x_speed(...)`, `y_speed(...)` — distance/speed over
a trajectory.

## Turning in place

`from ariel.simulation.tasks.turning_in_place import turning_in_place`
— `turning_in_place(xy_history: list[tuple[float, float]]) -> float`
([turning_in_place.py](../../src/ariel/simulation/tasks/turning_in_place.py)).

## Wiring fitness into an EA

Inside your `evaluate` operation, run the sim, pull the robot's positions (from a
[Tracker](../../src/ariel/utils/tracker.py), see [utils_io.md](utils_io.md)),
score, and assign:

```python
def evaluate(population: Population) -> Population:
    for ind in population:
        if ind.requires_eval:
            final_pos = run_and_get_position(ind)        # your sim
            ind.fitness = distance_to_target(final_pos, target)
    return population
```

Tracked positions come from `tracker.history["xpos"]` after the run. See the
`evaluate` / `run_simulation` pair in
[examples/re_book/2_body_brain_evolution.py](../../examples/re_book/2_body_brain_evolution.py).
