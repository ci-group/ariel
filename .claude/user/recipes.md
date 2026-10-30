# Recipes — end-to-end snippets

Copy-paste starting points. Each cites the **source example** to open for the
full, runnable version (these are condensed and elide arg-parsing / plotting).
When in doubt, run the cited example with `uv run examples/...`.

---

## 1. Spawn a prebuilt robot and simulate

Source: [examples/a_mujoco/0_mujoco_launcher.py](../../examples/a_mujoco/0_mujoco_launcher.py)

```python
import mujoco
from mujoco import viewer
from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider_with_blocks import (
    body_spider45,
)

mujoco.set_mjcb_control(None)
world = SimpleFlatWorld()
world.spawn(body_spider45().spec, position=[0, 0, 0.1])
model = world.spec.compile()
data = mujoco.MjData(model)
viewer.launch(model=model, data=data)        # interactive
```

## 2. Add a CPG and walk

Source: [examples/re_book/2_body_brain_evolution.py](../../examples/re_book/2_body_brain_evolution.py)

```python
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.simple_cpg import (
    SimpleCPG, create_fully_connected_adjacency,
)
from ariel.utils.tracker import Tracker

cpg = SimpleCPG(create_fully_connected_adjacency(model.nu))
tracker = Tracker(mujoco.mjtObj.mjOBJ_BODY, "core", ["xpos"])
ctrl = Controller(
    controller_callback_function=lambda m, d, *a, **k: cpg.forward(d.time),
    tracker=tracker,
)
ctrl.tracker.setup(world.spec, data)
mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, duration=30))
while data.time < 30:
    mujoco.mj_step(model, data)
```

## 3. Record a video of a run

Source: [examples/a_mujoco/2_video_recorder.py](../../examples/a_mujoco/2_video_recorder.py)

```python
from ariel.utils.renderers import VideoRecorder, video_renderer

rec = VideoRecorder(file_name="run", output_folder="__data__/videos")
video_renderer(model, data, duration=30, video_recorder=rec)
```

## 4. Brain-only evolution with CMA-ES (vision)

Source: [examples/re_book/1_brain_evolution.py](../../examples/re_book/1_brain_evolution.py)

```python
from evotorch.algorithms import CMAES
from evotorch.neuroevolution import NEProblem
from ariel.simulation.tasks.targeted_locomotion import distance_to_target

problem = NEProblem(
    objective_sense="min", network=network.eval(),
    network_eval_func=fitness_function, initial_bounds=(-0.5, 0.5), device="cpu",
)
searcher = CMAES(problem=problem, stdev_init=0.075, popsize=10)
for _ in range(budget + 1):
    searcher.step()
best = searcher.status["best"].values
```

## 5. Joint body + brain evolution with the EA engine

Source: [examples/re_book/2_body_brain_evolution.py](../../examples/re_book/2_body_brain_evolution.py)

```python
from ariel.ec import EA, EAOperation, EASettings, Individual, Population

cfg = EASettings(is_maximisation=False, num_steps=80,
                 target_population_size=80, output_folder=DATA)
population = Population([make_individual() for _ in range(80)])
ops = [
    EAOperation(parent_selection),
    EAOperation(reproduction),
    EAOperation(evaluate),
    EAOperation(survivor_selection),
]
ea = EA(population, operations=ops, num_steps=80,
        db_file_path=cfg.db_file_path, db_handling=cfg.db_handling)
ea.run()
best = ea.get_solution("best", only_alive=False)
```

## 6. Decode a CPPN genome into a body

Source: [examples/c_genotypes/5_body_evolution_cppn.py](../../examples/c_genotypes/5_body_evolution_cppn.py)
(see [bodies.md](bodies.md) for the import block)

```python
graph = MorphologyDecoderBestFirst(cppn_genome=genome, max_modules=10).decode()
spec = construct_mjspec_from_graph(graph).spec
world.spawn(spec, position=(-0.8, 0.0, 0.1))
```

## 7. Replay the best individual from the results DB

Source: [examples/c_genotypes/runner_best_individual.py](../../examples/c_genotypes/runner_best_individual.py),
[simulate_from_db.py](../../examples/c_genotypes/simulate_from_db.py) — load
`__data__/database.db`, rebuild the body from the stored genotype, and simulate.
DB schema: [../wiki/ariel_database.md](../wiki/ariel_database.md).
