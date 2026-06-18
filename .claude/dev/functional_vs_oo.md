# Functional-first, classes when justified

ARIEL's design bias: **prefer free functions and `@dataclass` containers.**
Reach for a class only when one of a small set of conditions genuinely applies.
Functions have a ceiling — when behaviour needs persistent state, identity, a
framework contract, or a fluent API, a class is the right tool. The skill is
knowing which side of that line you're on.

## Default: function or dataclass

Use a **plain function** when the work is a transformation: inputs → output, no
state to carry between calls.

```python
# src/ariel/simulation/tasks/gait_learning.py
def xy_displacement(
    xy1: tuple[float, float],
    xy2: tuple[float, float],
) -> float:
    ...
```

Real examples to imitate:
- Fitness tasks — [src/ariel/simulation/tasks/targeted_locomotion.py](../../src/ariel/simulation/tasks/targeted_locomotion.py),
  [gait_learning.py](../../src/ariel/simulation/tasks/gait_learning.py)
  (`distance_to_target`, `x_speed`, …): pure scoring functions.
- Tree operators — [src/ariel/ec/genotypes/tree/operators.py](../../src/ariel/ec/genotypes/tree/operators.py)
  (`add_node`, `subtree_swap`, `mutate_hoist`): take a `TreeGenome` as the first
  arg, mutate in place or return a new genome. **No operator class.**
- MuJoCo helpers — [src/ariel/utils/mujoco_ops.py](../../src/ariel/utils/mujoco_ops.py)
  (`euler_to_quat_conversion`, `has_self_collision`).

Use a **`@dataclass`** when you need a typed bag of fields with defaults but no
real behaviour:

```python
# src/ariel/simulation/controllers/controller.py
@dataclass
class Controller:
    controller_callback_function: Callable[..., Any]
    time_steps_per_ctrl_step: int = 50
    alpha: float = 0.5
    tracker: Tracker = field(default_factory=Tracker)
```

Also: the `World` configs in [src/ariel/simulation/environments/](../../src/ariel/simulation/environments/)
and `TreeGenome` ([tree_genome.py](../../src/ariel/ec/genotypes/tree/tree_genome.py)).

## When a class IS justified

Each row is a real call site. If your case matches one, write the class.

| Trigger | Example | Why a function won't do |
|---|---|---|
| **Persistent / DB-backed state + invariants** | `Individual(SQLModel)` — [src/ariel/ec/individual.py](../../src/ariel/ec/individual.py) | ORM row; `fitness` property guards access before evaluation |
| **Fluent / chainable API** | `Population.where(...).alive` — [src/ariel/ec/population.py](../../src/ariel/ec/population.py) | each call returns a new `Population`; reads like a query |
| **Algorithm bundling state + steps** | `RevDE` — [src/ariel/utils/optimizers/revde.py](../../src/ariel/utils/optimizers/revde.py); `EA` — [src/ariel/ec/ea.py](../../src/ariel/ec/ea.py) | holds matrices / population between `step()` calls |
| **`nn.Module` / framework contract** | `SimpleCPG`, `NaCPG` — [src/ariel/simulation/controllers/](../../src/ariel/simulation/controllers/) | PyTorch requires subclassing `nn.Module` |
| **Plugin point needing subclassing** | `Module(ABC)` — [src/ariel/body_phenotypes/robogen_lite/modules/module.py](../../src/ariel/body_phenotypes/robogen_lite/modules/module.py) | `Core`/`Brick`/`Hinge` override `rotate` |
| **Validated config** | Pydantic `BaseModel` / `EASettings(BaseSettings)` — [ea.py](../../src/ariel/ec/ea.py) | field validation + `.env` overrides |
| **Framework extension** | `BaseWorld` subclasses — [environments/_base_world.py](../../src/ariel/simulation/environments/_base_world.py) | builds on MuJoCo's `MjSpec` builder |

## The heuristic

> Reach for a class only when **state, identity, a framework contract, or a
> fluent API** demands it. Otherwise a function (or a `@dataclass`) is the
> limit-respecting default.

Anti-pattern to avoid: a class with one `__init__` and one method and no state
that outlives the call — that's a function wearing a costume. Conversely, don't
thread a growing tuple of state through ten free functions to avoid a class —
that's the ceiling, and a stateful class is cleaner.
