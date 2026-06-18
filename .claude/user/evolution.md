# Evolution — the EA engine

ARIEL's EA is a generational loop: a `Population` of `Individual`s is run through
an ordered list of `EAOperation`s each generation, and results persist to a
SQLite database. You can use the built-in `EA` engine, or drive a third-party
optimizer (CMA-ES, Nevergrad) directly for brain-only problems.

## Public API

```python
from ariel.ec import (
    EA, EAOperation, EASettings, Individual, Population, Archive,
    Crossover, FloatMutator, IntegerMutator,
)
```

All re-exported from [ec/__init__.py](../../src/ariel/ec/__init__.py).

| Symbol | Role |
|---|---|
| `Individual` | one solution; `.genotype` (dict), `.fitness`, `.alive`, `.tags`, `.requires_eval` (SQLModel row) |
| `Population` | list-like; `.where(pred)`, `.alive`, `.sort(sort="min", attribute="fitness_")`, `.extend(...)` |
| `EASettings` | config (Pydantic): `is_maximisation`, `num_steps`, `target_population_size`, `output_folder`, `db_file_name`, derived `db_file_path` |
| `EAOperation` | wraps a `Population -> Population` step (parent selection, reproduction, evaluate, survivor selection) |
| `EA` | the engine: `EA(population, operations=[...], num_steps=..., db_file_path=..., db_handling=...)`, `.run()`, `.get_solution("best"\|"median"\|"worst", only_alive=...)` |
| `Archive` | archive of individuals across the run |

## Minimal loop

```python
from ariel.ec import EA, EAOperation, Population, Individual

population = Population([make_individual() for _ in range(pop_size)])
ops = [
    EAOperation(parent_selection),    # each: Population -> Population
    EAOperation(reproduction),
    EAOperation(evaluate),            # sets ind.fitness for ind.requires_eval
    EAOperation(survivor_selection),
]
ea = EA(population, operations=ops, num_steps=budget)
ea.run()
best = ea.get_solution("best", only_alive=False)
```

Each operation is a function whose **first parameter is `Population` and which
returns `Population`** (`EAOperation` validates this). Full, working
body+brain example with all four operations, `EASettings`, and DB output:
[examples/re_book/2_body_brain_evolution.py](../../examples/re_book/2_body_brain_evolution.py).

### `Individual` patterns

```python
ind = Individual()
ind.genotype = {"morph": genome.to_dict(), "ctrl": [...]}
ind.tags["ps"] = False          # free-form tags (e.g. parent-selected)
ind.fitness = score             # also flips requires_eval False
```

`ind.fitness` raises if read before evaluation — that's by design
([individual.py](../../src/ariel/ec/individual.py)).

## Minimisation vs maximisation

Set `is_maximisation` on `EASettings`, and sort with the matching direction
(`population.sort(sort="min", attribute="fitness_")` for minimisation). The
locomotion examples **minimise distance to target**.

## Brain-only: external optimizers

For evolving just controller weights, skip the `EA` engine and use EvoTorch
CMA-ES or Nevergrad directly. See
[examples/re_book/1_brain_evolution.py](../../examples/re_book/1_brain_evolution.py)
(EvoTorch `CMAES` + `NEProblem`) and `3_body_brain_lr.py` (Nevergrad).
Algorithm internals: [../wiki/](../wiki/) (`CMA-ES_*.md`, `Nevergrad_*.md`) via
[`/query`](../commands/query.md).

## Results

Runs write a SQLite DB (default `__data__/database.db`). Reading/analysing it →
[utils_io.md](utils_io.md).
