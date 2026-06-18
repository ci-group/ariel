# Archive & Restarting from a Database

Two closely related features make it easy to reuse data across EA runs: the **`Archive`** class for querying historical individuals from a past experiment's database, and the **`restart`** parameter on `EA` for resuming an evolution from any saved generation.

---

## The Archive Class

`Archive` opens a SQLite database written by any previous `EA` run and exposes query methods for retrieving individuals from it.
It is read-only — it never modifies the source database.

```python
from pathlib import Path
from ariel.ec import Archive

archive = Archive("__data__/run_1/database.db")

print(archive)
# Archive(path=..., size=3200)  ← total evaluated individuals across all generations

print(archive.generation_range)
# (0, 99)  ← earliest birth, latest death recorded

print(archive.fitness_stats())
# {"min": 0.12, "max": 9.87, "mean": 4.3, "std": 1.2, "median": 4.1}
```

### Retrieving individuals

All query methods return `Individual` objects (or lists of them) with their genotype and fitness already populated.

```python
# Single best individual ever recorded
best = archive.best_individual(fitness_mode="min")
print(best.fitness)

# A random individual (uniform sample)
random_ind = archive.random_individual()

# Only individuals that died before generation 30
early = archive.random_individual(death_range=(0, 30))
```

### Retrieving populations

```python
# Hall of fame — the 10 best individuals of all time
hof = archive.hall_of_fame(n=10, fitness_mode="min")

# Random sample of 20 individuals
pool = archive.random_population(n=20)

# Everyone alive at generation 50
gen50 = archive.by_generation(generation=50)

# Individuals in the 40th–60th fitness percentile (mediocre-but-diverse sample)
mid_tier = archive.fitness_percentile_population(lo_pct=40, hi_pct=60, n=15)
```

### Tournament selection from the archive

`tournament_population` is the primary method of injecting historically successful individuals back into a stagnating population.

```python
# Fetch 10 individuals via tournament selection from the full history
resurrected = archive.tournament_population(
    n=10,
    tournament_size=4,
    fitness_mode="min",
    pool_multiplier=3,   # sample 30 candidates, run 10 tournaments
)
```

These can be passed directly into a running EA as injection candidates via a custom `EAOperation`.

---

## Restarting an EA from a Database

The `EA` class accepts a `restart` parameter that loads a previous run's final population (or any past generation) and continues evolution from there. No separate Archive interaction is needed.

### Resume from the latest generation

```python
from ariel.ec import EA

ea = EA(
    population=None,          # ignored when restart is set
    operations=my_operations,
    restart="__data__/run_1/database.db",
    num_steps=100,
)
ea.run()
```

The engine automatically picks the last recorded generation, copies those individuals into a fresh population with `requires_eval=False` (their fitness is already known), and continues from generation `last + 1`.

### Resume from a specific generation

Pass a `(path, generation)` tuple to restart from any historical snapshot:

```python
ea = EA(
    population=None,
    operations=my_operations,
    restart=("__data__/run_1/database.db", 50),  # resume from generation 50
    num_steps=50,
)
ea.run()
```

This is useful for branching experiments — run once, then fork from an intermediate checkpoint with different operators or parameters.

---

## Combining Both: Archive Injection + Restart

A common pattern is to restart a stagnated run and seed it with historically fit individuals pulled from the same (or a different) archive:

```python
from ariel.ec import EA, Archive, EAOperation

archive = Archive("__data__/run_1/database.db")

def inject_hall_of_fame(population):
    """Replace the bottom 10% of the population with all-time best individuals."""
    hof = archive.hall_of_fame(n=len(population) // 10, fitness_mode="min")
    worst = population.sort(key=lambda ind: ind.fitness, reverse=True)
    for slot, hero in zip(worst, hof):
        slot.genotype = hero.genotype
        slot.requires_eval = True

ea = EA(
    population=None,
    operations=[
        EAOperation(inject_hall_of_fame),
        EAOperation(my_evaluator),
        EAOperation(my_survivor_selection),
    ],
    restart="__data__/run_1/database.db",
    num_steps=100,
)
ea.run()
```

---

## Summary

| Feature | What it does | When to use |
|---|---|---|
| `Archive(path)` | Read-only query interface to a past run's SQLite DB | Post-hoc analysis, JESUS-style injection, cross-run comparison |
| `archive.hall_of_fame(n)` | Best `n` individuals across all generations | Seeding new runs with proven solutions |
| `archive.tournament_population(n)` | Tournament selection from full history | JESUS resurrection injection |
| `EA(..., restart=path)` | Resume from latest generation in a DB | Continuing an interrupted run |
| `EA(..., restart=(path, gen))` | Resume from a specific past generation | Branching experiments from a checkpoint |
