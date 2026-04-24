---
title: "Database Manipulation with Pandas"
authors: "ARIEL / ci-group"
year: 2024
source: "https://ci-group.github.io/ariel/source/Db_examples/db_with_pandas.html"
tags: [ariel, database, pandas, sqlite, evolutionary-algorithm, analysis, visualization]
---

# Database Manipulation with Pandas

One-sentence summary: Demonstrates how to read ARIEL's SQLite individual database into pandas, reconstruct per-generation populations from birth/death timestamps, compute fitness statistics, and plot progression curves.

## Database Structure

ARIEL stores all individuals ever created in a single SQL table (`individual`) inside `__data__/database.db` (SQLite). Unlike traditional EA implementations that store generation-based population lists, ARIEL uses a **temporal tracking** model.

### Schema — `individual` table

| Column | Type | Description |
|---|---|---|
| `id` | int | Individual identifier |
| `alive` | bool | Current status |
| `time_of_birth` | int | Generation when created |
| `time_of_death` | int | Generation when removed from population |
| `fitness_` | float | Fitness score |
| `genotype_` | JSON string | Genetic information |
| `tags_` | JSON string | Custom metadata |

## Loading Data

```python
import sqlite3
import json5
import pandas as pd

data = pd.read_sql(
    "SELECT * FROM individual",
    sqlite3.connect("__data__/database.db"),
)
data["genotype_"] = data["genotype_"].apply(json5.loads)
```

Note: `genotype_` is stored as a JSON string and must be decoded with `json5.loads`.

## Reconstructing Population per Generation

Because individuals have birth/death times rather than a generation membership list, population snapshots must be reconstructed:

```python
min_gen = int(data["time_of_birth"].min())
max_gen = int(data["time_of_death"].max())

population_per_gen = {
    gen: data.loc[
        (data["time_of_birth"] <= gen) & (data["time_of_death"] > gen), "id"
    ].tolist()
    for gen in range(min_gen, max_gen + 1)
}

pop_df = pd.DataFrame({
    "generation": list(population_per_gen.keys()),
    "individuals": list(population_per_gen.values()),
    "pop size": [len(v) for v in population_per_gen.values()],
})
```

An individual is **alive at generation `g`** if `time_of_birth <= g < time_of_death`.

## Computing Fitness Statistics per Generation

```python
fitness_by_id = data.set_index("id")["fitness_"]

means, stds, maxs = [], [], []

for gen in pop_df["generation"]:
    ids = population_per_gen.get(int(gen), [])
    fits = fitness_by_id.reindex(ids).dropna().astype(float).values
    if fits.size == 0:
        means.append(np.nan); stds.append(np.nan); maxs.append(np.nan)
    else:
        means.append(float(np.mean(fits)))
        stds.append(float(np.std(fits, ddof=0)))
        maxs.append(float(np.min(fits)))   # min = best for minimisation tasks

pop_df["fitness_mean"] = means
pop_df["fitness_std"]  = stds
pop_df["fitness_best"] = maxs
```

`np.min` is used for best fitness — confirms ARIEL minimises fitness by default (lower = better).

## Visualization

```python
import matplotlib.pyplot as plt

df   = pop_df.copy()
mask = df["fitness_mean"].notna()

x    = df.loc[mask, "generation"]
mean = df.loc[mask, "fitness_mean"]
std  = df.loc[mask, "fitness_std"]
best = df.loc[mask, "fitness_best"]

plt.figure(figsize=(10, 5))
plt.plot(x, mean, label="Fitness mean", color="C0", linewidth=2)
plt.plot(x, best, label="Fitness best", color="C1", linestyle="--", linewidth=2)
plt.fill_between(x, mean - std, mean + std, color="C0", alpha=0.25, label="Mean ± Std")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness statistics per generation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## Key Design Properties

- **Complete history**: The DB retains every individual ever created, not just survivors — enables post-hoc lineage and selection pressure analysis.
- **Generational reconstruction via temporal filter**: `time_of_birth <= g < time_of_death` is the canonical membership predicate.
- **Fitness convention**: ARIEL minimises fitness (best = `np.min`).
- **genotype_ encoding**: JSON stored as string; decode with `json5.loads` (tolerates trailing commas / single quotes).

## Related Topics

- [[ariel_individual_db]] — schema details and individual lifecycle
- [[ariel_ea_loop]] — how generations and selection interact
