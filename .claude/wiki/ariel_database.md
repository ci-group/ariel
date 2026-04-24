# ARIEL SQLite3 Database Reference

Source pages ingested:
- https://ci-group.github.io/ariel/source/Db_examples/db_with_sqlite3.html
- https://ci-group.github.io/ariel/source/Db_examples/db_with_pandas.html
- https://ci-group.github.io/ariel/source/Db_examples/db_with_polars.html

---

## Database File Location

ARIEL writes its experiment database to:

```
__data__/database.db
```

This is a standard SQLite3 file. The `__data__/` directory is relative to the working directory from which the EA is launched.

---

## Schema: `individual` Table

Only one table is documented across all three example pages: **`individual`**.

| Column | Type (Polars schema) | Description |
|---|---|---|
| `id` | `i64` | Unique identifier for this individual |
| `alive` | `i64` | Live/dead flag (0 or 1) |
| `time_of_birth` | `i64` | Generation number when this individual was created |
| `time_of_death` | `i64` | Generation number when this individual was removed |
| `requires_eval` | `i64` | Flag: does this individual need fitness evaluation |
| `fitness_` | `f64` | Fitness value (float) |
| `requires_init` | `i64` | Flag: does this individual need initialization |
| `genotype_` | `str` | JSON5-encoded genotype array (brain weights or body parameters) |
| `tags_` | `str` | JSON object for arbitrary metadata |

**Important:** ARIEL does **not** store an explicit list of individuals per generation. Instead, every individual carries `time_of_birth` and `time_of_death` timestamps, and population membership at any generation is derived computationally. This preserves the complete evolutionary history including individuals eliminated early.

---

## Data Access Patterns

### Pattern 1 — Raw sqlite3 (no pandas/polars dependency)

```python
import sqlite3
import json5
import numpy as np
import matplotlib.pyplot as plt

conn = sqlite3.connect("__data__/database.db")
cursor = conn.execute("SELECT * FROM individual")
rows = cursor.fetchall()
colnames = [desc[0] for desc in cursor.description]
individuals = [dict(zip(colnames, row)) for row in rows]
for ind in individuals:
    ind["genotype_"] = json5.loads(ind["genotype_"])
```

Each element of `individuals` is a plain dict with all columns. `genotype_` is decoded from JSON5 into a Python list (or nested structure).

### Pattern 2 — pandas

```python
import sqlite3
import json5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_sql(
    "SELECT * FROM individual", sqlite3.connect("__data__/database.db"),
)
data["genotype_"] = data["genotype_"].apply(json5.loads)
```

### Pattern 3 — Polars

```python
import sqlite3
import json5
import polars as pl

conn = sqlite3.connect("__data__/database.db")

data = pl.DataFrame(
    conn.execute("SELECT * FROM individual").fetchall(),
    schema=[col[1] for col in conn.execute("PRAGMA table_info(individual)")],
)

data = data.with_columns(
    pl.col("genotype_").map_elements(json5.loads).alias("genotype_")
)
```

Note the Polars pattern uses `PRAGMA table_info(individual)` to derive column names rather than `cursor.description`.

---

## Reconstructing Populations Per Generation

Because ARIEL uses birth/death timestamps, population membership at generation `gen` is:

```
time_of_birth <= gen < time_of_death
```

### sqlite3 / plain Python

```python
time_of_births = [ind["time_of_birth"] for ind in individuals]
time_of_deaths = [ind["time_of_death"] for ind in individuals]
min_gen = int(min(time_of_births))
max_gen = int(max(time_of_deaths))

population_per_gen = {}
for gen in range(min_gen, max_gen + 1):
    ids_alive = [
        ind["id"]
        for ind in individuals
        if ind["time_of_birth"] <= gen < ind["time_of_death"]
    ]
    population_per_gen[gen] = ids_alive
```

### pandas

```python
min_gen = int(data["time_of_birth"].min())
max_gen = int(data["time_of_death"].max())

population_per_gen = {
    gen: data.loc[
        (data["time_of_birth"] <= gen) & (data["time_of_death"] > gen), "id"
    ].tolist()
    for gen in range(min_gen, max_gen + 1)
}
```

### Polars

```python
min_gen = int(data["time_of_birth"].min())
max_gen = int(data["time_of_death"].max())

population_per_gen = {
    gen: data.filter(
        (pl.col("time_of_birth") <= gen) & (pl.col("time_of_death") > gen)
    )["id"].to_list()
    for gen in range(min_gen, max_gen + 1)
}
```

---

## Computing Fitness Statistics Per Generation

### sqlite3 / plain Python

```python
fitness_by_id = {
    ind["id"]: float(ind["fitness_"]) for ind in individuals
}

generations = list(range(min_gen, max_gen + 1))
means = []
stds = []
bests = []

for gen in generations:
    ids = population_per_gen.get(gen, [])
    fits = [fitness_by_id[i] for i in ids if i in fitness_by_id]

    if not fits:
        means.append(np.nan)
        stds.append(np.nan)
        bests.append(np.nan)
        continue

    arr = np.array(fits, dtype=float)
    means.append(float(arr.mean()))
    stds.append(float(arr.std(ddof=0)))
    bests.append(float(arr.min()))
```

**Note:** "best" is `arr.min()` — ARIEL fitness is minimized (lower is better), consistent with Nevergrad convention. See [[Nevergrad_Optimizers]] and [[CMA-ES_Algorithm]].

### pandas

```python
fitness_by_id = data.set_index("id")["fitness_"]

means, stds, maxs = [], [], []
for gen in pop_df["generation"]:
    ids = population_per_gen.get(int(gen), [])
    fits = fitness_by_id.reindex(ids).dropna().astype(float).values
    means.append(float(np.mean(fits)))
    stds.append(float(np.std(fits, ddof=0)))
    maxs.append(float(np.min(fits)))
```

### Polars

```python
fitness_by_id = data.select(["id", "fitness_"])
```

(The polars page shows the setup; statistics loop follows the same pattern as above.)

---

## Accessing Genotypes

Genotypes are stored as JSON5 strings in `genotype_`. After decoding with `json5.loads`, they become Python lists (likely numpy-array-compatible float arrays representing brain weights or body parameters).

```python
# sqlite3 style — after loading with Pattern 1 above:
genotype_of_best = individuals[best_idx]["genotype_"]  # already a list after json5.loads

# pandas style:
best_row = data.loc[data["fitness_"].idxmin()]
genotype = best_row["genotype_"]  # list, after apply(json5.loads)
```

To convert to numpy:

```python
weights = np.array(genotype_of_best, dtype=float)
```

---

## Plotting Fitness Progression

Standard pattern (sqlite3 example, adaptable to pandas/polars):

```python
gens = np.array(generations, dtype=int)
means_arr = np.array(means, dtype=float)
stds_arr = np.array(stds, dtype=float)
bests_arr = np.array(bests, dtype=float)
mask = ~np.isnan(means_arr)

x = gens[mask]
mean = means_arr[mask]
std = stds_arr[mask]
best = bests_arr[mask]

plt.figure(figsize=(10, 5))
plt.plot(x, mean, label="Fitness mean", linewidth=2)
plt.plot(x, best, "--", label="Fitness best", linewidth=2)
plt.fill_between(x, mean - std, mean + std, alpha=0.25, label="Mean ± Std")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness statistics per generation (sqlite3)")
plt.legend()
plt.yticks(range(0, int(max(mean) + 5), 2))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

Pandas variant uses named colors:

```python
plt.plot(x, mean, label="Fitness mean", color="C0", linewidth=2)
plt.plot(x, maxv, label="Fitness best", color="C1", linestyle="--", linewidth=2)
plt.fill_between(x, mean - std, mean + std, color="C0", alpha=0.25)
```

---

## Key Design Points

1. **All individuals are persisted.** The DB stores every individual ever created, not just survivors. Individuals eliminated mid-run are still present with their `time_of_death` set.

2. **No explicit generation table.** Generation range is derived from `min(time_of_birth)` to `max(time_of_death)`.

3. **Fitness convention.** The "best" individual in any generation is the one with the **minimum** `fitness_` value. This matches Nevergrad's minimization convention used throughout ARIEL.

4. **JSON5 for genotypes.** Requires the `json5` library (not the stdlib `json`). The distinction matters if genotype data contains trailing commas or non-standard floats.

5. **`tags_`** is a JSON blob for arbitrary metadata — the documentation does not specify what keys the EA writes there by default.

6. **db_handling modes.** The example pages do not document explicit `db_handling` configuration options. The database path `__data__/database.db` appears to be a convention, not a configurable parameter based on these pages.

---

## Dependencies

| Library | Role |
|---|---|
| `sqlite3` | stdlib; opens the `.db` file |
| `json5` | decodes genotype strings (not stdlib `json`) |
| `numpy` | fitness array statistics |
| `pandas` | optional; DataFrame-based access |
| `polars` | optional; DataFrame-based access |
| `matplotlib` | plotting |

---

## Cross-References

- [[CMA-ES_Algorithm]] — optimizer that generates the genotypes stored in `genotype_`
- [[Nevergrad_Optimizers]] — Nevergrad wraps CMA-ES; fitness minimization convention
- [[Nevergrad_Parametrization]] — how genotype parameter spaces are defined before being serialized to `genotype_`
- [[MjData.md]] — MuJoCo simulation variables whose outputs feed into the `fitness_` column
