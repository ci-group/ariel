---
title: "ARIEL Individual Database — Schema and Temporal Model"
tags: [ariel, database, sqlite, individual, temporal-model]
---

# ARIEL Individual Database

ARIEL uses a **flat append-only SQLite database** rather than per-generation population lists. Every individual that ever existed is stored in `__data__/database.db`, table `individual`.

## Temporal Membership Model

Each individual carries two timestamps:

| Field | Meaning |
|---|---|
| `time_of_birth` | Generation at which the individual was added to the population |
| `time_of_death` | Generation at which it was removed (exclusive upper bound) |

**Alive at generation g**: `time_of_birth <= g < time_of_death`

This lets you reconstruct any historical population snapshot with a simple pandas boolean mask — no separate population-list storage needed.

## Fitness Convention

ARIEL **minimises** fitness. When computing "best fitness" per generation use `np.min`, not `np.max`.

## Genotype Encoding

`genotype_` is stored as a JSON string (may use json5 syntax — trailing commas, single quotes). Always decode with `json5.loads`, not the stdlib `json` module.

```python
data["genotype_"] = data["genotype_"].apply(json5.loads)
```

## Usage Pattern

See [[ariel_db_pandas]] for the full pandas workflow to load, reconstruct generations, and plot fitness curves.
