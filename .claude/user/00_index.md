# user/ — building with ARIEL

Read this when you (or an AI you direct) are **writing a script that uses ARIEL**
— not editing ARIEL itself (that's [../dev/](../dev/00_index.md)).

Each file below is a **map**: public symbol → import path → minimal signature →
smallest working snippet → "see example X". Load **only the file matching the
question** so the answer stays cheap and traceable. These maps point at the real
`examples/` and `src/` instead of duplicating them.

## Routing table

| You want to… | Read |
|---|---|
| Build / decode a robot body, use a prebuilt robot | [bodies.md](bodies.md) |
| Set up a world, spawn, run the sim, render/record | [simulation.md](simulation.md) |
| Drive joints with a CPG or a neural network | [controllers.md](controllers.md) |
| Run an evolutionary algorithm (population loop) | [evolution.md](evolution.md) |
| Score behaviour (fitness functions) | [tasks_fitness.md](tasks_fitness.md) |
| Track data, save video, read the results DB | [utils_io.md](utils_io.md) |
| Copy a full end-to-end script | [recipes.md](recipes.md) |
| Just find "where does symbol X import from?" | [import_map.md](import_map.md) |

## Quick start

```bash
uv venv && uv sync
uv run examples/re_book/1_brain_evolution.py    # vision-based brain evolution
```

Install + run details: [../../README.md](../../README.md). Runnable examples
live in [../../examples/](../../examples/); `re_book/` has the most complete
end-to-end scripts.

## Token-efficiency rule (important)

These `user/` files **map and point — they do not reproduce full source.** For
exact, current signatures and values:

1. open the cited `examples/…` or `src/ariel/…` file, or
2. for third-party API facts (MuJoCo, CMA-ES, Nevergrad), use the deep
   reference in [../wiki/](../wiki/) via the [`/query`](../commands/query.md)
   command.

If a snippet here and the source ever disagree, **the source wins** — these maps
trail the code.

## The mental model

ARIEL = **genome → body/brain → MuJoCo simulation → fitness → evolution →
results DB**. A typical script: pick/decode a **body** ([bodies.md](bodies.md)),
put it in a **world** and spawn it ([simulation.md](simulation.md)), attach a
**controller** ([controllers.md](controllers.md)), score it with a **task**
([tasks_fitness.md](tasks_fitness.md)), and wrap it in an **EA**
([evolution.md](evolution.md)).
