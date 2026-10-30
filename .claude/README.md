# `.claude/` — AI assistant context for ARIEL

This folder gives an AI assistant (Claude or otherwise) the context it needs to
work on **ARIEL** efficiently. It is split by *intent*: are you **developing**
ARIEL, or **building something with** ARIEL?

## Where to look first

| If you are… | Read first | Then |
|---|---|---|
| **Writing / modifying ARIEL source** (`src/ariel/…`) | [dev/00_index.md](dev/00_index.md) | the relevant `dev/*.md` |
| **Building an experiment with ARIEL** (your own script) | [user/00_index.md](user/00_index.md) | the one matching `user/*.md` |
| **Needing exact MuJoCo / CMA-ES / Nevergrad facts** | [wiki/](wiki/) (+ `/query`) | [SCHEMA.md](SCHEMA.md) for the wiki rules |

Load only the file you need. The `dev/` and `user/` files are *maps and rules*;
they point at the real source and the deep `wiki/` reference rather than copying
either. This keeps lookups token-efficient and answers traceable.

## The three layers

- **`dev/`** — how to write ARIEL code that passes pre-commit on the first try:
  the [ruff.toml](../ruff.toml) standards and the *functional-first,
  classes-only-when-justified* philosophy, with real call sites.
- **`user/`** — a directed, per-subsystem map of the public API (imports,
  signatures, runnable snippets) so an AI loads one area instead of grepping
  all of `src/`.
- **`wiki/`** — 68 deep reference pages on third-party APIs and algorithms,
  governed by [SCHEMA.md](SCHEMA.md) and fed by the [`/ingest`](commands/ingest.md)
  command. Queried via [`/query`](commands/query.md).

## Golden rules (full detail in `dev/`)

- Line length **80**, 4-space indent, LF, **double quotes** (Black-compatible).
- **NumPy-style** docstrings; strict `mypy` + strict `pyright`; Python **3.12+**.
- **Functional-first**: free functions + `@dataclass`; reach for a class only
  when state, identity, a framework contract, or a fluent API demands it.
- `ruff` runs `select = ["ALL"]` — assume a rule is on unless `dev/` says it is
  ignored. **100% test coverage** is enforced.

## Contents

```
.claude/
  README.md      <- you are here (router)
  SCHEMA.md      wiki taxonomy + formatting rules
  settings.json  Claude Code permissions for this project
  commands/      /ingest and /query slash commands
  dev/           develop ARIEL (standards + philosophy)
  user/          build with ARIEL (per-area API maps)
  wiki/          deep third-party/algorithm reference
```
