# Coding standards

Distilled from [ruff.toml](../../ruff.toml). Ruff is both the linter *and* the
formatter here, with `fix = true`, `unsafe-fixes = true`, and `preview = true`.
When in doubt, run `ruff format` + `ruff check --fix` — but write it right the
first time using the rules below.

## Formatting (non-negotiable)

| Rule | Value |
|---|---|
| Target Python | `py312` |
| Line length | **80** (`max-line-length`, `max-doc-length` both 80) |
| Indent | 4 spaces, never tabs |
| Quotes | **double** (`"…"`) |
| Line ending | `lf` |
| Magic trailing comma | **respected** — a trailing comma forces multi-line |
| Docstring code | auto-formatted (`docstring-code-format = true`) |

The magic trailing comma matters: `f(a, b,)` will be expanded onto multiple
lines by the formatter. Add the trailing comma when you *want* the multi-line
form; omit it to let short calls stay on one line.

## Linting: `select = ["ALL"]`

Every ruff rule is on. Assume a rule applies unless it is in the ignore list
below. Don't add `# noqa` to dodge a rule unless you can justify it the way the
existing ignores are justified — and prefer fixing the code.

### Active ignores (and why)

| Code | Why it's ignored |
|---|---|
| `D203` | conflicts with `D211` (blank line before class) |
| `D212` | conflicts with `D213` (multi-line summary placement) |
| `TID252` | relative imports are allowed in this package |
| `ISC001` | conflicts with the formatter (`COM812` pairing) |
| `CPY001` | no per-file copyright header required |
| `TD003` | TODOs don't need an issue link |
| `INP001` | implicit-namespace `__init__` not required everywhere |
| `S311` | `random` is fine here — not used for cryptography |
| `PLR2004` | magic values in comparisons are tolerated |

Everything else under `ALL` is live (pyflakes, pycodestyle, isort, pydocstyle,
pyupgrade, bugbear, security `S…`, complexity `PLR…`, etc.).

### Per-file carve-outs

- `**/tests/**/*.py`, `*/test_*.py`, `noxfile.py` → `S101` allowed (asserts are
  fine in tests / nox config).

## Imports — three blocks

Observed everywhere in `src/ariel/`. Group with a blank line between, in this
order, each alphabetised by `isort`:

```python
# Standard library
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Third-party libraries
import mujoco as mj
import numpy as np
from pydantic import BaseModel

# Local libraries
from ariel.ec.individual import Individual
from ariel.parameters.ariel_types import Dimension
```

No wildcard imports. `mujoco` is commonly aliased `mj` inside `src/`, but
examples import it as `mujoco` — match the file you're in.

## Naming & misc

- **Dummy variables** must be underscore-prefixed (`_`, `_unused`); the regex
  `dummy-variable-rgx` enforces this. `for _ in range(n):` is the idiom.
- **Task tags** recognised in comments: `TODO`, `WARN`, and the checkbox set
  `[ ]`, `[~]`, `[x]`, `[?]`, `[!]`. Use these, not ad-hoc markers.
- Excluded from linting entirely: `.venv`, `.git`, `.typings`.

## Before / after

```python
# Before — single quotes, no trailing comma, >80 cols, untyped
def make(world, pos): return world.spawn(robot, position=pos)

# After — house style
def make(
    world: BaseWorld,
    pos: tuple[float, float, float],
) -> None:
    """Spawn ``robot`` into ``world`` at ``pos``."""
    world.spawn(robot, position=pos)
```

See [docstrings_and_types.md](docstrings_and_types.md) for the docstring/type
detail and [functional_vs_oo.md](functional_vs_oo.md) for structure choices.
