# dev/ — developing ARIEL

Read this when you (or an AI you direct) are **writing or modifying code inside
`src/ariel/`**. The goal: produce code that matches the house style and passes
`pre-commit` / `nox` on the first try.

## Load order

Start with the two that shape *how* you write code, then pull in the rest as
needed.

| File | Load when you are… |
|---|---|
| [coding_standards.md](coding_standards.md) | writing any code — the ruff/format rules |
| [functional_vs_oo.md](functional_vs_oo.md) | deciding function vs `@dataclass` vs class |
| [docstrings_and_types.md](docstrings_and_types.md) | adding docstrings or type hints |
| [testing_and_tooling.md](testing_and_tooling.md) | running tests, coverage, pre-commit, mypyc |
| [repo_map.md](repo_map.md) | deciding *where* new code belongs |
| [contribution_workflow.md](contribution_workflow.md) | preparing a PR |

## The one-paragraph version

ARIEL is strict: `ruff` runs `select = ["ALL"]`, `mypy` and `pyright` are both in
strict mode, docstrings are NumPy-style, lines are ≤ 80 chars, and test coverage
must be **100%**. Default to **free functions and `@dataclass`**; introduce a
class only when state, identity, a framework contract, or a fluent API genuinely
demands it (see [functional_vs_oo.md](functional_vs_oo.md)). Don't restate the
deep API facts that live in [../wiki/](../wiki/) — link to them.

## Canonical config files (source of truth)

These files *are* the standard; the `dev/` docs only distill them. If they
disagree, the config wins — update the doc.

- [ruff.toml](../../ruff.toml) — lint + format
- [mypy.ini](../../mypy.ini) — type checking
- [pyproject.toml](../../pyproject.toml) — deps, pytest, coverage
- [noxfile.py](../../noxfile.py) — task automation
- [.pre-commit-config.yaml](../../.pre-commit-config.yaml) — commit-time gates
