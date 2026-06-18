# Testing & tooling

ARIEL uses **uv** for environments, **nox** for task automation, **pytest** for
tests, and **pre-commit** for commit-time gates. Source of truth:
[noxfile.py](../../noxfile.py), [pyproject.toml](../../pyproject.toml),
[.pre-commit-config.yaml](../../.pre-commit-config.yaml),
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## Environment

`pyproject.toml` requires **Python ≥ 3.12**. Set up with uv:

```bash
uv venv
uv sync
uv run examples/re_book/1_brain_evolution.py   # run anything via `uv run`
```

(`CONTRIBUTING.md` also shows `uv install`; the README's `uv venv` + `uv sync`
is the current flow.)

## Tests & coverage

```bash
nox                       # full suite
nox --session=tests       # unit tests only
nox --list-sessions       # see available sessions
```

- Tests live in [tests/](../../tests/), written with `pytest`
  (`pytest`, `pytest-cov`, `pytest-modern`).
- **Coverage must be 100%** — `pyproject.toml` sets
  `[tool.coverage.report] fail_under = 100`. New code needs new tests or the
  suite fails. Excluded lines: `pragma: no cover`, `if TYPE_CHECKING:`.
- Asserts are allowed in test files (ruff `S101` carve-out for `tests/`,
  `test_*.py`, `noxfile.py`).

When adding a feature, add the test in the matching `tests/…` path **in the same
change** — don't defer it, or coverage drops below 100% and CI blocks.

## Pre-commit

Install the git hook once:

```bash
nox --session=pre-commit -- install
```

The hooks (see [.pre-commit-config.yaml](../../.pre-commit-config.yaml)) run
`ruff check` + `ruff format`, `pydoclint` (NumPy docstring linting), `prettier`,
and TOML/YAML validators. Code that isn't ruff-clean and numpydoc-compliant will
be rejected at commit time — write it to standard up front (see
[coding_standards.md](coding_standards.md)).

## Optional: mypyc compiled build

ARIEL can be compiled with `mypyc` for speed (opt-in, off by default). Triggered
by the `ARIEL_COMPILE_MYPYC=1` env var via [setup.py](../../setup.py); a
`tests_compiled` nox session builds and installs the compiled package. You won't
need this for normal development — only when validating the compiled path.
