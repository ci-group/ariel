# Contributing to ARIEL

Thank you for your interest in contributing to **ARIEL** — an open-source evolutionary robotics framework.
ARIEL is licensed under [GPL-3.0](https://opensource.org/licenses/GPL-3.0) and welcomes contributions of all kinds: bug reports, feature requests, documentation improvements, and pull requests.

---

## Table of Contents

1. [Resources](#resources)
2. [Code of Conduct](#code-of-conduct)
3. [Reporting Bugs](#reporting-bugs)
4. [Requesting Features](#requesting-features)
5. [Development Environment](#development-environment)
6. [Project Structure](#project-structure)
7. [Code Style & Quality](#code-style--quality)
8. [Type Checking](#type-checking)
9. [Writing Docstrings](#writing-docstrings)
10. [Testing](#testing)
11. [Documentation](#documentation)
12. [Pre-commit Hooks](#pre-commit-hooks)
13. [Submitting a Pull Request](#submitting-a-pull-request)
14. [Commit Message Guidelines](#commit-message-guidelines)

---

## Resources

| Resource | Link |
|---|---|
| Source code | <https://github.com/ci-group/ariel> |
| Issue tracker | <https://github.com/ci-group/ariel/issues> |
| Changelog | <https://github.com/ci-group/ariel/releases> |

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](../../CODE_OF_CONDUCT.md).
By participating, you agree to uphold these standards. Violations may be reported to the project maintainers.

---

## Reporting Bugs

Report bugs via the [Issue Tracker](https://github.com/ci-group/ariel/issues).

A good bug report answers all of the following:

- **OS & Python version** — e.g. Ubuntu 22.04, Python 3.12.3
- **What you did** — minimal code snippet to reproduce the problem
- **What you expected** — the behaviour you anticipated
- **What happened instead** — exact error message or observed behaviour

The fastest path to a fix is a minimal reproducible example or a failing test case.

---

## Requesting Features

Open a [GitHub Issue](https://github.com/ci-group/ariel/issues) and describe:

- The problem the feature would solve
- A rough API sketch or usage example
- Any relevant prior art or references

It is recommended to discuss the idea in an issue **before** writing code, so the approach can be validated early.

---

## Development Environment

### Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| Python ≥ 3.12.9 | Runtime | <https://www.python.org> |
| [uv](https://docs.astral.sh/uv/) | Package & environment manager | `pip install uv` |
| [Nox](https://nox.thea.codes/) | Task runner / test sessions | `pip install nox` |
| [pre-commit](https://pre-commit.com/) | Git hook manager | `pip install pre-commit` |

### Installation

Clone the repository and install the package in editable mode with all development dependencies:

```console
$ git clone https://github.com/ci-group/ariel.git
$ cd ariel
$ uv venv
$ uv sync
```

Verify the installation:

```console
$ uv run ariel/examples/a_mujoco/0_mujoco_launcher.py
```

---

## Project Structure

```
ariel/
├── src/ariel/              # Main package source
│   ├── body_phenotypes/    # Robot morphology (robot systems, CPPN-NEAT)
│   ├── ec/                 # Evolutionary computation engines, operators etc
│   ├── simulation/         # Physics simulation (MuJoCo)
│   ├── parameters/         # Configuration and parameter management
│   ├── utils/              # Shared utilities
│   └── visualisation/      # Dashboard and visualisation tools
├── tests/
│   ├── unit/               # Unit tests
│   └── functional/         # Functional / integration tests
├── docs/                   # Sphinx documentation
│   └── source/             # RST/Markdown source pages
├── examples/               # Standalone runnable examples
├── .github/workflows/      # CI/CD pipelines
├── pyproject.toml          # Project metadata and tool config
├── ruff.toml               # Ruff linter/formatter config
├── mypy.ini                # MyPy type checker config
├── pyrightconfig.json      # Pyright type checker config
└── noxfile.py              # Nox session definitions
```

A general heuristic for package dependencies: if adding a feature would introduce a dependency from one ARIEL sub-package to another, reconsider the design.
Refer to the package diagram in the documentation for the intended dependency direction.

---

## Code Style & Quality

ARIEL enforces a consistent code style using **Ruff** for both linting and formatting.

### Key formatting rules

| Setting | Value |
|---|---|
| Target Python | 3.12.9 |
| Line length | 80 characters |
| Quote style | Double quotes |
| Indent | 4 spaces |
| Line endings | LF (`\n`) |
| Docstring convention | NumPy |

All linting rules (`ALL`) are enabled except a small set of acknowledged exceptions (see `ruff.toml`).

### Running the formatter

```console
$ uv run ruff format .
```

### Running the linter

```console
$ uv run ruff check . --fix
```

Auto-fixes are applied automatically; review the diff before committing.

---

## Type Checking

All public API code must be **fully typed**. ARIEL runs two type checkers in strict mode:

| Tool | Config | Command |
|---|---|---|
| [MyPy](https://mypy.readthedocs.io/) | `mypy.ini` | `uv run mypy src/` |
| [Pyright](https://github.com/microsoft/pyright) | `pyrightconfig.json` | `uv run pyright` |

Both run in **strict** mode. Do not use `# type: ignore` comments without a clear explanation.

ARIEL ships a `py.typed` marker (PEP 561), so downstream packages can consume its type information.

---

## Writing Docstrings

All public functions, classes, and methods must have a docstring following the **NumPy style**:

```python
def evaluate(robot: Robot, config: SimConfig) -> float:
    """Evaluate the fitness of a robot in simulation.

    Parameters
    ----------
    robot : Robot
        The robot morphology and controller to evaluate.
    config : SimConfig
        Simulation configuration (duration, timestep, etc.).

    Returns
    -------
    float
        Scalar fitness value. Higher is better.

    Raises
    ------
    SimulationError
        If the physics engine fails to initialise.
    """
```

Docstrings are validated by **pydoclint** (NumPy style) via pre-commit.
Code examples inside docstrings are auto-formatted by Ruff.

---

## Testing

### Running the test suite

```console
$ nox
```

List all available Nox sessions:

```console
$ nox --list-sessions
```

Run only unit tests:

```console
$ nox --session=tests
```

Run tests directly with pytest (faster iteration):

```console
$ uv run pytest tests/unit/
$ uv run pytest tests/functional/
```


### Writing tests

- Place **unit tests** in `tests/unit/` and **functional tests** in `tests/functional/`.
- Use [pytest](https://docs.pytest.org/) conventions — no custom test classes unless necessary.
- Prefer small, focused tests with descriptive names.
- `assert` statements are allowed in tests (the `S101` rule is disabled in test files).

---

## Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/), the [Furo](https://pradyunsg.me/furo/) theme, and [MyST Parser](https://myst-parser.readthedocs.io/) for Markdown support.
API reference pages are generated automatically via [AutoAPI](https://sphinx-autoapi.readthedocs.io/).

### Building docs locally

```console
$ nox --session=docs
```

This cleans the `docs/_build/` and `docs/_autoapi/` directories, then launches a live-reload server at `http://localhost:8000`.

### Documentation guidelines

- All public symbols must have a complete NumPy-style docstring (see [Writing Docstrings](#writing-docstrings)).
- New features should include a tutorial page or example notebook in `docs/source/` or `examples/`.
- If your changes add or remove a public API, update the relevant `.md` source page.
- Documentation is deployed automatically to GitHub Pages on every push to `main`.

---

## Submitting a Pull Request

1. **Open an issue first** for non-trivial changes to align on the approach before writing code.
2. **Fork** the repository and create a branch from `main`:
   ```console
   $ git checkout -b feat/my-feature
   ```
3. Make your changes, ensuring:
   - The full test suite passes (`nox`)
   - Code coverage is maintained
   - New public APIs are fully typed and documented
4. Push your branch and open a Pull Request against `main`.
5. The PR description should explain **what** changed and **why**.

### PR checklist

- [ ] Tests pass (`nox`)
- [ ] Code coverage (`uv run pytest --cov=ariel`)
- [ ] Type checks pass (`uv run mypy src/` and `uv run pyright`)
- [ ] All new public symbols have NumPy-style docstrings
- [ ] Documentation updated if API changed
- [ ] PR description explains the motivation

> **Merging policy:**
> - Into `main`: use **Squash and Merge** to keep a clean linear history.

---

## Commit Message Guidelines

ARIEL follows a conventional commit style to make the history readable:

```
<type>(<scope>): <short summary>

<optional body>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

**Examples:**

```
feat(ec): add restart-from-database support for EC engines
fix(simulation): correct MuJoCo worker timestep initialisation
docs(contributing): expand PR checklist and type-checking section
test(body_phenotypes): add coverage for CPPN-NEAT edge cases
```

Keep the summary line under 72 characters.
Use the body to explain *why* the change was made, not *what* — the diff already shows the what.
