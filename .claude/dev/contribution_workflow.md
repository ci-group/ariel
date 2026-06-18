# Contribution workflow

Condensed from the canonical [CONTRIBUTING.md](../../CONTRIBUTING.md) — read that
for the full text and links. This page is the checklist.

## Before you start

- **Open an issue first** for anything non-trivial, to validate the approach
  with the maintainers. Bugs and feature requests both go on the
  [issue tracker](https://github.com/Jacopo-DM/ariel/issues).
- Bug reports should include OS, Python version, project version, repro steps,
  and ideally a failing test case.

## The loop

1. Branch off the default branch (don't commit straight to it).
2. Set up: `uv venv` → `uv sync` (see
   [testing_and_tooling.md](testing_and_tooling.md)).
3. Install the pre-commit hook: `nox --session=pre-commit -- install`.
4. Write code to the house standard ([coding_standards.md](coding_standards.md),
   [docstrings_and_types.md](docstrings_and_types.md),
   [functional_vs_oo.md](functional_vs_oo.md)).
5. Add/extend tests in [tests/](../../tests/) — coverage must stay **100%**.
6. Run `nox` (full suite); it must pass **without errors or warnings**.
7. Update docs if you added functionality.

## PR acceptance criteria (from CONTRIBUTING.md)

- [ ] Nox suite passes with no errors *and* no warnings.
- [ ] Unit tests included; 100% coverage maintained.
- [ ] Documentation updated for new functionality.
- [ ] Pre-commit (ruff + pydoclint + prettier) clean.

Submitting early for iteration is welcome — but the gates above are what get a
PR merged. Distributed under GPL-3.0; by contributing you agree to those terms
and the [Code of Conduct](../../CODE_OF_CONDUCT.md).
