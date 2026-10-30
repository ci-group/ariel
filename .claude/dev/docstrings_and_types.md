# Docstrings & type hints

## Docstrings — NumPy convention

`ruff.toml` sets `[lint.pydocstyle] convention = "numpy"` and `pydoclint` runs in
pre-commit. `max-doc-length = 80`. Every public function/class/module needs a
docstring; the `D…` rules enforce it.

Structure: one-line summary, then `Parameters` / `Returns` / `Raises` sections
with the underlined-dashes style. Types in the signature are authoritative, so
the parameter-type column in the docstring is optional (ARIEL files often omit
it and just name the param). Code inside docstrings is auto-formatted
(`docstring-code-format = true`), so keep snippets valid.

```python
# src/ariel/simulation/tasks/gait_learning.py
def xy_displacement(
    xy1: tuple[float, float],
    xy2: tuple[float, float],
) -> float:
    """
    Calculate the displacement between two points in 2D space.

    Parameters
    ----------
    xy1
        Coordinates of the first point (x1, y1).
    xy2
        Coordinates of the second point (x2, y2).

    Returns
    -------
    float
        The Euclidean distance between the two points.
    """
```

For longer, real-world Parameters blocks see `EASettings` in
[src/ariel/ec/ea.py](../../src/ariel/ec/ea.py) and `Tracker` in
[src/ariel/utils/tracker.py](../../src/ariel/utils/tracker.py).

## Type hints — strict everywhere

Both checkers are strict: `mypy.ini` has `strict = True`; `pyrightconfig.json`
sets `typeCheckingMode = "strict"`. Annotate everything that's public.

- **Modern syntax (3.12+)**: `X | Y` unions, built-in generics (`list[int]`,
  `dict[str, list[int]]`), and the `type` alias statement.
- **Type aliases** live in [src/ariel/parameters/ariel_types.py](../../src/ariel/parameters/ariel_types.py):

  ```python
  type Dimension = tuple[float, float, float]
  type Position = tuple[float, float, float]
  type FloatArray = npt.NDArray[ND_FLOAT_PRECISION]
  ```

  Use these instead of re-spelling the tuple/ndarray types.
- **Callables**: import from `collections.abc` (`from collections.abc import
  Callable`), not `typing`.
- **Deferred annotations**: newer structural modules start with
  `from __future__ import annotations`. Add it when forward references or
  heavy typing would otherwise need quotes.
- **MuJoCo**: `mypy.ini` sets `mujoco.*` to follow untyped imports, so MuJoCo
  objects are effectively `Any` at the boundary — annotate your own wrappers
  precisely anyway.

## Numeric types

`ariel_types.py` defines the project float/int precision aliases
(`FloatArray`, `IntArray`). Prefer them for array params so precision stays
consistent across the codebase.
