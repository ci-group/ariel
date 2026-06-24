---
type: source_summary
tags: [nevergrad, parametrization, ng.p, source]
source: https://facebookresearch.github.io/nevergrad/parametrization.html
date_ingested: 2026-06-24
---

# Nevergrad Parametrization

Official Nevergrad docs page for the `ng.p.*` parametrization classes (Array, Scalar, Log, Choice, TransitionChoice, Tuple, Dict, Instrumentation) and their configuration methods (`set_mutation`, `set_bounds`, `set_integer_casting`, `spawn_child`, `register_cheap_constraint`). Note: the upstream page states parametrization "is still a work in progress and changes are on their way" — confirm against the installed Nevergrad version.

## Entity Pages Created

- [[Nevergrad_Parametrization]] — already covered all classes and methods exposed by this docs page; no new content to append from this re-fetch.

## Gaps Not Covered by Upstream Docs

The docs page does NOT enumerate:
- The full `set_bounds(method=...)` options (e.g. `clipping`, `arctan`, `tanh`, `gaussian`, `bouncing`) — these exist in source but not in this page. Source: `nevergrad/parametrization/data.py` `BoundsChecker` / `Bound`.
- Per-element sigma support on `ng.p.Array` (whether `set_mutation` accepts an array). Workaround documented in [[Nevergrad_Parametrization]]: use `ng.p.Tuple` with separate Array objects.
- Custom mutation classes (`Gaussian`, `Cauchy`) — not present on this docs page. Source: `nevergrad/parametrization/mutation.py`.
