---
type: api_reference
tags: [nevergrad, cma-es, optimizer, parametrized]
source: https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/optimization/optimizerlib.py
date_ingested: 2026-06-24
---

# ParametrizedCMA

Nevergrad's configurable factory for CMA-ES variants. Each `ng.optimizers.CMA*` name is a pre-set `ParametrizedCMA` instance. The `inopts` dict is forwarded to the underlying [[CMAEvolutionStrategy]] (pycma) — every `CMAOptions` key documented there is reachable through this class.

## Signature

```python
def __init__(
    self,
    *,
    scale: float = 1.0,
    elitist: bool = False,
    popsize: tp.Optional[int] = None,
    popsize_factor: float = 3.0,
    diagonal: bool = False,
    zero: bool = False,
    high_speed: bool = False,
    fcmaes: bool = False,
    random_init: bool = False,
    inopts: tp.Optional[tp.Dict[str, tp.Any]] = None,
    algorithm: str = "quad",
) -> None:
```

## Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `scale` | float | `1.0` | Multiplier on initial σ. Stacks with [[Nevergrad_Parametrization]] `set_mutation(sigma=...)`. |
| `elitist` | bool | `False` | Forward to pycma `CMA_elitist`. Keeps best parent across generations. |
| `popsize` | int or None | `None` | λ. If `None`, computed from `popsize_factor` and dimension via pycma default `4 + 3·ln(n)`. |
| `popsize_factor` | float | `3.0` | Multiplier on the default popsize formula. |
| `diagonal` | bool | `False` | Forward to pycma `CMA_diagonal=True`. Restricts C to diagonal — cheaper, no rotation learned. |
| `zero` | bool | `False` | Zero rank-one term (internal experimental). |
| `high_speed` | bool | `False` | Use the faster but less robust backend. |
| `fcmaes` | bool | `False` | Use the `fcmaes` backend instead of `pycma`. |
| `random_init` | bool | `False` | Randomize mean initialization instead of using parametrization init. |
| `inopts` | dict or None | `None` | Forwarded verbatim to [[CMAEvolutionStrategy]] CMAOptions (e.g. `tolflatfitness`, `BoundaryHandler`, `CMA_active`, `seed`). |
| `algorithm` | str | `"quad"` | Internal selection for chained / variant algorithm. |

## Pre-registered Variants

```python
OldCMA       = ParametrizedCMA().set_name("OldCMA", register=True)
LargeCMA     = ParametrizedCMA(scale=3.0).set_name("LargeCMA", register=True)
LargeDiagCMA = ParametrizedCMA(scale=3.0, diagonal=True).set_name("LargeDiagCMA", register=True)
TinyCMA      = ParametrizedCMA(scale=0.33).set_name("TinyCMA", register=True)
CMAbounded   = ParametrizedCMA(scale=1.5884, popsize_factor=1, elitist=True,  diagonal=True,  fcmaes=False).set_name("CMAbounded", register=True)
CMAsmall     = ParametrizedCMA(scale=0.3607, popsize_factor=3, elitist=False, diagonal=False, fcmaes=False).set_name("CMAsmall", register=True)
CMAstd       = ParametrizedCMA(scale=0.4699, popsize_factor=3, elitist=False, diagonal=False, fcmaes=False).set_name("CMAstd", register=True)
CMApara      = ParametrizedCMA(scale=0.8905, popsize_factor=8, elitist=True,  diagonal=True,  fcmaes=False).set_name("CMApara", register=True)
CMAtuning    = ParametrizedCMA(scale=0.4847, popsize_factor=1, elitist=True,  diagonal=False, fcmaes=False).set_name("CMAtuning", register=True)
DiagonalCMA  = ParametrizedCMA(diagonal=True).set_name("DiagonalCMA", register=True)
EDCMA        = ParametrizedCMA(diagonal=True, elitist=True).set_name("EDCMA", register=True)
SDiagonalCMA = ParametrizedCMA(diagonal=True, zero=True).set_name("SDiagonalCMA", register=True)
FCMA         = ParametrizedCMA(fcmaes=True).set_name("FCMA", register=True)
```

| Name | When to use |
|---|---|
| `OldCMA` | Plain defaults — baseline. |
| `CMAstd` / `CMAsmall` | Continuous control, moderate dim. `CMAstd` has slightly larger initial σ. |
| `CMApara` | Highly parallel (`popsize_factor=8`, elitist+diagonal) — good for large `num_workers`. |
| `CMAbounded` | Box-bounded search; elitist+diagonal. |
| `CMAtuning` | Hyperparameter tuning style (`popsize_factor=1`). |
| `DiagonalCMA` / `EDCMA` / `LargeDiagCMA` | High-dim where full C is too expensive or rotation is irrelevant. |
| `FCMA` | Uses `fcmaes` backend (different impl). |
| `TinyCMA` / `LargeCMA` | σ-scaled presets. |

## Examples

### Drop-in replacement for `ng.optimizers.CMA`

```python
optimizer = ng.optimizers.ParametrizedCMA(
    popsize=24,
    elitist=False,
    diagonal=False,
    inopts={
        "tolflatfitness": 50,      # patience on plateaus (see CMA-ES_Practical_Concerns §B.4)
        "BoundaryHandler": "BoundPenalty",
        "CMA_active": True,
        "seed": 42,
        "verbose": -9,
    },
)(parametrization=param, budget=200 * 24, num_workers=24)
```

### Diagonal warm-up for n≥10

```python
optimizer = ng.optimizers.ParametrizedCMA(
    diagonal=True,
    popsize=24,
    inopts={"CMA_diagonal": 50},   # diagonal-only for first 50 iters
)(parametrization=param, budget=B, num_workers=W)
```

## Notes

- **`inopts` is the escape hatch** — anything pycma's `CMAOptions` accepts (see [[CMAEvolutionStrategy]] §CMAOptions Keys) can be set here. Use it for `tolflatfitness`, `CMA_diagonal` (int form), `BoundaryHandler`, `integer_variables`, etc. that `ParametrizedCMA` doesn't expose directly.
- **`scale` vs `set_mutation`**: both multiply σ. Set sigma at the parametrization level for clarity; leave `scale=1.0`.
- **`fcmaes=True`** switches to a different implementation (`fcmaes` package). Behavior and option support differ from pycma.
- **No IPOP/BIPOP wrappers** found in this file. Restart strategies in Nevergrad live elsewhere (likely `nevergrad/optimization/recastlib.py` or `families.py`) — flagged as a gap.

## See Also

- [[Nevergrad_Optimizers]]
- [[Nevergrad_Parametrization]]
- [[CMAEvolutionStrategy]]
- [[CMA-ES_Parameters]]
- [[CMA-ES_Practical_Concerns]]
