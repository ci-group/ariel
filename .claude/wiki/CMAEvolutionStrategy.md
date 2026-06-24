---
type: api_reference
tags: [cma-es, pycma, optimizer, python, class]
source: https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html
date_ingested: 2026-06-24
---

# CMAEvolutionStrategy

Main class of the `pycma` library — a reference Python implementation of CMA-ES. This is the engine that Nevergrad's `ng.optimizers.CMA` / `ParametrizedCMA` wraps. Options passed to Nevergrad via `inopts=...` are forwarded here.

## Signature

```python
cma.CMAEvolutionStrategy(x0, sigma0, inopts=None, options=None)
```

## Parameters

| Name | Type | Description |
|---|---|---|
| `x0` | array-like or callable | Initial solution (phenotype). Callable form is re-evaluated each restart. |
| `sigma0` | float | Initial overall standard deviation. Per [[CMA-ES_Parameters]]: `0.3 * (upper - lower)`. |
| `inopts` | dict or None | Options dict — keys listed in `## CMAOptions Keys` below. |
| `options` | dict or None | Alias for `inopts` (for consistency with `fmin2`). |

## Core Methods

| Method | Signature | Purpose |
|---|---|---|
| `ask` | `ask(number=None, xmean=None, sigma_fac=1, gradf=None, args=(), **kwargs)` | Sample new candidate solutions |
| `tell` | `tell(solutions, function_values, check_points=None, copy=False, constraints_values=None)` | Update with fitness values |
| `stop` | `stop(check=True, ignore_list=(), check_in_same_iteration=False, get_value=None)` | Check termination conditions; returns dict of triggered criteria |
| `ask_and_eval` | `ask_and_eval(func, args=(), gradf=None, number=None, xmean=None, sigma_fac=1, evaluations=1, aggregation=np.median, kappa=1, parallel_mode=False)` | Sample + evaluate in one call; supports re-evaluation for noisy fitness via `evaluations>1` |
| `disp` | `disp(modulo=None, overwrite=None)` | Print current state (single-line) |
| `optimize` | `optimize(objective_fct, ...)` | Run full optimization loop |
| `inject` | `inject(solutions, force=None)` | Inject external candidate solutions (e.g. warm-start) |
| `feed_for_resume` | `feed_for_resume(X, function_values)` | Resume from solution history |
| `reset_options` | `reset_options(**kwargs)` | Clear termination state and set options |
| `result_pretty` | `result_pretty(number_of_restarts=0, time_str=None, fbestever=None)` | Pretty-print results |
| `pickle_dumps` | `pickle_dumps()` | Serialize instance to bytes |
| `plot` | `plot(*args, **kwargs)` | Plot state via matplotlib |

## Attributes / Properties

| Name | Type | Description |
|---|---|---|
| `result` | `CMAEvolutionStrategyResult2` | Result object (`.xbest`, `.fbest`, `.evals`, ...) |
| `popsize` | int | λ |
| `mean` | ndarray | Distribution mean (phenotype) |
| `sigma` | float | Step-size |
| `stds` | ndarray | Coordinate-wise standard deviations (phenotypic) |
| `condition_number` | float | cond(C); termination at >1e14 per [[CMA-ES_Practical_Concerns]] |
| `isotropic_mean_shift` | float | Normalized last mean shift |
| `best` | Solution | Best solution found |
| `best_feasible` | Solution | Best feasible (with constraints) |
| `countiter` | int | Iteration counter |
| `countevals` | int | Function evaluation counter |

## CMAOptions Keys

### Population strategy
| Key | Default | Role |
|---|---|---|
| `popsize` | `4 + 3*log(n)` | λ. See [[CMA-ES_Parameters]] §Population Size Guidance. |
| `CMA_mu` | `popsize/2` | μ — number of parents selected for recombination |
| `CMA_cmean` | adaptive | Learning rate for mean update |
| `CMA_rankmu` | `True` | Enable rank-μ update |
| `CMA_rankone` | `True` | Enable rank-one update |
| `CMA_diagonal` | `0` for n≤40, else diagonal | Iterations to run diagonal-only (cheap, no rotation learned). Use a positive int for "diagonal for first N iters then full" |
| `CMA_active` | `True` for small populations | Active CMA with negative weights |
| `CMA_elitist` | `False` | Elitist selection |
| `CMA_mirrors` | `0` | Number of mirrored vectors (see [[CMA-ES_Mirrored_Sampling]]) |

### Step-size
| Key | Default | Role |
|---|---|---|
| `AdaptSigma` | `True` | Enable σ adaptation (CSA) |
| `adapt_sigma_consequence` | `1` | Adjust λ on σ divergence |

### Termination
| Key | Default | Role |
|---|---|---|
| `maxiter` | `100 + 150*log(n)` | Max generations |
| `maxfevals` | `inf` | Max function evaluations |
| `tolx` | `1e-7` | Convergence in x — per [[CMA-ES_Practical_Concerns]] §B.3 default is `1e-12 * σ⁽⁰⁾` in Hansen's notation; pycma default is looser |
| `tolfun` | `1e-12` | Convergence in f-value |
| `tolstagnation` | `100 + 2*log(n)` | Stagnation horizon |
| `tolflatfitness` | `int(100 + 100*n/popsize)` | Plateau patience — directly addresses [[CMA-ES_Practical_Concerns]] §B.4 flat-fitness |

### Scaling & boundaries
| Key | Default | Role |
|---|---|---|
| `bounds` | `None` | `[lower, upper]`, each scalar/array/None |
| `BoundaryHandler` | `BoundPenalty` | Penalty-based handling (matches [[CMA-ES_Practical_Concerns]] §B.5 recommendation). Alternative: `BoundTransform` (arctan-style squash). **Pass a class, not a string** — `"BoundPenalty"` triggers `AttributeError: 'str' object has no attribute 'has_bounds'`. Usually you don't need to set this at all — just pass `bounds=[lo, hi]` and the default `BoundPenalty` is used. |
| `mindx` | `0` | Min coordinate std |
| `minstd` | `0` | Min step-size |
| `maxstd` | `inf` | Max step-size |
| `integer_variables` | `[]` | Indices of integer dims |

### Output & debugging
| Key | Default | Role |
|---|---|---|
| `verbose` | `3` | Verbosity (`-9` suppresses) |
| `verb_disp` | `100` | Display interval |
| `verb_log` | `1` | Log interval |
| `seed` | `None` | RNG seed |
| `callback` | `None` | User callback(s) |

### Advanced
| Key | Default | Role |
|---|---|---|
| `CMA_on` | `True` | Enable CMA (off ⇒ pure ES) |
| `conditioncov_alleviate` | `0` | Cap condition number |
| `transformation` | `None` | Custom genotype↔phenotype map |
| `metric` | `None` | Custom metric |

## Examples

### Basic loop

```python
import cma

es = cma.CMAEvolutionStrategy(x0=[0]*4, sigma0=0.3)

while not es.stop():
    X = es.ask()
    fit = [cma.ff.sphere(x) for x in X]
    es.tell(X, fit)
    es.disp()

print(es.result.xbest)
print(es.result.fbest)
print(es.result.evals)
```

### Compact form

```python
es = cma.CMAEvolutionStrategy(x0, sigma0)
while not es.stop():
    es.tell(*es.ask_and_eval(objective_func))
```

### With options

```python
opts = {'maxiter': 500, 'popsize': 50, 'seed': 42, 'verbose': -9}
es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
```

## Notes

- **Forwarded from Nevergrad**: `ng.optimizers.ParametrizedCMA(..., inopts={...})` passes the dict here. Use this to set `CMA_diagonal`, `CMA_active`, `tolflatfitness`, `BoundaryHandler`, etc. that the Nevergrad wrapper does not expose directly.
- **Noisy fitness**: prefer `ask_and_eval(..., evaluations=k, aggregation=np.median)` for per-candidate re-evaluation. Per [[CMA-ES_Parameters]] §What NOT to Change, also consider increasing λ.
- **Bound handling**: `BoundPenalty` (default) matches the penalty-based scheme in [[CMA-ES_Practical_Concerns]] §B.5 — safer than naive clipping for CMA distributional assumptions.
- **`CMA_diagonal > 0`**: useful for n ≥ 10 when early generations would otherwise waste compute learning the rotation matrix on a roughly axis-aligned problem.

## See Also

- [[CMA-ES_Algorithm]]
- [[CMA-ES_Parameters]]
- [[CMA-ES_Practical_Concerns]]
- [[Nevergrad_Optimizers]]
