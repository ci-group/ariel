---
type: api_reference
tags: [nevergrad, optimizer, cma-es, ask-tell, parallel]
source: https://facebookresearch.github.io/nevergrad/optimization.html
---

# Nevergrad Optimizers

## Instantiation Signature

```python
optimizer = ng.optimizers.<Name>(
    parametrization,   # ng.p.* object or integer (dimension)
    budget,            # total number of evaluations (ask() calls)
    num_workers=1,     # number of parallel workers
)
```

## Available Optimizers

| Optimizer | Best for |
|-----------|----------|
| `NGOpt` / `NgIohTuned` | General adaptive meta-optimizer (recommended default) |
| `CMA` | Continuous control problems, moderate dimensions |
| `TwoPointsDE` | Differential evolution, robust |
| `TBPSA` | Noisy or overparameterized problems |
| `PSO` | Robust baseline, particle swarm |
| `OnePlusOne` | Simple continuous problems |
| `RandomSearch` | Baseline / ablation |
| `ScrHammersleySearchPlusMiddlePoint` | Highly parallel / multimodal |
| `DiscreteOnePlusOne` | Discrete optimization |
| `PortfolioDiscreteOnePlusOne` | Mixed discrete settings |
| `LHSSearch` | Latin hypercube, good for initial exploration |
| `DE` | Differential evolution base |
| `Chaining` | Sequential composition of algorithms |

All optimizers are accessible via `ng.optimizers.registry`.

> **Default assumption:** All optimizers assume centered and reduced prior — zero mean, unit standard deviation — at initialization. They can still discover distant optima.

## Ask / Tell Interface

```python
# Core loop
x = optimizer.ask()                        # returns a Candidate object
loss = function(*x.args, **x.kwargs)       # evaluate (or use x.value directly)
optimizer.tell(x, loss)                    # report result back

# Get best found solution
recommendation = optimizer.provide_recommendation()
best_value = recommendation.value          # numpy array or structured value
```

**Minimize shorthand:**
```python
recommendation = optimizer.minimize(function)
```

## Parallel Evaluation (Several Workers)

```python
# ask() num_workers candidates before any tell()
candidates = [optimizer.ask() for _ in range(num_workers)]
# ... evaluate all in parallel ...
for cand, loss in zip(candidates, losses):
    optimizer.tell(cand, loss)
```

**With ProcessPoolExecutor:**
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=n) as executor:
    optimizer.minimize(func, executor=executor, batch_mode=False)
```

**batch_mode:**
- `False` (default): optimizer updates as each result comes in
- `True`: optimizer waits for the full batch before updating

## Callbacks

```python
optimizer.register_callback("tell", callback_func)
# ask callback:  callback(optimizer)
# tell callback: callback(optimizer, candidate, value)
```

## Constraints

```python
# Hard constraints via violation functions
optimizer.tell(candidate, loss, [violation1, violation2])

# Or via minimize:
optimizer.minimize(loss_func, constraint_violations_func)

# Cheap parametrization-level constraint (applied at sampling):
optimizer.parametrization.register_cheap_constraint(lambda x: x[0] >= 1)
```

## Multiobjective

```python
optimizer.tell(ng.p.MultiobjectiveReference(), [5, 5])   # set reference point
optimizer.minimize(multiobjective_func)
front = optimizer.pareto_front(k, subset="random|loss-covering|domain-covering|EPS")
```

## Information Injection

Inject known good points before optimization starts:

```python
optimizer.suggest(*args, **kwargs)
# or manually:
candidate = optimizer.parametrization.spawn_child(new_value=known_value)
optimizer.tell(candidate, known_loss)
```

## Algorithm Chaining

```python
from nevergrad.optimization import Chaining, LHSSearch, DE

DEwithLHS = Chaining([LHSSearch, DE], [100])
# runs 100 evals of LHSSearch, then remainder of budget with DE
```

## Reproducibility

```python
import numpy as np
np.random.seed(32)                                        # affects parametrization init
optimizer.parametrization.random_state.seed(12)           # direct seeding
```

## Verbosity

```python
optimizer = ng.optimizers.CMA(parametrization=..., budget=..., verbosity=1)
# verbosity=0: silent, 1: basic, 2: verbose
```
