---
type: api_reference
tags: [nevergrad, parametrization, ng.p, array, instrumentation, mutation, bounds]
source: https://facebookresearch.github.io/nevergrad/parametrization.html
---

# Nevergrad Parametrization (ng.p.*)

> **Note:** Nevergrad documentation states parametrization is "still a work in progress" with "breaking changes" anticipated. Verify against installed version.

## Core Classes

### ng.p.Array

Continuous multidimensional parameter backed by a NumPy array.

```python
param = ng.p.Array(shape=(n,))          # shape-based init
param = ng.p.Array(init=np.array(...))  # value-based init
```

**Configuration methods (chainable):**
```python
param.set_mutation(sigma=10)            # scalar sigma — same for all dims
param.set_bounds(lower, upper)          # box constraints
param.set_integer_casting()             # round values to int
```

> **Per-element sigma**: The docs show `sigma` as a **scalar only** in all examples. Array sigma is **[unverified — not confirmed in docs]**. Use separate `ng.p.Array` objects per group if different sigmas are needed (see Tuple below).

### ng.p.Scalar

Single numeric value (inherits all `Array` methods).

```python
param = ng.p.Scalar()
param = ng.p.Scalar(init=0.5).set_mutation(sigma=0.1)
```

### ng.p.Log

Logarithmically distributed value between bounds.

```python
param = ng.p.Log(lower=0.01, upper=1.0)
```

### ng.p.Choice

Unordered categorical selection (uses softmax internally).

```python
param = ng.p.Choice(["option_a", "option_b", "option_c"])
param = ng.p.Choice([1, 2, 4, 8, 16])
```

### ng.p.TransitionChoice

Ordered categorical with local transitions (adjacent items more likely).

```python
param = ng.p.TransitionChoice(range(10), repetitions=3)
```

### ng.p.Tuple

Composes multiple parameters into one. Useful for **per-group sigma**:

```python
tubes_p   = ng.p.Array(init=tubes_init).set_mutation(sigma=0.27)
weights_p = ng.p.Array(init=weights_init).set_mutation(sigma=0.10)
param     = ng.p.Tuple(tubes_p, weights_p)

# Access values after ask():
candidate = optimizer.ask()
tubes_val, weights_val = candidate.value   # tuple unpacking
genome = np.concatenate([tubes_val, weights_val])
```

### ng.p.Dict

Dictionary-keyed composition:

```python
param = ng.p.Dict(
    log   = ng.p.Log(lower=0.01, upper=1.0),
    array = ng.p.Array(shape=(2,)),
    char  = ng.p.Choice(["a", "b", "c"]),
)
candidate = optimizer.ask()
candidate.value["log"]    # float
candidate.value["array"]  # np.ndarray
```

### ng.p.Instrumentation

Container for mixed positional + keyword parameters. Used when the function signature has multiple args:

```python
instru = ng.p.Instrumentation(
    ng.p.Scalar(),           # positional arg 0
    ng.p.Array(shape=(5,)),  # positional arg 1
    lr=ng.p.Log(0.001, 0.1), # keyword arg
)

# After ask():
candidate = optimizer.ask()
result = func(*candidate.args, **candidate.kwargs)
```

## set_mutation

```python
param.set_mutation(sigma=value)
```

- **`sigma`**: scalar float — sets initial step size for all elements of this parameter
- Controls how far CMA-ES explores from the current mean in this parameter's space
- Rule of thumb per `CMA-ES_Parameters.md`: `sigma = 0.3 * (upper - lower)`
- **Array sigma** (per-element): not confirmed in docs — use [[Nevergrad_Parametrization#ngpTuple|ng.p.Tuple]] with separate Array objects instead

## set_bounds

```python
param.set_bounds(lower, upper)
# lower, upper: scalar or array matching param shape
```

Clips or penalizes values outside `[lower, upper]`. The exact method (clipping vs penalty) depends on Nevergrad internals — per `CMA-ES_Practical_Concerns.md` (B.5), simple clipping can violate CMA's distributional assumptions. Use explicit boundary penalty in your fitness function when control is needed.

## Common Operations

```python
# Access current value
param.value                              # numpy array or structured value

# Create a child with a specific value (for information injection)
child = param.spawn_child(new_value=np.array([...]))

# Export to optimizer's internal standardized space
raw = param.get_standardized_data(reference=param)

# Mutate in-place (for testing)
param.mutate()

# Reproducibility
param.random_state.seed(42)
```

## Constraints via Parametrization

```python
# Applied at sampling time (fast, approximate):
param.register_cheap_constraint(lambda x: x[0] >= 1.0)
```

## Practical Pattern for lynx_mjspec

Split genome into tubes + weights with independent sigmas:

```python
from ariel.body_phenotypes.lynx_mjspec.unified_pipeline.common import TUBE_MIN, TUBE_MAX, NUM_TUBES

tubes_init   = np.full(NUM_TUBES, (TUBE_MIN + TUBE_MAX) / 2)   # 0.55
weights_init = rng.uniform(-0.1, 0.1, size=num_weights)

tubes_p   = ng.p.Array(init=tubes_init).set_mutation(sigma=0.27)
weights_p = ng.p.Array(init=weights_init).set_mutation(sigma=0.10)
param     = ng.p.Tuple(tubes_p, weights_p)

optimizer = ng.optimizers.CMA(parametrization=param, budget=budget, num_workers=workers)

# In ask/tell loop:
candidates = [optimizer.ask() for _ in range(population)]
genomes = [np.concatenate([c.value[0], c.value[1]]) for c in candidates]
```
