---
type: reference
tags: [nevergrad, machine-learning, optimizer-selection, reinforcement-learning, ask-tell, parallel]
source: https://facebookresearch.github.io/nevergrad/machinelearning.html
---

# Nevergrad: ML Optimization Patterns

## Optimizer Selection by Problem Type

| Problem | Recommended Optimizer(s) |
|---------|--------------------------|
| Continuous hyperparameters (moderate dim) | `CMA`, `PSO`, `TwoPointsDE` |
| Mixed continuous+discrete, good initial guess | `PortfolioDiscreteOnePlusOne` |
| Large-scale mixed | `TwoPointsDE` |
| Noisy / reinforcement learning (low noise) | `CMA` |
| Noisy / reinforcement learning (high noise) | `TBPSA`, `NaiveTBPSA`, `PortfolioNoisyDiscreteOnePlusOne` |
| Fully parallel (workers = budget) | `ScrHammersleySearch` |
| Baseline comparison | `RandomSearch`, `ScrHammersleySearch` |

> **RL note:** "We do not average evaluations over multiple episodes — the algorithm is in charge of averaging." Do not pre-average; pass raw noisy fitness to `tell()`.

> **TBPSA** = population-control mechanism, robust to overparameterized/noisy objectives. Relevant for neural network weight evolution with stochastic rollouts.

## Synchronous Parallel Ask/Tell Pattern

Ask `num_workers` candidates, evaluate all, then tell all — before asking again:

```python
optim = ng.optimizers.registry[name](
    parametrization=parametrization,
    budget=budget,
    num_workers=3,
)

for u in range(budget // 3):
    x1 = optim.ask()
    x2 = optim.ask()
    x3 = optim.ask()
    y1 = train_and_return_test_error(*x1.args, **x1.kwargs)
    y2 = train_and_return_test_error(*x2.args, **x2.kwargs)
    y3 = train_and_return_test_error(*x3.args, **x3.kwargs)
    optim.tell(x1, y1)
    optim.tell(x2, y2)
    optim.tell(x3, y3)

recommendation = optim.recommend()
```

## Asynchronous Pattern with Executor

```python
from concurrent import futures

optim = ng.optimizers.registry[name](parametrization=instru, budget=budget)
with futures.ThreadPoolExecutor(max_workers=optim.num_workers) as executor:
    recommendation = optim.minimize(train_and_return_test_error, executor=executor)
```

Use `ProcessPoolExecutor` instead of `ThreadPoolExecutor` for CPU-bound work (MuJoCo simulations, numpy-heavy fitness functions).

## Hyperparameter Optimization Example

```python
import nevergrad as ng

def myfunction(lr, num_layers, arg3, arg4, other):
    return -accuracy   # minimize negative accuracy

lr         = ng.p.Log(lower=0.0001, upper=1.0)
num_layers = ng.p.TransitionChoice([4, 5, 6])
parametrization = ng.p.Instrumentation(lr, num_layers, 3.0, arg4=4.0)

optim = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=100)
recommendation = optim.minimize(myfunction)
best_lr, best_layers, _, _ = recommendation.args
```

## Mixed Discrete+Continuous Example

```python
arg1  = ng.p.TransitionChoice(["a", "b"])           # ordered discrete
arg2  = ng.p.Choice(["a", "c", "e"])                 # unordered discrete
value = ng.p.Scalar(init=1.0).set_mutation(sigma=2)  # continuous

instru = ng.p.Instrumentation(arg1, arg2, "blublu", value=value)
```

Dimension count: `1 (TransitionChoice "a/b") + 3 (softmax for 3-way Choice) + 1 (Scalar) = 5 continuous dims` internally.

## High-Dimensional Continuous (300D)

```python
def train_and_return_test_error(x):
    return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in x])

parametrization = ng.p.Array(shape=(300,))
optim = ng.optimizers.CMA(parametrization=parametrization, budget=1200, num_workers=1)
```

Budget 1200 for 300D = 4 evaluations/dimension — **very tight**. Per `CMA-ES_Parameters.md`, CMA needs O(n²/μ_eff) generations for covariance adaptation. Use `TwoPointsDE` if budget is this constrained.

## Budget Guidelines

- "budget = 1200" shown for 300D continuous and mixed examples in docs — treat as a lower bound for exploration, not sufficient for full CMA-ES convergence
- `TwoPointsDE` "might need a budget in the hundreds"
- For RL/noisy: budget should account for variance — more evaluations needed per useful gradient signal

## Accessing Recommendation

```python
recommendation = optim.recommend()   # or optim.provide_recommendation()
result = train_and_return_test_error(*recommendation.args, **recommendation.kwargs)

# Or for Array parametrization:
best_genome = np.asarray(recommendation.value)
```

## Setting Initial Values

```python
# Make sure parametrization.value holds your initial guess before creating optimizer
param = ng.p.Array(init=my_initial_guess)
# It is automatically populated, but can be updated manually
param.value = my_updated_guess
```

## Lynx Application Notes

- Rollouts are stochastic (random network weights → variable distances) → treat as **noisy** objective → prefer `TBPSA` or `CMA` (Nevergrad handles noise internally)
- With 259 genome dims (hidden=8): use `TwoPointsDE` or increase budget substantially before switching to `CMA`
- With 65 genome dims (hidden=2): `CMA` is viable with budget ~5000+
- `num_workers` in Nevergrad must match actual parallel workers or CMA's internal state gets desynchronized
