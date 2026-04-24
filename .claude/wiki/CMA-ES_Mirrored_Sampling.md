---
type: reference
tags: [cma-es, mirrored-sampling, tpa, sigma-adaptation, parallel, variance-reduction]
source: https://inria.hal.science/inria-00530202v2/document
paper: Brockhoff, Auger, Hansen, Arnold, Hohm (2010)
---

# CMA-ES: Mirrored Sampling and Two-Point Adaptation (TPA)

Technique introduced in [[Brockhoff_2010]] to reduce fitness evaluations via symmetric paired sampling and an alternative step-size adaptation rule compatible with batch evaluation.

See also: [[CMA-ES_Algorithm]] (base algorithm), [[CMA-ES_Parameters]] (default λ, σ), [[CMA-ES_Practical_Concerns]] (termination, flat fitness).

---

## What Mirrored Sampling Is

Standard CMA-ES draws λ offspring **independently** from N(**m**, σ²**C**). Each offspring is evaluated without any structured relationship to its siblings.

Mirrored sampling enforces that offspring are generated in **symmetric pairs** around the current mean **m**:

```
x⁺ = m + σ * B * D * z       (original)
x⁻ = m − σ * B * D * z       (mirror)
```

where `z ~ N(0, I)` is a single random vector drawn once per pair, and `B * D` is the square root of the current covariance matrix **C** (see [[CMA-ES_Algorithm]], eigendecomposition). The total offspring count λ remains the same; only λ/2 independent `z` vectors are drawn instead of λ.

**Key property:** The sample mean of a mirrored pair is exactly **m** regardless of `z`, so the expected recombination mean is unchanged. The variance of the weighted mean shift — which drives learning of both **m** and **C** — is reduced.

---

## Why It Helps

In standard sampling, by chance all λ draws may cluster on one side of the mean, producing a biased recombination step for that generation. Mirrored sampling prevents this: for every `+z` there is a `-z`, so the sample is always symmetric around **m**. This quasi-random structure reduces the number of generations (and therefore fitness evaluations) needed for the mean to track the true gradient direction.

The variance reduction is analogous to antithetic variates in Monte Carlo integration.

---

## Two-Point Adaptation (TPA)

TPA is a step-size adaptation rule designed for use with mirrored sampling. It replaces or supplements the standard Cumulative Step-size Adaptation (CSA) described in [[CMA-ES_Algorithm]] (Eq. 43–44).

### Core idea

Within each mirrored pair `(x⁺, x⁻)`, exactly one member receives a better fitness rank than the other (ties are broken arbitrarily). TPA uses this within-pair competition to judge whether the current σ is appropriate:

- If `x⁺` (the `+z` pole) consistently ranks better than `x⁻` across many pairs, the distribution is moving in the direction of `z` — the step size should increase.
- If pair members alternate winning randomly, the algorithm is already well-calibrated.
- If `x⁻` consistently wins, the mean is overshooting and σ should decrease.

TPA aggregates these binary "who won the pair?" signals across all λ/2 pairs in a generation to compute a step-size update.

### Advantages over CSA in batch settings

CSA accumulates an **evolution path** across generations (Eq. 43 in [[CMA-ES_Algorithm]]):
```
p_σ ← (1 − c_σ) * p_σ + sqrt(c_σ*(2−c_σ)*μ_eff) * C^{−1/2} * <y>_w
```
This requires memory of previous generations and assumes temporally consistent search. In asynchronous or restart scenarios the path can be stale.

TPA is **stateless** in this sense — it derives its signal entirely from within-generation pair comparisons. This makes it robust to restarts and compatible with batch evaluation where generation boundaries are well-defined.

### Requirement

TPA **only works with mirrored sampling**. Without paired candidates there is no within-pair comparison signal. Do not enable TPA without ensuring mirrored pairs are properly formed.

---

## Sequential Selection

Sequential selection (also introduced in [[Brockhoff_2010]]) evaluates and ranks candidates **one at a time** during a generation. As each new evaluation arrives, the running ranking is updated.

When combined with mirrored sampling:
- Evaluate `x⁺` first; observe its rank position.
- If `x⁺` already secures a top-μ rank position, `x⁻` may provide only marginal additional information.
- An early-stopping rule can skip some evaluations, saving budget.

**Note:** Sequential selection is incompatible with fully parallel batch evaluation. It applies only to serial or semi-serial workloads. For parallel robotics evaluation (e.g., batch MuJoCo rollouts), mirrored sampling without sequential selection is the relevant technique.

---

## Ask/Tell Interface and the `check_consistency` Warning

Nevergrad implements mirrored-sampling variants of CMA-ES. Because offspring must be generated and evaluated as pairs, the ask-tell loop has stricter requirements than standard CMA.

### Correct usage pattern

```python
# num_workers must equal lambda (population size) for batch synchronous mode
optimizer = ng.optimizers.CMA(
    parametrization=dim,
    budget=total_budget,
    num_workers=lambda_size   # must be even for mirrored sampling
)

# Ask exactly lambda_size times per generation
candidates = [optimizer.ask() for _ in range(lambda_size)]

# Evaluate (can be parallel — pairs are independent of each other)
losses = [evaluate(c) for c in candidates]

# Tell all results back
for cand, loss in zip(candidates, losses):
    optimizer.tell(cand, loss)
```

### What `check_consistency` means

Nevergrad fires a `check_consistency` warning when:
1. `ask()` is called an **odd number of times** before `tell()` completes the batch, so one pair member has no partner.
2. The number of `ask()` calls does not match the optimizer's expected batch size (λ).

**Effect:** The broken pair's mirror is missing. The TPA within-pair comparison cannot be computed for that pair. The variance-reduction guarantee is partially lost. The optimizer degrades gracefully (it still runs) but behaves more like standard CMA-ES for the affected pairs.

**Fix:** Ensure `num_workers` equals λ and always `ask()` exactly λ times per generation cycle. If using `batch_mode=False` in Nevergrad, always ask 2 candidates at a time to preserve pairs.

---

## Practical Guidance

| Scenario | Recommendation |
|---|---|
| Batch MuJoCo rollouts (all parallel) | Use mirrored CMA-ES with `num_workers=λ`; set `batch_mode=True` |
| Serial optimization | Use mirrored sampling + sequential selection |
| Asynchronous workers (results arrive out of order) | Standard CMA-ES without mirrored sampling is safer |
| Noisy fitness (e.g., stochastic simulators) | TPA signal is unreliable; revert to CSA-based step-size control |
| IPOP restart (see [[CMA-ES_Practical_Concerns]]) | TPA is restart-compatible; reset pair counters on restart |

### Population size with mirrored sampling

The default λ from [[CMA-ES_Parameters]] (`λ = 4 + floor(3 * ln(n))`) remains a good starting point. Mirrored sampling **improves efficiency at fixed λ** — it does not require halving λ. If you increase λ for a hard problem, keep it even so pairs divide cleanly.

---

## Summary Table

| Property | Standard CMA-ES | + Mirrored Sampling | + TPA |
|---|---|---|---|
| Offspring draw | λ i.i.d. | λ/2 pairs (z, −z) | same as mirrored |
| Step-size rule | CSA (evolution path) | CSA | Within-pair comparison |
| Stateless across generations | No (CSA path) | No | Yes |
| Parallel-batch compatible | Yes | Yes (batch size = λ) | Yes |
| `check_consistency` risk | No | Yes if λ not matched | Yes |
| Speedup (unimodal benchmarks) | baseline | ~20–50% fewer evals | additional |
