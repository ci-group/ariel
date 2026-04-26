---
type: reference
tags: [cma-es, parameters, defaults, tuning]
source: https://arxiv.org/pdf/1604.00772 (Table 1, Appendix A, pp. 28-31)
---

# CMA-ES Default Parameters

All default parameters from Table 1 (Hansen 2023), where n = search space dimension.

> **Note from paper:** "We do not recommend changing this setting, apart from increasing the population size λ and possibly decreasing α_cov on noisy functions."

## Selection and Recombination

```
λ  = 4 + floor(3 * ln(n))        # population size (offspring)       [48]
μ  = floor(λ / 2)                 # parents (number of selected)

# Preliminary weights (convex shape):
w_i' = ln((λ+1)/2) - ln(i)       for i = 1, ..., λ                   [49]

# Final weights (positive sum to 1, negative scaled for active CMA):
w_i = w_i' / Σ|w_j'|+   if w_i' >= 0    (positive weights sum to 1)
    = min(α_μ⁻, α_μeff⁻, α_posdef⁻) / Σ|w_j'|⁻ * w_i'  if w_i' < 0  [53]

μ_eff = (Σ_{i=1}^{μ} w_i)² / Σ_{i=1}^{μ} w_i²    # variance-effective selection mass [8]
# Rule of thumb: μ_eff ≈ λ/4

c_m = 1                           # learning rate for the mean        [54]
```

## Step-Size Control

```
c_σ = (μ_eff + 2) / (n + μ_eff + 5)                                   [55]
d_σ = 1 + 2*max(0, sqrt((μ_eff - 1)/(n + 1)) - 1) + c_σ              [55]
```

- `1/c_σ` is the backward time horizon of the conjugate evolution path p_σ
- `d_σ ≈ 1` is the damping — controls how fast σ changes

## Covariance Matrix Adaptation

```
c_c = (4 + μ_eff/n) / (n + 4 + 2*μ_eff/n)           # cumulation decay [56]

c_1 = α_cov / ((n + 1.3)² + μ_eff)    with α_cov = 2  # rank-one rate  [57]

c_μ = min(1 - c_1,
          α_cov * (1/4*μ_eff + 1/μ_eff - 2) / ((n+2)² + α_cov*μ_eff/2))
      with α_cov = 2                                   # rank-μ rate     [58]
```

- `c_μ ≈ μ_eff / n²` is a reasonable first approximation
- `1/c_μ` is the backward time horizon for rank-μ update (characteristic time ≈ n²/μ_eff)
- `c_c ≈ 1/n`; time horizon for evolution path is between √n and n

## Negative Weights Bounds

```
α_μ⁻     = 1 + c_1/c_μ                                               [50]
α_μeff⁻  = 1 + 2*μ_eff⁻ / (μ_eff + 2)                               [51]
α_posdef⁻ = (1 - c_1 - c_μ) / (n * c_μ)                             [52]
```

## Quick Reference: Typical Values

| n | λ (default) | μ | μ_eff (approx) | c_σ | d_σ | c_c | c_1 | c_μ |
|---|-------------|---|----------------|-----|-----|-----|-----|-----|
| 5 | 8 | 4 | ~2.2 | ~0.44 | ~1.44 | ~0.44 | ~0.17 | ~0.16 |
| 10 | 10 | 5 | ~2.8 | ~0.37 | ~1.37 | ~0.31 | ~0.07 | ~0.07 |
| 20 | 13 | 6 | ~3.2 | ~0.29 | ~1.29 | ~0.22 | ~0.03 | ~0.04 |
| 80 | 17 | 8 | ~4.2 | ~0.21 | ~1.21 | ~0.12 | ~0.003 | ~0.007 |

> For n≈80 (5 tubes + ~75 weights): **λ_default = 4 + floor(3·ln(80)) ≈ 17**. Using λ=24 (as in evolve.py) is fine — larger λ improves global search.

## Population Size Guidance

- **Increasing λ**: improves global search capability and robustness, at the price of slower convergence per function evaluation
- **Decreasing λ**: "not recommended — too small values have strong adverse effects"
- **Restarts with increasing λ** (IPOP/BIPOP): useful policy for multimodal problems
- Convergence rate per f-evaluation is roughly independent of λ when λ/μ_eff ≈ 4

## Sigma Initialization Rule

```
σ⁽⁰⁾ = 0.3 * (b - a)    if search interval is [a, b]ⁿ
```

- The optimum should lie within **m⁽⁰⁾ ± 3σ⁽⁰⁾** in every coordinate
- If different coordinates have different search ranges Δs_i, initialize **C** diagonally: `c_{ii} = (Δs_i)²`
- The Δs_i should **not differ by several orders of magnitude** — if they do, rescale variables

### Example for lynx_mjspec

Genome mixes tube lengths [0.1, 1.0] and network weights [~-1, ~1]:
- Tube range: Δs_tube = 0.9 → σ_tube ≈ 0.27
- Weight range: Δs_weight ≈ 2.0 → σ_weight ≈ 0.6
- Using a single σ=0.1 with uniform C=I is suboptimal; consider separate scaling or diagonal C initialization

## What NOT to Change

Per Hansen: the default parameters (53)–(58) "are in particular chosen to be a robust setting and therefore, to our experience, applicable to a wide range of functions to be optimized."

Only tune:
1. **λ** (increase for harder/multimodal problems)
2. **α_cov** (decrease slightly on noisy functions)
3. **σ⁽⁰⁾** and **m⁽⁰⁾** (always problem-specific)
