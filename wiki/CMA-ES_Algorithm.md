---
type: algorithm_reference
tags: [cma-es, evolution-strategy, optimization, covariance-matrix]
source: https://arxiv.org/pdf/1604.00772
---

# CMA-ES: Core Algorithm

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a stochastic black-box optimizer for real-parameter, non-linear, non-convex functions. It maintains a multivariate normal distribution N(**m**, σ²**C**) and adapts all three parameters — mean **m**, step-size σ, and covariance **C** — from ranked fitness evaluations only.

## Search Distribution

At generation g, offspring are sampled from:

```
x_k^(g+1)  ~  m^(g) + σ^(g) * N(0, C^(g))      for k = 1, ..., λ    [Eq. 5]
```

Equivalently via eigendecomposition C = B D² Bᵀ:

```
z_k ~ N(0, I)
y_k = B * D * z_k    ~  N(0, C)
x_k = m + σ * y_k   ~  N(m, σ²C)
```

- **B**: orthonormal eigenvectors of **C** (columns)
- **D**: diagonal, d_i = sqrt(eigenvalue_i) = standard deviation along eigenvector i
- **C⁻¹/² = B D⁻¹ Bᵀ** (needed for step-size path update)

## Complete Algorithm (μ/μ_W, λ)-CMA-ES [Figure 6 / Appendix A]

### Initialization
```
p_σ = 0,  p_c = 0,  C = I,  g = 0
Choose m ∈ Rⁿ (problem-dependent)
Choose σ ∈ R_{>0} (problem-dependent, see [[CMA-ES_Practical_Concerns]])
```

### Per-generation loop (until termination)

**1. Sample λ offspring**
```python
z_k ~ N(0, I)                                           # [38]
y_k = B * D * z_k          # ~ N(0, C)                  # [39]
x_k = m + sigma * y_k      # ~ N(m, sigma^2 * C)        # [40]
```

**2. Evaluate and rank** — sort x_1,...,x_λ by f(x) ascending (minimization).  
x_{1:λ} is the best, x_{λ:λ} is the worst.

**3. Selection and recombination (move mean)**
```
<y>_w = sum_{i=1}^{mu} w_i * y_{i:lambda}              # [41]
m  ←  m + c_m * sigma * <y>_w                          # [42]
   (= sum_{i=1}^{mu} w_i * x_{i:lambda}  when c_m = 1)
```

**4. Step-size control (CSA — Cumulative Step-size Adaptation)**
```
p_σ ← (1 - c_σ) * p_σ  +  sqrt(c_σ*(2-c_σ)*μ_eff) * C^{-1/2} * <y>_w   # [43]

σ   ← σ * exp( c_σ/d_σ * ( ||p_σ|| / E||N(0,I)|| - 1 ) )                  # [44]
```

where `E||N(0,I)|| = sqrt(2) * Γ((n+1)/2) / Γ(n/2) ≈ sqrt(n) * (1 - 1/(4n) + 1/(21n²))`

- **||p_σ|| > E||N(0,I)||** → steps are correlated → increase σ
- **||p_σ|| < E||N(0,I)||** → steps are anti-correlated → decrease σ

**5. Covariance matrix adaptation**

Evolution path for rank-one update:
```
h_σ = 1  if  ||p_σ|| / sqrt(1-(1-c_σ)^{2(g+1)}) < (1.4 + 2/(n+1)) * E||N(0,I)||
      0  otherwise   (Heaviside — stalls p_c when σ is far too small)

p_c ← (1-c_c) * p_c  +  h_σ * sqrt(c_c*(2-c_c)*μ_eff) * <y>_w           # [45]
```

Adjusted weights for negative entries (active CMA):
```
w_i° = w_i * (1  if w_i >= 0  else  n / ||C^{-1/2} * y_{i:λ}||²)        # [46]
```

Full covariance update (rank-one + rank-μ):
```
C ← (1 + c_1*δ(h_σ) - c_1 - c_μ*Σw_j) * C
    + c_1 * p_c * p_cᵀ                           # rank-one update
    + c_μ * sum_{i=1}^{λ} w_i° * y_{i:λ} * y_{i:λ}ᵀ   # rank-μ update   # [47]
```

where `δ(h_σ) = (1-h_σ)*c_c*(2-c_c)` ≤ 1 (minor correction term, usually ~0).

**6. Re-decompose C** (not every generation — see [[CMA-ES_Practical_Concerns]] B.2)
```
C = B D² Bᵀ   (eigendecomposition)
```

## What Each Update Does

| Update | Purpose |
|--------|---------|
| Mean **m** | Move toward best offspring (gradient-like) |
| Step-size σ (CSA) | Globally scale exploration; increase if steps correlated, decrease if anti-correlated |
| Rank-μ covariance | Use population-level shape info from current generation |
| Rank-one (evolution path) | Use sign info from consecutive steps across generations |

## Objective

The covariance matrix adaptation approximates the **inverse Hessian** of the objective function. On convex-quadratic functions, C = H⁻¹ reduces the search to an isotropic sphere problem.

## Key Invariances

- Invariance to order-preserving (monotonic) transformations of fitness — only **ranking** is used
- Invariance to rotation, reflection, and translation of search space
- Scale invariance (if σ⁽⁰⁾ and **m**⁽⁰⁾ chosen accordingly)
- Affine invariance (if **C**⁽⁰⁾ = **A**⁻¹(**A**⁻¹)ᵀ and **m**⁽⁰⁾ transformed accordingly)

## When CMA-ES Beats Classical Methods

- Non-convex or rugged landscapes (sharp bends, discontinuities, local optima)
- Ill-conditioned problems (high condition number of Hessian)
- Non-separable problems (correlated variables)
- When gradient information is unavailable or unreliable
