---
type: reference
tags: [cma-es, initialization, termination, boundaries, flat-fitness, constraints]
source: https://arxiv.org/pdf/1604.00772 (Sections B.1–B.5, pp. 32–35)
---

# CMA-ES: Practical Concerns

Implementation details from Appendix B of Hansen (2023).

## B.1 Sampling from N(m, σ²C)

Use eigendecomposition **C = B D² Bᵀ**:

```python
z = randn(n)          # N(0, I)
y = B @ D @ z         # N(0, C)
x = m + sigma * y     # N(m, sigma^2 * C)

# For C^{-1/2} * <y>_w  (needed in step-size path update):
# C^{-1/2} * y  =  B @ D^{-1} @ (Bᵀ @ y)  =  B @ sum_i(w_i * z_{i:λ})
```

**Do NOT use Cholesky** — the step-size path update (Eq. 43) requires **C⁻¹/²** = **B D⁻¹ Bᵀ**, which cannot be computed from a Cholesky factor alone.

## B.2 Eigendecomposition Frequency

Re-decompose **B**, **D** only every `max(1, floor(1 / (10*n*(c_1+c_μ))))` generations.

- For typical c_1 + c_μ ≈ 1/n, this is every ~n/10 generations
- Reduces cost from O(n³) to O(n²) per generated search point
- Enforce symmetry before decomposition: `C = triu(C) + triu(C,1)ᵀ`

## B.3 Termination Criteria

Stop when any of these trigger (from cmaes.m defaults):

| Criterion | Condition | Meaning |
|-----------|-----------|---------|
| **NoEffectAxis** | Adding 0.1σ·d_ii·b_i to m doesn't change m | Search range collapsed in an eigenvector direction |
| **NoEffectCoord** | Adding 0.2σ·√c_{ii} to m_i doesn't change m_i | Numerical underflow in a coordinate |
| **ConditionCov** | cond(**C**) > 10¹⁴ | Covariance matrix ill-conditioned; algorithm broken |
| **EqualFunValues** | Range of best f-values in last 10+⌊30n/λ⌋ gens = 0 | Flat fitness plateau |
| **Stagnation** | Median of recent 30% not better than first 30%, over min(120+30n/λ, 20000) iters | No progress |
| **TolXUp** | σ·max(diag(**D**)) increased by > 10⁴ | σ diverging — initial σ too small, or fitness diverging |
| **TolFun** | Range of best f-values < TolFun, all recent < TolFun | Converged to precision TolFun |
| **TolX** | σ·p_c and σ smaller than TolX (default = 10⁻¹²·σ⁽⁰⁾) | Converged to precision TolX |

**TolXUp** is the most important for catching divergence. If σ explodes, it usually means σ⁽⁰⁾ was too small or the fitness is ill-posed (e.g., all evaluations return the same penalty value).

After triggering termination: **restart** (eventually with increased λ, see IPOP [ref 3]) or reconsider the objective function encoding.

## B.4 Flat Fitness

If multiple individuals share the same f-value, CMA-ES can stall:

```matlab
% Escape flat fitness (lines 92-96 from MATLAB source):
if arfitness(1) == arfitness(ceil(0.7 * lambda))
    sigma = sigma * exp(0.2 + cs/damps);
    disp('warning: flat fitness, consider reformulating the objective');
end
```

**Flat fitness is a symptom** — it means the objective function has plateaus (e.g., a binary touched/not-touched signal). The paper says: "observation of a flat fitness should be rather a termination criterion and consequently lead to a reconsideration of the objective function formulation."

Relevance for lynx_mjspec: the `999.0` invalid fitness and the binary touch bonus/penalty create flat regions. CMA-ES cannot differentiate between candidates that all fail to touch the target.

## B.5 Boundary and Constraint Handling

### Case 1: Optimal solution is NOT near boundary

Two simple approaches:

**Option A — Penalize and re-map:**
```python
f_fitness(x) = f_max + ||x - x_feasible||
```
where `f_max` > worst feasible fitness, `x_feasible` is any constant feasible point (e.g., center of domain).

**Option B — Resample:**
Reject and resample infeasible x until feasible. Works if feasible fraction is not too small.

### Case 2: Repair available (e.g., box-clipping)

**Avoid simple repair** — clipping x to box bounds before evaluation violates CMA-ES's distributional assumptions and can cause step-size divergence.

**Penalization approach (recommended):**
```python
x_repaired = clip(x, lb, ub)          # closest feasible point
f_fitness(x) = f(x_repaired) + alpha * ||x - x_repaired||²    # [Eq. 63]
```
The repaired solution is discarded; only the fitness is used. Choose α so that the f term and the penalty term have similar magnitude.

### Case 3: No repair available (general constraints c_i(x) ≤ 0)

```python
f_fitness(x) = f_offset + alpha * sum_i [ c_i(x) > 0 ] * c_i(x)²    # [Eq. 64]
```
where `f_offset = median_k f(x_k)` over feasible points in same generation.

## B.2 (continued) — Initialization

```python
# If search interval is [a, b] per coordinate:
sigma_0 = 0.3 * (b - a)
m_0 = uniform(a, b)   # random in search interval

# If coordinates have different scales Δs_i:
C_0 = diag([Δs_i²])   # NOT C = I
# Note: Δs_i should not differ by orders of magnitude — rescale variables if so
```

The fundamental requirement: **the optimum must lie within m⁽⁰⁾ ± 3σ⁽⁰⁾** in every coordinate.

## MATLAB Reference Implementation (Appendix C, simplified)

```matlab
% --- Strategy parameters ---
N = 10;                          % problem dimension
lambda = 4 + floor(3*log(N));    % population size
mu = lambda/2;
weights = log(mu+1/2)-log(1:mu); % weights
mu = floor(mu);
weights = weights/sum(weights);
mueff = sum(weights)^2/sum(weights.^2);

cc = (4+mueff/N)/(N+4+2*mueff/N);
cs = (mueff+2)/(N+mueff+5);
c1 = 2/((N+1.3)^2+mueff);
cmu = min(1-c1, 2*(mueff-2+1/mueff)/((N+2)^2+mueff));
damps = 1 + 2*max(0,sqrt((mueff-1)/(N+1))-1) + cs;

% --- Initialization ---
pc = zeros(N,1); ps = zeros(N,1);
B = eye(N); D = eye(N); C = B*D*(B*D)';
chiN = N^0.5*(1-1/(4*N)+1/(21*N^2));

% --- Generation loop ---
% Sample
arz(:,k) = randn(N,1);
arx(:,k) = xmean + sigma * (B*D * arz(:,k));   % Eq. 40

% Recombine
xmean = arx(:,arindex(1:mu))*weights;           % Eq. 42
zmean = arz(:,arindex(1:mu))*weights;

% Step-size
ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (B*zmean);    % Eq. 43
sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));    % Eq. 44

% Covariance
pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (B*D*zmean);   % Eq. 45
C = (1-c1-cmu) * C ...
    + c1 * (pc*pc' + (1-hsig) * cc*(2-cc) * C) ...             % rank-one
    + cmu * (B*D*arz(:,arindex(1:mu))) ...
           * diag(weights) * (B*D*arz(:,arindex(1:mu)))';       % rank-μ   Eq. 47

% Re-decompose (not every generation)
[B,D] = eig(C);
D = diag(sqrt(diag(D)));
```
