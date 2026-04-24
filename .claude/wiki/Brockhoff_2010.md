---
type: paper
tags: [cma-es, mirrored-sampling, tpa, sigma-adaptation, parallel, nevergrad]
source: https://inria.hal.science/inria-00530202v2/document
---

# Mirrored Sampling and Sequential Selection for Evolution Strategies

**Authors:** Dimo Brockhoff, Anne Auger, Nikolaus Hansen, Dirk V. Arnold, Tim Hohm
**Year:** 2010
**Identifier:** inria-00530202v2
**Venue:** HAL open archive (INRIA), Computer Science / Machine Learning [cs.LG]

---

## Core Contribution

The paper introduces **mirrored sampling** — generating candidate solutions in symmetric pairs around the population mean — and **Two-Point Adaptation (TPA)** for step-size control, demonstrating that these techniques reduce the number of fitness evaluations required for convergence on standard benchmarks by roughly 20–50% compared to baseline evolution strategies.

---

## Methodology

### Mirrored Sampling

Standard CMA-ES draws each offspring independently:
```
x_k = m + σ * z_k,   z_k ~ N(0, I)
```

With mirrored sampling, offspring are generated in **pairs**:
```
x⁺ = m + σ * z       (original)
x⁻ = m − σ * z       (mirror)
```
where the same random vector `z` is negated to form the mirror. For a population of size λ, only λ/2 independent random vectors are drawn; the other λ/2 are their negatives. Total evaluations per generation remain λ.

**Why it helps:** Mirrored pairs reduce the variance of the weighted recombination step. If `z` happens to point in a poor direction, its mirror `-z` partially compensates. This quasi-random technique enforces a symmetry constraint on the sample that standard i.i.d. sampling lacks.

**Ranking:** Both members of a pair are ranked independently by fitness value. There is no forced pairing in the selection step — whichever of `x⁺` or `x⁻` has better fitness simply ranks higher.

### Two-Point Adaptation (TPA)

TPA is a step-size adaptation mechanism that exploits mirrored pairs to replace or augment the standard Cumulative Step-size Adaptation (CSA, see [[CMA-ES_Algorithm]]).

The key idea: within a mirrored pair `(x⁺, x⁻)`, one can assess whether the search direction `z` was favorable by comparing the fitness ranks of `x⁺` and `x⁻`. If the better-ranked pair member appears consistently in the upper half of the population ranking, the step size should increase; if mirrored pairs split evenly across the ranking, σ is approximately correct.

Concretely, TPA tracks which "pole" of mirrored pairs (the `+z` or `-z` direction) tends to win, and adjusts σ using a two-point comparison rather than the evolution path length used by CSA. This makes TPA **compatible with ask-tell batch evaluation** because it requires no memory of consecutive generations — only the within-generation ranks of mirrored pairs.

### Sequential Selection

Sequential selection evaluates candidates **one at a time** and updates the running ranking after each evaluation. When combined with mirrored sampling, the algorithm can:
- Evaluate `x⁺`, observe its rank, then decide whether to evaluate `x⁻`
- Achieve early termination of a generation if enough high-quality candidates have been found

In a pure parallel/batch setting, sequential selection is not applicable (all evaluations happen simultaneously), but it provides additional efficiency savings in serial or semi-serial scenarios.

### Ask/Tell Implications and the `check_consistency` Warning

Mirrored sampling requires that `ask()` is called **in multiples of 2** (one call per pair member), and that the full batch of `2 * (λ/2) = λ` candidates is requested before any `tell()` call in batch mode.

In Nevergrad's implementation (see [[Nevergrad_Optimizers]]):

- The optimizer internally generates `z` vectors and registers them as pairs.
- If `ask()` is called an **odd number of times** (or a number not matching the expected batch size), the pairing structure is broken — one pole of some pair was evaluated without its mirror.
- Nevergrad fires a **`check_consistency` warning** in this situation.
- **What it means:** The algorithm received unpaired evaluations. It can still proceed, but the variance-reduction guarantee of mirrored sampling is violated for the incomplete pair, and TPA's within-pair comparison cannot be computed for that pair.
- **When it fires in practice:** If `num_workers` is not set to match the population size λ, or if `batch_mode=False` is used with an algorithm that expects synchronous batch delivery of mirrored pairs.

**Recommended pattern for mirrored-sampling optimizers in Nevergrad:**
```python
optimizer = ng.optimizers.CMA(parametrization=dim, budget=budget, num_workers=lambda_)
# Use batch_mode=True or ensure ask() is called exactly lambda_ times per generation
candidates = [optimizer.ask() for _ in range(lambda_)]
losses = parallel_evaluate(candidates)
for cand, loss in zip(candidates, losses):
    optimizer.tell(cand, loss)
```

---

## Key Findings

- Mirrored sampling consistently reduces function evaluations to reach a target precision across unimodal benchmark problems. Reported gains are in the range of **20–50% fewer evaluations** versus standard (μ/μ_W, λ)-CMA-ES.
- TPA is effective as a step-size adaptation scheme specifically when mirrored pairs are available; it is not recommended as a drop-in replacement for CSA without mirrored sampling.
- Sequential selection provides additional savings in serial evaluation scenarios, independently of mirrored sampling; the two techniques are **complementary and can be combined**.
- Benefits of mirrored sampling are most pronounced on **unimodal, moderately conditioned** problems. On highly multimodal or noisy problems, gains are smaller or absent.
- Mirrored sampling does **not** change the expected value of the sample mean; it only reduces its variance. The algorithm remains unbiased.

---

## Practical Recommendations

### When to use mirrored sampling

- Serial or parallel optimization where fitness evaluation is the dominant cost.
- Unimodal or weakly multimodal problems.
- Any setting where you can guarantee that `ask()` is called in even-numbered batches.

### When NOT to use it (or use with caution)

- Highly asymmetric fitness landscapes (where the symmetry assumption actively misleads).
- Noisy objectives where the within-pair fitness comparison used by TPA is unreliable.
- Asynchronous parallel settings where workers return results in arbitrary order — this breaks the pairing structure and triggers `check_consistency`.

### Population size effects

- With mirrored sampling the effective information per unique `z` vector doubles (both `+z` and `-z` are evaluated), so **one can use half as many independent directions** for the same information content.
- Default λ from [[CMA-ES_Parameters]] is still appropriate; the technique improves efficiency at fixed λ rather than requiring λ to be halved.
- For parallel compute farms: set `num_workers = λ` to evaluate exactly one full generation (λ paired candidates) per synchronisation step.

### Interaction with batch/parallel ask-tell

| Evaluation mode | Mirrored sampling compatible? | Notes |
|---|---|---|
| Serial (one-at-a-time) | Yes, with sequential selection | Maximum efficiency |
| Batch synchronous | Yes, if batch size = λ | Natural fit; set `num_workers=λ` |
| Asynchronous parallel | Partial | Pairs may be separated; triggers warning |
| `batch_mode=False` (Nevergrad) | Risky | Only safe if you always ask exactly 2 at a time |

---

## Limitations and Failure Modes

- **Asymmetric landscapes:** The variance-reduction argument assumes the fitness landscape is approximately symmetric around the mean. On strongly skewed or valley-like landscapes, forcing symmetric sampling may waste evaluations.
- **Noisy functions:** TPA's within-pair comparison relies on a meaningful fitness difference between `x⁺` and `x⁻`. High noise can make this signal unreliable and degrade step-size adaptation.
- **`check_consistency` warning:** Signals broken pair structure in the ask-tell interface. Does not crash the optimizer but degrades the guarantees of mirrored sampling and TPA.
- **Not a universal substitute for larger λ:** Mirrored sampling improves efficiency at a given λ but cannot substitute for the diversity benefits of genuinely larger populations on multimodal problems.

---

## Mathematical Summary

Standard ES offspring:
```
x_k = m + σ * z_k,   z_k ~ N(0, I),   k = 1, ..., λ
```

Mirrored ES offspring (λ must be even):
```
x_{2i-1} = m + σ * z_i
x_{2i}   = m − σ * z_i
for i = 1, ..., λ/2,   z_i ~ N(0, I)
```

The weighted recombination mean shift (see [[CMA-ES_Algorithm]], Eq. 41–42) computed from mirrored samples has lower variance than from i.i.d. samples, which accelerates adaptation of both `m` and `C`.

---

## Cross-references

- [[CMA-ES_Algorithm]] — full (μ/μ_W, λ)-CMA-ES algorithm, sampling equations
- [[CMA-ES_Parameters]] — default λ, μ, weights; population size guidance
- [[CMA-ES_Practical_Concerns]] — flat fitness, termination, initialization
- [[CMA-ES_Mirrored_Sampling]] — dedicated theory page for mirrored sampling and TPA
- [[Nevergrad_Optimizers]] — ask/tell interface, `num_workers`, `batch_mode`
- [[Source - CMA-ES Tutorial (Hansen 2023)]] — parent tutorial for the base algorithm
