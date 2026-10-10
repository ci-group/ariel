---
type: concept_reference
tags: [neat, neuroevolution, ga, algorithm, concept]
source: https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf
date_ingested: 2026-05-14
---

# neat_speciation

Speciation in NEAT groups individuals into species by genetic similarity so that new structural mutations are protected from elimination before their weights can be optimized. Each species competes only internally; the global population is partitioned into niches.

## Theory

### Compatibility distance

Two genomes `i` and `j` are assigned to the same species if their compatibility distance δ < δ_t:

```
δ = (c1 * E / N) + (c2 * D / N) + (c3 * W̄)

E   = number of excess genes (beyond the range of the shorter genome)
D   = number of disjoint genes (gaps within the shared range)
N   = number of genes in the larger genome (size normalization)
W̄  = mean weight difference of matching genes
c1, c2, c3 = importance coefficients (problem-dependent)
```

For small genomes (N < 20) the paper recommends N = 1 (no normalization).

### Fitness sharing (niche pressure)

Each individual's fitness is divided by the number of species-mates to prevent any one species from dominating offspring allocation:

```
f'_i = f_i / Σ_j sh(δ(i, j))

sh(δ) = 1  if δ < δ_t
sh(δ) = 0  otherwise
```

Species are allocated offspring in proportion to their **total adjusted fitness** (sum of `f'_i` within the species).

### Representative-based assignment

Each species keeps one **representative** genome (typically the previous generation's champion). New genomes are assigned to the first species whose representative is within δ_t; if none match, a new species is created.

## In Ariel

ARIEL's current CPPN evolution (`src/ariel/body_phenotypes/robogen_lite/cppn_neat/`, `examples/c_genotypes/5_body_evolution_cppn.py`) does **not** implement speciation. The `CPPNEvolution` class uses a flat tournament-style parent selection (top 50% by fitness) with no niche protection.

This means new structural mutations in ARIEL are not shielded from selection pressure — a known limitation. Adding speciation would require:
1. A compatibility distance function over `Genome` objects.
2. A species registry with representatives.
3. Per-species offspring quotas in `reproduction()`.

See [[NEAT_Algorithm]] for the crossover mechanics that speciation coordinates with.

## Practical Notes

- δ_t typically requires hand-tuning: too low → too many micro-species (slows convergence), too high → no protection for innovations.
- c1 = c2 = 1.0, c3 = 0.4 is a common starting point from the original paper; W̄ carries less signal than topology differences.
- Species with stagnating max fitness for N generations are typically culled (not part of the original paper formulation but standard in practice).
- Fitness sharing reduces effective selection pressure — compensate by raising raw population size or number of generations.

## See Also

- [[NEAT_Algorithm]] — full algorithm including crossover and mutation operators
- [[competing_conventions]] — the problem speciation + historical markings jointly solve
- [[cppn_neat_genome]] — ARIEL genome class where speciation would be added
