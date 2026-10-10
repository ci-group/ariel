---
type: concept_reference
tags: [neat, neuroevolution, ga, concept, algorithm]
source: https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf
date_ingested: 2026-05-14
---

# competing_conventions

The competing conventions problem occurs when crossover is applied to two genomes that encode the same function via *different* network topologies or node orderings. Recombining them produces offspring with contradictory or disrupted weight assignments, making crossover harmful rather than beneficial.

## Theory

In a fixed-topology network, position-based crossover works because node i in parent A corresponds to node i in parent B. When topology evolves, two networks may use the same nodes in different roles — there is no consistent positional correspondence. Crossing them at the weight level mixes incompatible representations.

Example: parent A uses node 3 as a hidden feature detector; parent B uses node 3 as a pass-through. Their offspring inherits node 3's weights from both parents, destroying both parents' solutions.

This is structurally analogous to the permutation invariance problem in neural network weight space.

## In Ariel

[[NEAT_Algorithm]] resolves this via **innovation numbers**: each structural gene (connection or node) carries a globally unique ID assigned at creation time. Crossover aligns genes by innovation number rather than by position, so matching genes always correspond to the same historical structural event.

Two genes with the same innovation number represent the same structural addition (same edge between the same two conceptual roles), regardless of what other mutations have occurred in each lineage. Genes with no matching partner (disjoint/excess) are inherited from the fitter parent, avoiding destructive mixing from non-corresponding genes.

This is implemented in `Genome.crossover()` at `src/ariel/body_phenotypes/robogen_lite/cppn_neat/genome.py`.

## Practical Notes

- Competing conventions are only a problem when crossover is used *and* topology is variable. If you only mutate (no crossover), the problem does not arise.
- Even with innovation numbers, crossover between very dissimilar topologies (large δ) tends to produce unfit offspring. [[neat_speciation]] addresses this by restricting crossover to individuals within the same species.
- Alternative: abandon crossover entirely for topology-evolving systems and rely on mutation only. Faster per-generation but slower to combine building blocks across lineages.

## See Also

- [[NEAT_Algorithm]] — how innovation numbers resolve this problem
- [[neat_speciation]] — species restrict crossover to genetically similar individuals, reducing competing-convention disruption
