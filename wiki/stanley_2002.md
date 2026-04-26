---
title: "Evolving Neural Networks through Augmenting Topologies"
authors: "Stanley, Miikkulainen"
year: 2002
source: "https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf"
tags: [neat, neuroevolution, topology, speciation, genetic-algorithms, evolutionary-computation, cppn]
---

# Evolving Neural Networks through Augmenting Topologies (NEAT)

**Stanley, K.O. & Miikkulainen, R. (2002)**
*Evolutionary Computation, Vol. 10, No. 2*

This is the foundational paper for NEAT — the algorithm underlying [[cppn_neat_genome]] in ARIEL. It introduces three key innovations that together make topology-evolving neuroevolution tractable: historical markings, speciation, and complexification from minimal initial topologies.

---

## Core Problem

Prior neuroevolution methods either (a) fixed the topology and only evolved weights (losing expressivity), or (b) evolved topology but couldn't do meaningful crossover between differently-shaped networks (the **competing conventions problem**).

NEAT solves all three simultaneously.

---

## Genome Encoding

Every individual is a **genome** consisting of two gene lists:

### Node genes
```
node_id | type (input / hidden / output)
```

### Connection genes
```
in_node | out_node | weight | enabled | innovation_number
```

The `enabled` flag allows connections to be "silenced" without deletion (important after add-node mutations).

---

## Innovation Numbers (Historical Markings)

Every structural mutation (add-node, add-connection) receives a **global innovation number** from a shared counter. This number is permanent and unique to that structural event.

- If the same structural mutation occurs in two individuals in the same generation, both get the **same** innovation number.
- This creates a chronological historical record embedded in the genome.

**Why it matters:** Innovation numbers enable meaningful crossover between differently-shaped topologies — align genes by innovation number rather than by position.

---

## Crossover

Given two parents A and B:

1. **Align** connection genes by innovation number.
2. **Matching genes** (same innovation number in both): randomly inherit from either parent.
3. **Disjoint genes** (gap within the range of the other parent): inherit from the **fitter** parent.
4. **Excess genes** (beyond the range of the other parent): inherit from the **fitter** parent.

When parents have equal fitness, disjoint/excess genes are inherited from both (randomly chosen).

This resolves the competing conventions problem without requiring a bijective mapping between network positions.

---

## Speciation

New topological innovations start at a disadvantage (no refined weights yet). Without protection, they are eliminated before they can be optimised.

**Solution:** Group organisms into **species** by genetic similarity. Organisms only compete within their species.

### Compatibility distance

$$\delta = \frac{c_1 \cdot E}{N} + \frac{c_2 \cdot D}{N} + c_3 \cdot \bar{W}$$

| Symbol | Meaning |
|--------|---------|
| E | Number of excess genes |
| D | Number of disjoint genes |
| N | Number of genes in the larger genome (normalises for size) |
| W̄ | Average weight difference of **matching** genes |
| c1, c2, c3 | Tunable importance coefficients |

If δ < δ_t (threshold), two organisms belong to the same species.

### Fitness sharing (niche pressure)

Adjusted fitness of organism i:

$$f'_i = \frac{f_i}{\sum_{j=1}^{n} \text{sh}(\delta(i, j))}$$

where sh(δ) = 1 if δ < δ_t, else 0. This divides fitness by the number of organisms in the same species, preventing any one species from taking over the population.

Each species is **allocated offspring proportional to its total adjusted fitness**.

---

## Mutation Operators

| Mutation | Description | Structural? |
|----------|-------------|-------------|
| Weight perturbation | Add Gaussian noise to existing weights | No |
| Weight replacement | Replace weight with new random value | No |
| Add connection | New edge between two previously unconnected nodes — gets a new innovation number | Yes |
| Add node | Split an existing connection: disable it, insert a new node, add two new connections (weight-1 in, original-weight out) | Yes |

The add-node mutation uses **two** new innovation numbers (one per new connection).

---

## Complexification from Minimal Initial Topology

All organisms start with the **same minimal topology**: input nodes connected directly to output nodes, **no hidden nodes**. Complexity emerges only through mutation.

**Why:** Minimal starting topologies mean the search begins in a low-dimensional weight space where early fitness evaluation is cheap. Complexity is added only when it is useful.

---

## Evaluation Benchmarks

### XOR problem
- Requires non-linear separation (cannot be solved by direct I→O weights)
- NEAT reliably evolves minimal 2-hidden-node solutions

### Double pole-balancing without velocity information
- A control task where the agent cannot directly observe velocity (must infer from history — requires recurrent connections or memory)
- NEAT solved this significantly faster than competing algorithms (SANE, ESP)
- Final networks were minimal, interpretable topologies

### Key metrics reported
- Generations to solution (lower = better)
- Network complexity at solution (number of hidden nodes/connections)
- Success rate across multiple runs

---

## Comparison to Baselines

| Method | Double pole (no vel.) | Topology evolved? | Crossover meaningful? |
|--------|----------------------|-------------------|-----------------------|
| SANE | Often fails | No | N/A |
| ESP | Often fails | No | N/A |
| Fixed-topology NE | N/A | No | Yes |
| **NEAT** | Reliably solves | **Yes** | **Yes** |

---

## Theoretical Contributions

1. **Historical markings solve competing conventions** without requiring a topology-normalisation step before crossover.
2. **Speciation provides temporal protection** for structural innovations long enough for weight optimisation to catch up.
3. **Incremental complexification** is sufficient to discover minimal network solutions — you do not need to search over a large topology space up front.

---

## Limitations

- Speciation parameters (δ_t, c1, c2, c3) require tuning per problem.
- Scalability to very deep or very wide networks was not demonstrated.
- The population-level innovation counter can cause collisions if parallelised naively.
- Recurrent topologies complicate activation evaluation (requires iterative relaxation).

---

## Relation to ARIEL / CPPN-NEAT

ARIEL uses this exact algorithm to evolve **CPPN genomes** (where the network encodes spatial morphology patterns rather than control policies). See [[cppn_neat_genome]] for ARIEL-specific API details and [[neat_speciation]] for deeper speciation notes.

The key adaptation: CPPN outputs are **decoded into robot morphologies** (via `MorphologyDecoderBestFirst`) rather than used as controllers.
