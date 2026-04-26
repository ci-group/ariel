---
title: "NEAT Speciation and Historical Markings"
authors: "Stanley, Miikkulainen"
year: 2002
source: "https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf"
tags: [neat, speciation, innovation-numbers, crossover, neuroevolution, evolutionary-computation]
---

# NEAT Speciation and Historical Markings

Detailed notes on the two structural mechanisms that make NEAT's topology evolution tractable. Based on [[stanley_2002]].

---

## The Competing Conventions Problem

When two networks encode the same function using different topologies, naive crossover produces broken offspring — genes that are functionally unrelated get swapped. This is the **competing conventions problem** that prevented earlier topology-evolving approaches from using crossover effectively.

NEAT's innovation numbers solve this by providing a **shared historical coordinate system** across the entire population.

---

## Innovation Numbers in Detail

A global counter tracks every distinct structural mutation ever performed in a run.

**Add-connection mutation:**
```
new connection (A → B) → gets innovation_number = global_counter++
```

**Add-node mutation** (splits connection A→B):
```
1. Disable connection A→B
2. Add node C
3. New connection A→C → innovation_number = global_counter++
4. New connection C→B → innovation_number = global_counter++
```

**Key rule:** If the same structural mutation (same in-node, same out-node) occurs **more than once within the same generation**, all instances get the **same innovation number**. This prevents the counter from exploding and ensures convergent innovations are correctly identified as the same gene.

Within a generation, a lookup table checks: "has this (in_node, out_node) pair already been mutated this generation?" If yes, reuse the existing innovation number.

---

## Compatibility Distance

$$\delta = \frac{c_1 \cdot E}{N} + \frac{c_2 \cdot D}{N} + c_3 \cdot \bar{W}$$

- **E (Excess genes):** Connection genes whose innovation numbers lie *beyond* the range of the other genome (they're newer than the other genome's most recent gene).
- **D (Disjoint genes):** Connection genes whose innovation numbers fall *within* the range of the other genome but are not present in it (innovations the other lineage missed).
- **N:** Max gene count between the two genomes — normalises so large genomes aren't automatically far from everything.
- **W̄:** Mean |weight difference| across matching genes (same innovation number in both).

**Typical values** from the paper: c1=1.0, c2=1.0, c3=0.4 with δ_t=3.0 (problem-dependent).

For small genomes (N < 20), the paper drops the N normalisation.

---

## Species Assignment

Each generation:
1. Pick a random **representative** from each existing species (from the previous generation).
2. For each new organism, compute δ against each representative in order.
3. If δ < δ_t for some representative, assign organism to that species.
4. If no match found, create a new species with this organism as its representative.

Species membership is **re-evaluated every generation**. Species without offspring go extinct.

---

## Adjusted Fitness and Offspring Allocation

Within-species fitness sharing prevents a single species from dominating:

$$f'_i = \frac{f_i}{\text{species\_size}}$$

(This is a simplification; the full formula sums sh(δ(i,j)) over all j, but since sh=1 within species and 0 outside, it reduces to dividing by species size.)

**Offspring per species** = (species total adjusted fitness) / (population total adjusted fitness) × population\_size

The **best-performing organism** in each species is copied unchanged into the next generation (**elitism within species**).

Low-performing members of a species are culled before reproduction (e.g. bottom 20%).

---

## Effect of Speciation on Search

Without speciation, novel topologies (with unoptimised weights) are immediately outcompeted by incumbents. Speciation provides a **grace period**:

1. A new structural mutation creates an organism in a new species (or small species).
2. That species gets offspring proportional to its (currently modest) adjusted fitness.
3. Over generations, if the topology is useful, its fitness grows and the species expands.
4. If it's not useful, the species dwindles and goes extinct.

This is analogous to **ecological niching** in biology — different species occupy different fitness peaks without direct competition during early exploration.

---

## Implementation Notes for ARIEL

In ARIEL's `cppn_neat` module:
- Innovation numbers are tracked via `id_manager` (passed into `Genome.random()` and `genome.mutate()`).
- The `next_innov_id` counter must be **shared across the entire population** — do not instantiate per-individual.
- Species assignment logic is not built into `Genome` — it must be implemented at the population/EA level.

See [[cppn_neat_genome]] for the genome API and [[stanley_2002]] for the full algorithmic description.
