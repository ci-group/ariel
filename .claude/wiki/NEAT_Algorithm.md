---
type: algorithm_reference
tags: [neat, neuroevolution, cppn, ga, algorithm]
source: https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf
date_ingested: 2026-05-14
---

# NEAT_Algorithm

NEAT (NeuroEvolution of Augmenting Topologies) simultaneously evolves the weights *and* topology of neural networks using a genetic algorithm with historical gene markings, speciation, and incremental complexification. It is the algorithm underlying ARIEL's CPPN genome evolution (`src/ariel/body_phenotypes/robogen_lite/cppn_neat/`).

## Formulation

### Genome encoding

Two gene lists per individual:

```
Node gene:   node_id | type ∈ {input, hidden, output}

Connection:  in_node | out_node | weight | enabled | innovation_number
```

`enabled=False` silences a connection without removing it (used after add-node mutations).

### Innovation numbers (historical markings)

Every structural mutation receives a globally unique, permanent **innovation number** from a shared counter. If the same structural event occurs in two individuals within one generation, both receive the **same** number.

Innovation numbers align heterogeneous topologies for crossover without needing explicit topology normalization.

### Crossover (align by innovation number)

```
For each innovation ID in union(parent_A, parent_B):
  matching gene  → inherit randomly from A or B
  disjoint gene  → inherit from fitter parent
  excess gene    → inherit from fitter parent
  (equal fitness → inherit from both, randomly)
```

### Structural mutations

| Mutation | Action | New innovation IDs |
|---|---|---|
| Add connection | New enabled edge between two unconnected nodes | 1 |
| Add node | Disable existing edge; insert hidden node; add two new edges (weight=1 in, original weight out) | 2 |

### Complexification from minimal topology

All genomes start with direct input→output connections only (no hidden nodes). Hidden structure grows exclusively via add-node mutation.

## Parameters

| Name | Default | Role |
|---|---|---|
| `node_add_rate` | 0.2 (ARIEL) | Probability of add-node mutation per call |
| `conn_add_rate` | 0.3 (ARIEL) | Probability of add-connection mutation per call |
| `c1`, `c2`, `c3` | problem-dependent | Coefficients for [[neat_speciation]] compatibility distance |
| `δ_t` | problem-dependent | Speciation compatibility threshold |

## Implementation Notes

- **Feed-forward ordering**: use Kahn's topological sort (BFS with in-degree tracking). Raise on cycle detection. See `get_node_ordering()` in [[cppn_neat_genome]].
- **Recurrent fallback**: if a cycle is detected, fall back to iterative relaxation for `len(nodes) + 1` steps.
- **Cache activation topology**: `incoming_map`, topological order, and input/output ID lists should be cached per genome and invalidated only on structural mutation or connection disable.
- **Innovation ID collisions in parallel runs**: the shared counter assumes single-threaded access; parallelism requires a lock or per-island counters.
- **Fitness in crossover**: the fitter-parent weighting requires that `genome.fitness` is synced from the EA individual's fitness before calling `crossover()`.

## When to Use

- When network topology is unknown and should be discovered by evolution rather than fixed by design.
- When starting from a small, fast-to-evaluate network and growing complexity incrementally is preferable to searching a large fixed topology space.
- When meaningful crossover between heterogeneous topologies is required — NEAT's innovation numbers solve the [[competing_conventions]] problem that defeats position-based crossover.

Prefer fixed-topology neuroevolution (e.g., CMA-ES on weight vectors) when: topology is well-understood, population is large enough to make speciation overhead costly, or strict inference-time constraints make variable graph size problematic.

## See Also

- [[neat_speciation]] — speciation mechanics, compatibility distance formula, fitness sharing
- [[competing_conventions]] — the problem NEAT's historical markings solve
- [[cppn_neat_genome]] — ARIEL implementation of this algorithm
- [[CMA-ES_Algorithm]] — alternative optimizer for fixed-topology neuroevolution
