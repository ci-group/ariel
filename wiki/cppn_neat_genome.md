---
title: "CPPN-NEAT Genome in ARIEL"
authors: "CI-Group (VU Amsterdam)"
year: 2024
source: "https://ci-group.github.io/ariel/autoapi/ariel/body_phenotypes/robogen_lite/cppn_neat/genome/index.html"
tags: [cppn, neat, genome, evolutionary-robotics, ariel, neuroevolution]
---

# CPPN-NEAT Genome in ARIEL

ARIEL uses CPPN (Compositional Pattern Producing Networks) evolved with NEAT (NeuroEvolution of Augmenting Topologies) as the genotype for robot morphologies.

## What is CPPN-NEAT?

- **CPPN:** A neural network whose topology encodes spatial patterns. When queried at (x, y, z, ...) coordinates, it outputs module type/rotation scores for those positions.
- **NEAT:** An evolutionary algorithm that evolves both network weights *and* topology (adding nodes/connections over generations), using innovation numbers to track gene history for crossover.

## Genome Class API

See [[robogen_lite_api]] for full method signatures.

### Key operations
| Operation | Method |
|-----------|--------|
| Create random genome | `Genome.random(num_inputs, num_outputs, next_node_id, next_innov_id)` |
| Structural mutation | `genome.mutate(node_add_rate, conn_add_rate, ...)` |
| Recombination | `genome.crossover(other)` |
| Forward pass | `genome.activate(inputs) → list[float]` |
| Serialize | `genome.to_dict()` / `Genome.from_dict(data)` |
| Deep copy | `genome.copy()` |

### Node Gene
```python
Node(_id, _typ, _activation: ActivationFunction, _bias)
```
Node types: input, hidden, output. Supports multiple activation functions via `ActivationFunction` enum.

### Topological Evaluation
- `get_node_ordering()` runs **Kahn's algorithm** for feed-forward networks.
- `activate()` falls back to **iterative relaxation** when cycles exist (recurrent topologies).

## Decoding CPPN → Morphology

The `MorphologyDecoderBestFirst` queries the genome at candidate attachment positions, applies **softmax** to raw output scores, and greedily selects the best placement until `max_modules` is reached. Result is a `nx.DiGraph`.

See [[robogen_lite_api#decoders]] for decoder details.

## Innovation Numbers & Speciation

NEAT tracks structural gene identity via innovation numbers (`next_innov_id`). This enables meaningful crossover: matching genes by innovation number rather than position. ARIEL exposes this via `id_manager`.

For the full NEAT algorithm (compatibility distance formula, speciation mechanics, fitness sharing): see [[stanley_2002]] and [[neat_speciation]].

## Best Practices

- Keep `max_modules` consistent across a population (affects fitness landscape).
- Use `genome.copy()` before mutation to preserve parent for elitism.
- `from_dict` / `to_dict` enable checkpointing entire populations.
