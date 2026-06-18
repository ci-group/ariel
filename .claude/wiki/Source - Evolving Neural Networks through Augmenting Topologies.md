---
type: source_summary
tags: [source, neat, neuroevolution, cppn, algorithm]
source: https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf
author: Stanley, K.O. & Miikkulainen, R.
date_ingested: 2026-05-14
---

# Source - Evolving Neural Networks through Augmenting Topologies

Foundational 2002 paper introducing NEAT — the algorithm used by ARIEL's CPPN genome evolution. Covers genome encoding with innovation numbers, speciation, fitness sharing, and incremental complexification from minimal topologies.

## Entity Pages Created

- [[NEAT_Algorithm]] — full algorithm reference: genome encoding, crossover by innovation number, structural mutations, complexification strategy, implementation notes for ARIEL
- [[neat_speciation]] — concept page: compatibility distance formula, fitness sharing, niche pressure, note that ARIEL's current CPPN loop does not implement speciation
- [[competing_conventions]] — concept page: why position-based crossover fails for variable-topology networks, and how innovation numbers resolve it
