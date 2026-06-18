# Bodies — genotypes, decoders, prebuilt robots

How to get a robot body (a `CoreModule` / MuJoCo spec) to spawn into a world.
Three routes: **prebuilt**, **decode a genome**, or **build a graph by hand**.

## Prebuilt robots (fastest)

Each returns a `CoreModule`; pass `.spec` to `world.spawn(...)`.

| Function | Import |
|---|---|
| `body_spider45()` | `from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider_with_blocks import body_spider45` |
| `body_spider()` | `…prebuilt_robots.spider_with_blocks import body_spider` |
| `spider()` | `…prebuilt_robots.spider import spider` |
| `gecko()` | `…prebuilt_robots.gecko import gecko` |
| `random_spider20(seed=1337)` | `…prebuilt_robots.random_20 import random_spider20` |

More (snake, turtle, iguana, centipede_3/4/5, spider_8/12/16, …) in
[prebuilt_robots/john_set.py](../../src/ariel/body_phenotypes/robogen_lite/prebuilt_robots/john_set.py).

```python
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider_with_blocks import (
    body_spider45,
)
core = body_spider45()
world.spawn(core.spec, position=[0, 0, 0.1])   # see simulation.md
```

## Decode a genome → body

### CPPN-NEAT genome → graph → spec

```python
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    MorphologyDecoderBestFirst,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)

genome = Genome.random(
    num_inputs=6, num_outputs=NUM_CPPN_OUTPUTS,
    next_node_id=..., next_innov_id=...,
)
graph = MorphologyDecoderBestFirst(cppn_genome=genome, max_modules=10).decode()
spec = construct_mjspec_from_graph(graph).spec   # CoreModule -> .spec
```

`Genome` key methods (see [genome.py](../../src/ariel/body_phenotypes/robogen_lite/cppn_neat/genome.py)):
`Genome.random(...)`, `.mutate(...)`, `.crossover(other, is_maximisation=...)`,
`.to_dict()` / `Genome.from_dict(...)`, `.get_node_ordering()`.
`IdManager` hands out node/innovation IDs. Full usage:
[examples/re_book/2_body_brain_evolution.py](../../examples/re_book/2_body_brain_evolution.py).

### Probability matrices → graph

`HighProbabilityDecoder` — [hi_prob_decoding.py](../../src/ariel/body_phenotypes/robogen_lite/decoders/hi_prob_decoding.py).
Example: [examples/b_robots/_hi_prob_dec.py](../../examples/b_robots/_hi_prob_dec.py).

### Vector → graph

`VectorDecoder` — [vector_decoding.py](../../src/ariel/body_phenotypes/robogen_lite/decoders/vector_decoding.py).

### Tree genome (alternative encoding)

`TreeGenome` + functional operators (`random_tree`, `add_node`, `subtree_swap`,
…) in [ec/genotypes/tree/](../../src/ariel/ec/genotypes/tree/). Example:
[examples/c_genotypes/1_body_evolution_tree.py](../../examples/c_genotypes/1_body_evolution_tree.py).

## Build / save a graph by hand

`construct_mjspec_from_graph(graph: DiGraph) -> CoreModule` turns a NetworkX
graph of modules into a spec. JSON save/load of graphs:
[examples/b_robots/_graph_to_robot.py](../../examples/b_robots/_graph_to_robot.py).

## Module vocabulary

Module types (`Core`, `Brick`, `Hinge`), faces, and rotations are defined in
[robogen_lite/config.py](../../src/ariel/body_phenotypes/robogen_lite/config.py)
(`NUM_OF_TYPES_OF_MODULES`, `NUM_OF_ROTATIONS`, `ModuleFaces`, …). Concepts:
[../../wiki/robogen_lite_api.md](../../wiki/robogen_lite_api.md).

## Next

Spawn it → [simulation.md](simulation.md). Drive it → [controllers.md](controllers.md).
