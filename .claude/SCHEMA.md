# Wiki Schema — ariel EvoDevo Robotics Project

This wiki is a technical knowledge base for the **ariel** project: a framework for co-evolving modular robot morphologies and neural controllers via MuJoCo simulation, neuroevolution, and black-box optimization.

## Page Types

Five page types exist. Every file **must** have YAML frontmatter with `type`, `tags`, `source`, and `date_ingested`.

---

### 1. `api_reference` — External library API

For classes, functions, or modules from third-party libraries (MuJoCo, Nevergrad, EvoTorch, DEAP, dm-control, PyTorch, NumPy, etc.)

**Naming:** Match the Python name exactly — `MjData.md`, `mujoco_simulation_functions.md`, `Nevergrad_Optimizers.md`

**Required sections:**
- One-sentence definition of what this object/function does
- `## Signature` — exact Python signature in a `python` code block
- `## Parameters` — table with columns: Name | Type | Description
- `## Returns` — (for functions/methods) what is returned and its type
- `## Examples` — concrete usage code blocks extracted verbatim from source
- `## Notes` — (optional) gotchas, constraints, undocumented behavior

**Frontmatter:**
```yaml
---
type: api_reference
tags: [mujoco, python, class]
source: <docs URL>
date_ingested: YYYY-MM-DD
---
```

---

### 2. `algorithm_reference` — Algorithm or optimization method

For evolutionary, optimization, or control algorithms (CMA-ES, NEAT, CPG, RevDE, DEAP operators, etc.)

**Naming:** Descriptive, PascalCase or hyphen-separated — `CMA-ES_Algorithm.md`, `CPG_Controller.md`, `NEAT_Genome.md`

**Required sections:**
- One-paragraph overview: what it does, when to use it
- `## Formulation` — key equations and/or pseudocode in a code block; preserve notation exactly
- `## Parameters` — table with columns: Name | Default | Role
- `## Implementation Notes` — initialization, termination criteria, numerical concerns
- `## When to Use` — conditions that favor this algorithm; tradeoffs vs alternatives
- `## See Also` — `[[wikilinks]]` to related algorithms or concepts

**Frontmatter:**
```yaml
---
type: algorithm_reference
tags: [cma-es, evolution-strategy, optimization]
source: <paper URL or "ariel project">
date_ingested: YYYY-MM-DD
---
```

---

### 3. `concept_reference` — Scientific or domain concept

For domain knowledge: genotype-phenotype mapping, morphological descriptors, fitness landscapes, modular robotics assembly rules, CPG theory, co-evolution dynamics, etc.

**Naming:** snake_case — `genotype_phenotype_mapping.md`, `morphological_descriptors.md`, `fitness_landscape.md`

**Required sections:**
- Definition paragraph
- `## Theory` — underlying principles, key equations if applicable
- `## In Ariel` — how ariel implements or applies this concept; cite specific modules/classes with paths
- `## Practical Notes` — pitfalls, rules of thumb, parameter ranges observed in practice
- `## See Also`

**Frontmatter:**
```yaml
---
type: concept_reference
tags: [morphology, evolution, genotype]
source: <paper or "ariel project">
date_ingested: YYYY-MM-DD
---
```

---

### 4. `ariel_reference` — Ariel internal API

For classes, functions, or modules defined inside the ariel codebase (`src/ariel/`).

**Naming:** Match the Python name — `TreeGenome.md`, `construct_mjspec_from_graph.md`, `Controller.md`

**Required sections:**
- One-sentence purpose
- `## Location` — `src/ariel/<module_path>.py`
- `## Signature` — exact class definition or function signature in a `python` code block
- `## Parameters / Attributes` — table with columns: Name | Type | Description
- `## Examples` — usage snippets drawn from actual call sites in the codebase
- `## Notes` — design decisions, known limitations, coupling to other ariel components

**Frontmatter:**
```yaml
---
type: ariel_reference
tags: [ariel, ec, morphology]
source: ariel project source code
date_ingested: YYYY-MM-DD
---
```

---

### 5. `source_summary` — Ingested source record

One file per ingested source. Records what was extracted.

**Naming:** `Source - <Descriptive Title>.md`

**Required sections:**
- 1–2 sentences describing the source and its relevance to ariel
- `## Entity Pages Created` — bulleted `[[wikilinks]]` to every page created or updated, with one-line annotation per link

**Frontmatter:**
```yaml
---
type: source_summary
tags: [source]
source: <URL or filename>
author: <author(s)>
date_ingested: YYYY-MM-DD
---
```

---

## Global Formatting Rules

1. **H1** = filename without `.md` extension (must always match)
2. **H2** for major sections, **H3** for subsections within a section
3. **Code blocks**: ` ```python ` for Python, plain ` ``` ` for pseudocode or math notation
4. **Wiki links**: `[[PageName]]` (Obsidian-style) for every reference to another page in this wiki
5. **Tables**: preferred over prose for parameter/attribute lists with ≥ 3 items
6. **No padding**: be terse. Prefer a table over a paragraph. Skip sections that have no content.

---

## Tag Vocabulary

Use these tags consistently to keep the wiki searchable.

| Domain | Tags |
|--------|------|
| Simulation | `mujoco`, `dm-control`, `physics`, `mjspec`, `mjdata`, `mjmodel` |
| Evolution / Optimization | `cma-es`, `neat`, `deap`, `evotorch`, `nevergrad`, `ga`, `evolution-strategy`, `black-box` |
| Morphology | `morphology`, `modular-robot`, `body-phenotype`, `tree-genome`, `cppn`, `graph` |
| Controllers | `cpg`, `neural-controller`, `neuroevolution`, `pytorch`, `nde` |
| Ariel internals | `ariel`, `ec`, `simulation`, `tracker`, `runner`, `lynx` |
| Meta | `source`, `algorithm`, `concept` |

---

## Ariel Project Context (for cross-referencing)

Key internal modules to link when relevant:

| Module path | What it is |
|-------------|-----------|
| `src/ariel/ec/` | EA base classes, Individual, mutation/crossover operators |
| `src/ariel/ec/genotypes/tree/` | Tree-based genome (JSON-serializable, NetworkX-based) |
| `src/ariel/ec/genotypes/nde/` | Neural Developmental Encoding (PyTorch) |
| `src/ariel/ec/genotypes/cppn/` | CPPN decoders for morphology generation |
| `src/ariel/body_phenotypes/robogen_lite/` | Modular robot body: Core, Brick, Hinge modules |
| `src/ariel/body_phenotypes/lynx_mjspec/` | MuJoCo spec pipeline for Lynx arm |
| `src/ariel/simulation/controllers/` | CPG-based and neural controllers |
| `src/ariel/simulation/environments/` | Terrain generators (flat, rugged, crater, etc.) |
| `src/ariel/simulation/tasks/` | Evaluation tasks (gait, locomotion, turning) |
| `src/ariel/parameters/` | Pydantic configs (servo specs, MuJoCo params) |
| `src/ariel/utils/` | Tracker, runners, renderers, morphological descriptors |
