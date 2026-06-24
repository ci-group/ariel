---
type: source_summary
tags: [nevergrad, cma-es, source]
source: https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/optimization/optimizerlib.py
date_ingested: 2026-06-24
---

# Nevergrad optimizerlib.py

Source file containing Nevergrad's optimizer factories. Ingested for the `ParametrizedCMA` signature and all pre-registered CMA variants — the upstream HTML docs only list names, not constructor parameters or `inopts` passthrough.

## Entity Pages Created

- [[ParametrizedCMA]] — full `__init__` signature, all 13 pre-registered CMA variants with their preset scale/popsize_factor/elitist/diagonal values, `inopts` escape hatch to [[CMAEvolutionStrategy]] CMAOptions.

## Entity Pages Updated

- [[Nevergrad_Optimizers]] — added pointer to [[ParametrizedCMA]] for CMA variants.

## Gaps

- IPOP/BIPOP restart wrappers not found in this file; likely in `nevergrad/optimization/recastlib.py` or `families.py`.
- `algorithm: str = "quad"` parameter not explained in source comments — meaning unverified.
