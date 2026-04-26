---
type: source_summary
tags: [cma-es, optimization, evolution-strategy, black-box]
source: https://arxiv.org/pdf/1604.00772
author: Nikolaus Hansen (Inria)
date_ingested: 2026-04-09
---

# The CMA Evolution Strategy: A Tutorial (Hansen, 2023)

Authoritative tutorial on the CMA-ES algorithm by its creator. Covers derivation, all update equations, default parameters, implementation concerns, and boundary handling. arXiv:1604.00772v2, 39 pages.

## Pages covered
- pp. 1–39 (complete document)

## Entity pages created

- [[CMA-ES_Algorithm]] — Core algorithm: sampling, mean update, step-size control, covariance matrix adaptation (full pseudocode from Figure 6 / Appendix A)
- [[CMA-ES_Parameters]] — Complete default parameter table (Table 1), all formulas for λ, μ, weights, c_σ, d_σ, c_c, c_1, c_μ
- [[CMA-ES_Practical_Concerns]] — Initialization, termination criteria, flat fitness, boundary/constraint handling
