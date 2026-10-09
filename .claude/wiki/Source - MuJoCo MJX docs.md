---
type: source_summary
tags: [source, mujoco, mjx, jax, gpu]
source: https://mujoco.readthedocs.io/en/stable/mjx.html
author: Google DeepMind MuJoCo team
date_ingested: 2026-10-07
---

# Source - MuJoCo MJX docs

Official MuJoCo documentation for MJX: the JAX-backed implementation of MuJoCo that enables GPU/TPU simulation and parallel-environment batching via `jax.vmap`. Core reference for everything we do with mjx + brax PPO + mujoco_playground.

## Entity Pages Created / Updated

- [[mjx_overview]] — two-impl overview (MJX-JAX vs MJX-Warp), feature parity matrix, minimal batched example. *Already current; no changes needed.*
- [[mjx_core_functions]] — `put_model`, `make_data`, `put_data`, `step`. **Updated**: added full sections for `mjx.forward`, `mjx.inverse`, and MJX-Warp's `data.where(done, reset_data)` batched-reset helper.
- [[mjx_performance]] — solver tuning, broadphase params, Triton flag, mesh vertex limits, scaling characteristics. *Already current.*
- [[mjx_warp]] — Warp impl specifics: `naconmax`/`njmax` semantics, GraphMode options, batch rendering, multi-GPU `pmap`. *Already current.*
