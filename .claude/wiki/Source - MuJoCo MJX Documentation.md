---
type: source_summary
tags: [source, mujoco, mjx, jax, gpu]
source: https://mujoco.readthedocs.io/en/stable/mjx.html
author: Google DeepMind
date_ingested: 2026-04-13
---

# Source - MuJoCo MJX Documentation

Official MuJoCo XLA (MJX) documentation covering the JAX-based hardware-accelerated MuJoCo API. Relevant to ariel for GPU-accelerated batched robot simulation during evolutionary search.

## Entity Pages Created

- [[mjx_overview]] — Installation, two implementations (MJX-JAX vs MJX-Warp), minimal example, full feature parity table
- [[mjx_core_functions]] — Core API: `put_model`, `make_data`, `put_data`, `step`; struct semantics for `mjx.Model`/`mjx.Data`
- [[mjx_warp]] — MJX-Warp: `GraphMode` enum, CUDA graph tuning, batch rendering API (`create_render_context`, `render`, `get_rgb`), multi-GPU with `pmap`
- [[mjx_performance]] — MJX-JAX performance tuning parameters, broadphase config, GPU env vars, sharp bits (single-scene slowdown, mesh vertex limits, contact scaling)
