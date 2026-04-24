---
type: concept_reference
tags: [mujoco, mjx, jax, gpu, simulation, performance, optimization]
source: https://mujoco.readthedocs.io/en/stable/mjx.html
date_ingested: 2026-04-13
---

# mjx_performance

Performance tuning guide and known limitations for MJX-JAX. MJX-Warp mitigates most of these issues; prefer MJX-Warp for large scenes or production RL pipelines.

## Theory

MJX-JAX specializes in large batches of parallel identical scenes on SIMD hardware (GPU/TPU). It uses branchless algorithms to maintain vectorizability. Single-scene simulation or scenes with many contacts break this specialization.

## MJX-JAX Tuning Parameters

| Parameter | Recommendation | Reason |
|---|---|---|
| `option/iterations` | Reduce below MuJoCo defaults | Fewer solver iterations = more throughput; RL tolerates approximate physics |
| `option/ls_iterations` | Reduce below MuJoCo defaults | Same as above |
| `option/solver` | Use `NEWTON` (1 iteration often sufficient); `CG` for TPU | NEWTON converges fast on GPU |
| `contact/pair` | Explicitly list valid geom pairs | Reduces contacts MJX-JAX must consider; dramatic effect on performance |
| `maxhullvert` | Set to `64` or less | Better convex mesh collision performance |
| `option/flag/eulerdamp` | Disable if not needed for stability | Improves performance |
| `option/jacobian` | Set `dense` for GPU, `sparse` for TPU | GPU: dense is faster if it fits; TPU: sparse with Newton = 2–3x speedup |

### Broadphase (Experimental)

MJX-JAX requires explicit broadphase parameters (custom numeric):

| Parameter | Effect |
|---|---|
| `max_contact_points` | Caps contact points sent to solver per condim type |
| `max_geom_pairs` | Caps geom-pairs sent to collision functions per geom-type pair |

Example: the shadow hand environment uses these parameters.

### GPU Environment Variable

```shell
XLA_FLAGS=--xla_gpu_triton_gemm_any=true
```

Enables the Triton-based GEMM emitter for any supported GEMM. Yields ~30% speedup on NVIDIA GPUs.

## In Ariel

When using MJX-JAX for batched evaluation of robot morphologies:
- Use the `NEWTON` solver with 1–2 iterations for fast throughput
- Pre-declare contact pairs for expected ground-contact geoms
- Set `maxhullvert=64` when using modular robot meshes
- Set `XLA_FLAGS=--xla_gpu_triton_gemm_any=true` in the environment before launching

## Practical Notes (Sharp Bits)

### Single scene is 10x slower

MJX-JAX simulating a single scene is **~10x slower** than CPU MuJoCo. MJX-JAX only outperforms CPU at thousands or tens of thousands of parallel instances.

### Mesh collision vertex limits

MJX-JAX uses a branchless SAT (Separating Axis Test) rather than MPR or GJK/EPA:

| Collision type | Max vertices for reasonable performance |
|---|---|
| Convex mesh vs primitive | ~200 vertices |
| Convex mesh vs convex mesh | <32 vertices |

Use `maxhullvert` in the MuJoCo compiler to enforce limits. See shadow hand config for an example of tuned mesh collisions.

### Large scenes with many contacts scale poorly

GPU branching penalties affect broadphase collision detection. As a scene adds more bodies/contacts, MJX-JAX throughput drops faster than CPU MuJoCo. Benchmark on Humanoid scaling (1–10 humanoids, batch 8192, A100):
- 1 humanoid: 1.8M SPS (GPU) vs 650K (CPU)
- 10 humanoids: GPU drops sharply; CPU scales much better

**MJX-Warp resolves this** via a superior contact/constraint solver.

## See Also

- [[mjx_overview]] — feature parity table, implementation comparison
- [[mjx_core_functions]] — `put_model`, `make_data`, `step`
- [[mjx_warp]] — MJX-Warp graph modes that mitigate these issues
