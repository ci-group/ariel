---
type: api_reference
tags: [mujoco, mjx, jax, gpu, simulation, physics]
source: https://mujoco.readthedocs.io/en/stable/mjx.html
date_ingested: 2026-04-13
---

# mjx_overview

MuJoCo XLA (MJX) provides a JAX API for hardware-accelerated MuJoCo simulation on GPUs, TPUs, and Apple Silicon. Distributed as a separate `mujoco-mjx` PyPI package.

## Signature

```python
import mujoco
from mujoco import mjx
```

## Two Implementations

| Feature | MJX-Warp | MJX-JAX |
|---|---|---|
| Hardware | NVIDIA GPU only | NVIDIA/AMD GPU, TPU, Apple Silicon |
| Autodiff | ✗ | ✓ (mostly) |
| Contacts | `Data._impl` (private) | `Data.contact` |
| Single-scene speed | ~CPU-parity | ~10x slower than CPU |
| Mesh collisions | All | Convex only, <200 vertices |
| Primary use | Production RL, large scenes | RL + gradients, diverse hardware |

**MJX-Warp** uses MuJoCo Warp under the hood, resolving performance bottlenecks around contacts and constraints. Does not support automatic differentiation.

**MJX-JAX** is a pure-JAX re-implementation (successor to the Brax generalized pipeline). Runs on all XLA-supported hardware.

## Installation

```shell
pip install mujoco-mjx

# For Warp support (NVIDIA GPU only):
pip install mujoco-mjx[warp]
```

## Minimal Example

```python
# Throw a ball at 100 different velocities.

import jax
import mujoco
from mujoco import mjx

XML=r"""
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom size=".15" mass="1" type="sphere"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
mjx_model = mjx.put_model(model)

@jax.vmap
def batched_step(vel):
  mjx_data = mjx.make_data(mjx_model)
  qvel = mjx_data.qvel.at[0].set(vel)
  mjx_data = mjx_data.replace(qvel=qvel)
  pos = mjx.step(mjx_model, mjx_data).qpos[0]
  return pos

vel = jax.numpy.arange(0.0, 1.0, 0.01)
pos = jax.jit(batched_step)(vel)
print(pos)
```

## Enums

MJX enums follow the pattern `mjx.EnumType.ENUM_VALUE`, e.g. `mjx.JointType.FREE`. Enums for unsupported features are omitted.

## Command-Line Tools

```shell
# Performance metrics for an MJCF
mjx-testspeed --mjcf=/PATH/TO/MJCF/ --base_path=.

# Visualize model stepped by MJX physics
mjx-viewer --help
```

## Feature Parity Table

| Category | MJX-Warp | MJX-JAX |
|---|---|---|
| Dynamics | Forward, Inverse | Forward, Inverse |
| Differentiability | ✗ | ✓ |
| Joint types | All | FREE, BALL, SLIDE, HINGE |
| Transmission | All | JOINT, JOINTINPARENT, SITE, TENDON |
| Actuator Dynamics | All | NONE, INTEGRATOR, FILTER, FILTEREXACT, MUSCLE |
| Actuator Gain | All | FIXED, AFFINE, MUSCLE |
| Actuator Bias | All | NONE, AFFINE, MUSCLE |
| Geom | All | PLANE, HFIELD, SPHERE, CAPSULE, BOX, MESH fully; ELLIPSOID/CYLINDER vs primitives only |
| Integrators | All except IMPLICIT | EULER, RK4, IMPLICITFAST |
| Cone | All | PYRAMIDAL, ELLIPTIC |
| Condim | All | 1, 3, 4, 6 (1 not with ELLIPTIC) |
| Solver | All except PGS, noslip | CG, NEWTON |
| Fluid | All | flInertia only |
| Tendon Wrapping | All | JOINT, SITE, PULLEY, SPHERE, CYLINDER |
| Tendons | All | Fixed, Spatial |
| Sensors | All except PLUGIN | Large subset (see note) |
| Flex | VERTCOLLIDE, ELASTICITY | Not supported |
| Mass matrix | Sparse and Dense | Sparse and Dense |
| Jacobian | DENSE only | DENSE and SPARSE |
| Ray | All, BVH for meshes/hfield/flex | Slow for meshes; hfield/flex unimplemented |

**MJX-JAX sensors supported:** MAGNETOMETER, CAMPROJECTION, RANGEFINDER, JOINTPOS, TENDONPOS, ACTUATORPOS, BALLQUAT, FRAMEPOS, FRAMExAXIS (x/y/z), FRAMEQUAT, SUBTREECOM, CLOCK, VELOCIMETER, GYRO, JOINTVEL, TENDONVEL, ACTUATORVEL, BALLANGVEL, FRAMELINVEL, FRAMEANGVEL, SUBTREELINVEL, SUBTREEANGMOM, TOUCH, CONTACT, ACCELEROMETER, FORCE, TORQUE, ACTUATORFRC, JOINTACTFRC, TENDONACTFRC, FRAMELINACC, FRAMEANGACC

**Unsupported geoms (MJX-JAX):** SDF. Collisions between (SPHERE, BOX, MESH, HFIELD) and CYLINDER; (BOX, MESH, HFIELD) and ELLIPSOID.

## Notes

- MJX raises an exception if asked to copy an `MjModel` to device that uses unsupported features.
- MJX functions are **not JIT-compiled by default** — user must call `jax.jit` explicitly.
- MJX-JAX is successor to the Brax generalized pipeline (no longer maintained in Brax).

## See Also

- [[mjx_core_functions]] — `put_model`, `make_data`, `step`, structs
- [[mjx_warp]] — MJX-Warp graph modes, batch rendering
- [[mjx_performance]] — tuning parameters and sharp bits
- [[MjModel]] — CPU-side model
- [[MjData]] — CPU-side data
