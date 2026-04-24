---
type: api_reference
tags: [mujoco, mjx, jax, gpu, simulation, warp, rendering]
source: https://mujoco.readthedocs.io/en/stable/mjx.html
date_ingested: 2026-04-13
---

# mjx_warp

MJX-Warp is the NVIDIA GPU-optimized MuJoCo implementation in MJX. It resolves key performance bottlenecks of [[mjx_overview#MJX-JAX]] around contacts and constraints, and fully supports mesh collisions. Does not support autodiff.

## Basic Usage

```python
mj_model = mujoco.MjModel.from_xml_path(...)
model = mjx.put_model(mj_model, impl='warp')
data = mjx.make_data(mj_model, impl='warp', naconmax=naconmax, njmax=njmax)
```

`naconmax` = max contacts across **all worlds combined** (scale by number of environments).
`njmax` = max constraints per world.

## Contacts

In MJX-Warp, contacts live in **`mjx.Data._impl`** (private), not `mjx.Data.contact`. Read contacts via [contact sensors](sensor-contact) only.

## GraphMode

Controls CUDA graph caching behavior. Set via `mjx.put_model`:

```python
import mujoco.mjx.warp as mjxw

model = mjx.put_model(mj_model, impl='warp', graph_mode=mjxw.GraphMode.WARP_STAGED)
```

| Mode | Description |
|---|---|
| `JAX` | Incompatible with MuJoCo Warp (child graph nodes can't roll up into XLA graph) |
| `WARP` | **Default.** Warp captures CUDA graph internally, cached by XLA buffer pointers. Re-captures if buffer pointers change (JAX/XLA may change them unexpectedly). |
| `WARP_STAGED` | Staging buffers ensure stable CUDA pointers; one-time capture. Higher memory use, avoids excessive recaptures. |
| `WARP_STAGED_EX` | Like `WARP_STAGED` but copies moved outside initial graph capture. |

### Performance Comparison (Steps per Second)

| Configuration | Humanoid | Aloha Pot |
|---|---|---|
| Pure Warp (No JAX FFI) | 3.35M | 2.45M |
| JAX FFI (`WARP`) | 2.96M | 2.33M |
| JAX FFI (`WARP`, forced recaptures every step) | 0.80M | 0.65M |
| JAX FFI (`WARP_STAGED`) | 2.67M | 1.96M |

If bottlenecked by excessive recaptures, use `WARP_STAGED` or `WARP_STAGED_EX`.

## Batch Rendering

Hardware-accelerated multi-environment pixel observations (RGB, depth).

### create_render_context

```python
from mujoco.mjx import create_render_context

rc = create_render_context(
    mjm=m,
    nworld=nworld,
    cam_res=(width, height),
    use_textures=True,
    use_shadows=True,
    render_rgb=[True] * ncam,
    render_depth=[False] * ncam,
    enabled_geom_groups=[0, 1, 2],
)
```

| Parameter | Description |
|---|---|
| `mjm` | CPU-side `MjModel` |
| `nworld` | Number of parallel worlds (fixed at creation time) |
| `cam_res` | `(width, height)` camera resolution |
| `use_textures` | Enable texture mapping |
| `use_shadows` | Enable shadow rendering |
| `render_rgb` | Per-camera RGB enable list |
| `render_depth` | Per-camera depth enable list |
| `enabled_geom_groups` | List of geom group indices to render |

Hold a reference to `rc` for the lifetime of the program. Pass `rc.pytree()` to JAX-compiled functions.

### render

```python
from mujoco.mjx import get_rgb

@jax.jit
def render_fn(mx, d, rc_pytree):
    # 1. Update BVH for current scene state
    d = mjx.refit_bvh(mx, d, rc_pytree)

    # 2. Render all configured cameras
    pixels, _ = mjx.render(mx, d, rc_pytree)

    # 3. Extract RGB for camera 0
    rgb = get_rgb(rc_pytree, 0, pixels)

    return rgb, d

rgb, d = render_fn(mx, d, rc.pytree())
```

| Function | Description |
|---|---|
| `mjx.refit_bvh(mx, d, rc_pytree)` | Updates bounding volume hierarchy for current scene state |
| `mjx.render(mx, d, rc_pytree)` | Renders all configured cameras; returns `(pixels, ...)` |
| `get_rgb(rc_pytree, cam_idx, pixels)` | Extracts RGB tensor for a specific camera index |

**Warning:** `nworld` is fixed at context creation. `mjx.render` always returns a leading batch dim of `nworld`. Known issue: does not compose with `jax.vmap(jax.lax.scan)`.

## Multi-GPU with pmap

```python
ndevices = jax.local_device_count()
nworld_per_device = nworld // ndevices

rc = create_render_context(
    mjm=m,
    nworld=nworld_per_device,
    devices=[f'cuda:{i}' for i in range(ndevices)],
    cam_res=(width, height),
)
```

Create one render context per device by passing `devices`. Then use `jax.pmap` to parallelize. See [visualize_render.py](https://github.com/google-deepmind/mujoco/blob/main/mjx/mujoco/mjx/warp/visualize_render.py) for a complete example.

## Notes

- MJX-Warp requires `pip install mujoco-mjx[warp]`
- No autodiff support; no near-term plans to add it
- Fully supports mesh collisions (no vertex count restrictions)
- Contacts accessed via `Data._impl` only — use contact sensors for contact readout

## See Also

- [[mjx_overview]] — implementation comparison, feature parity
- [[mjx_core_functions]] — `put_model`, `make_data`, `step`
- [[mjx_performance]] — MJX-JAX performance tuning and sharp bits
