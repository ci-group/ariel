---
type: api_reference
tags: [mujoco, mjx, jax, gpu, simulation, api]
source: https://mujoco.readthedocs.io/en/stable/mjx.html
date_ingested: 2026-04-13
---

# mjx_core_functions

Core MJX functions for placing MuJoCo models/data on device and stepping simulation. Mirrors the MuJoCo C API with PEP 8-compliant names.

## Structs: mjx.Model and mjx.Data

`mjx.Model` and `mjx.Data` are the device-side equivalents of [[MjModel]] and [[MjData]]:

- Contain JAX arrays (copied onto device)
- Some fields are absent (unsupported features or implementation-private fields)
- Support batch dimensions for domain randomization (`mjx.Model`) or parallel simulation (`mjx.Data`)
- NumPy fields in `mjx.Model`/`mjx.Data` are **structural** — modifying them triggers JAX recompilation
- JAX array fields (e.g. `jnt_range`) can be modified at runtime

**Neither is meant to be constructed manually.**

## put_model

```python
mjx.put_model(model: mujoco.MjModel, impl: str | None = None, graph_mode: GraphMode | None = None) -> mjx.Model
```

Places an `MjModel` onto device.

| Parameter | Type | Description |
|---|---|---|
| `model` | `mujoco.MjModel` | CPU-side compiled model |
| `impl` | `str \| None` | `'warp'` for MJX-Warp; `None` defaults to MJX-JAX |
| `graph_mode` | `GraphMode \| None` | CUDA graph mode (MJX-Warp only); see [[mjx_warp]] |

### Returns
`mjx.Model` — device-resident model pytree.

## make_data

```python
mjx.make_data(
    model: mujoco.MjModel | mjx.Model,
    impl: str | None = None,
    naconmax: int | None = None,
    njmax: int | None = None
) -> mjx.Data
```

Creates a zeroed `mjx.Data` on device. Preferred over `put_data` when constructing batched data inside `vmap`.

| Parameter | Type | Description |
|---|---|---|
| `model` | `MjModel \| mjx.Model` | Model (CPU or device) |
| `impl` | `str \| None` | `'warp'` for MJX-Warp |
| `naconmax` | `int \| None` | MJX-Warp only: max contacts across all worlds combined |
| `njmax` | `int \| None` | MJX-Warp only: max constraints per world |

**`naconmax` and `njmax`** must be set for MJX-Warp. Tune via the viewer; scale `naconmax` by the number of environments.

### Returns
`mjx.Data` — zeroed device data.

### Example

```python
model = mujoco.MjModel.from_xml_string("...")
mjx_model = mjx.put_model(model)
mjx_data = mjx.make_data(mjx_model)
```

MJX-Warp example:
```python
mj_model = mujoco.MjModel.from_xml_path(...)
model = mjx.put_model(mj_model, impl='warp')
data = mjx.make_data(mj_model, impl='warp', naconmax=naconmax, njmax=njmax)
```

## put_data

```python
mjx.put_data(model: mujoco.MjModel, data: mujoco.MjData) -> mjx.Data
```

Copies an existing `MjData` (with current simulation state) to device.

| Parameter | Type | Description |
|---|---|---|
| `model` | `mujoco.MjModel` | CPU-side model |
| `data` | `mujoco.MjData` | CPU-side data to copy |

### Returns
`mjx.Data` — device copy of the data.

### Example

```python
model = mujoco.MjModel.from_xml_string("...")
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)
```

## step

```python
mjx.step(model: mjx.Model, data: mjx.Data) -> mjx.Data
```

Advances simulation by one timestep. Returns a new `mjx.Data`; does not mutate in place (JAX functional semantics).

| Parameter | Type | Description |
|---|---|---|
| `model` | `mjx.Model` | Device model |
| `data` | `mjx.Data` | Current simulation state |

### Returns
`mjx.Data` — updated simulation state after one step.

### Notes
- Not JIT-compiled by default; wrap with `jax.jit`.
- Use `jax.vmap` over `step` (or a function calling it) for batched parallel simulation.

## Notes

- All MJX functions are **not JIT-compiled by default** — users must explicitly apply `jax.jit`.
- MuJoCo function names map to PEP 8 snake_case in MJX (e.g. `mj_forward` → `mjx.forward`).
- Most [[mujoco_simulation_functions]] and some sub-components are available from the top-level `mjx` module.

## See Also

- [[mjx_overview]] — overview, feature parity, installation
- [[mjx_warp]] — MJX-Warp graph modes, batch rendering
- [[mjx_performance]] — tuning parameters
- [[MjModel]], [[MjData]] — CPU-side counterparts
