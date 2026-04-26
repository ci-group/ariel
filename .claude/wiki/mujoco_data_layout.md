---
type: api_reference
tags: [mujoco, python, simulation, mjdata, mjmodel]
source: https://mujoco.readthedocs.io/en/stable/programming/simulation.html
date_ingested: 2026-04-13
---
# mujoco_data_layout

Memory layout conventions in MuJoCo: matrix format, quaternions, sparse matrices, buffer layout, and the internal stack.

## Matrix Format: Row-Major

All matrices in MuJoCo are **row-major**. A linear array `(a0, a1, a2, a3, a4, a5)` represents:
```
a0 a1 a2
a3 a4 a5
```

All utility functions (`mju_mulMatMat`, `mju_mulMatVec`, etc.) assume this layout.

> **Gotcha:** `mjContact.frame` stores 3 frame axes as rows, but most other frame matrices (e.g. `xmat`, `geom_xmat`) store axes in row-major order with axes as rows too — the convention is consistent but easy to misread.

## Quaternions

MuJoCo uses unit quaternions `q = (w, x, y, z)`:
- `(x, y, z)` = rotation axis unit vector × sin(a/2), where `a` = rotation angle in radians
- `w = cos(a/2)`
- Null rotation = `(1, 0, 0, 0)` — this is the default for all quaternion fields in MJCF

## Sparse Matrices

MuJoCo exploits sparsity for O(N) vs O(N³) scaling:

- `data.qM` (joint-space inertia) and `data.qLD` (LTDl factorization): always sparse
  - `qM` uses custom tree-topology format; `qLD` uses CSR
- `data.efc_J` (constraint Jacobian): sparse when `mj_isSparse(model)` returns true

### CSR Format for Constraint Jacobian

For a matrix `A` of shape `(m, n)`:

| Variable | Size | Meaning |
|----------|------|---------|
| `A` | `m × n` | Real-valued data (worst-case allocation) |
| `A_rownnz` | `m` | Number of non-zeros per row |
| `A_rowadr` | `m` | Starting index of row data in `A` and `A_colind` |
| `A_colind` | `m × n` | Column indices of non-zero elements |

Element at row `r`, entry `k`: `A[A_rowadr[r] + k]` at column `A_colind[A_rowadr[r] + k]`, where `k < A_rownnz[r]`.

MuJoCo uses "uncompressed" layout internally: `A_rowadr[r] = r * n` (not `A_rowadr[r-1] + A_rownnz[r-1]`).

### Dense conversion (avoid if possible)

```python
M_dense = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M_dense, data.qM)  # MuJoCo never does this internally
```

## Buffer Layout / Pointer Aliasing

`mjModel.buffer` and `mjData.buffer` are single large allocations partitioned with byte-alignment. **Do not assume** adjacent fields are contiguous:

```python
# WRONG — there may be padding between qpos and qvel
state = data.qpos[:model.nq + model.nv]   # undefined behavior

# CORRECT — copy individually
qpos = data.qpos.copy()   # shape (nq,)
qvel = data.qvel.copy()   # shape (nv,)
```

## Internal Stack (Arena)

`mjData.arena` holds two types of dynamic memory:
1. **Constraint-related** (contact count unknown until detection runs)
2. **Temporary stack** (managed per-function)

Most top-level MuJoCo functions push/pop the stack internally. After any function returns, `mjData.pstack` is unchanged. Exception: `mj_resetData` sets `pstack = 0`.

### Using the stack in user code

```python
# Mark a stack frame
mujoco.mj_markStack(data)

# Allocate arrays (size checked; advances stack pointer)
my_arr = mujoco.mj_stackAllocNum(data, model.nq)   # mjtNum array
my_int = mujoco.mj_stackAllocInt(data, model.nv)   # int array

# Restore frame (free allocations)
mujoco.mj_freeStack(data)
```

> If an instability is detected inside `mj_step`, `mj_step1`, or `mj_step2`, `mj_resetData` is called internally, resetting `pstack`. Any user stack allocations across such calls are invalidated.

### Monitoring arena usage

```python
# Maximum arena bytes used since last reset
print(data.maxuse_arena)
```

Use this to tune the MJCF `<size memory="..."/>` attribute.

## 6D Spatial Vectors

Fields prefixed with `c` in `mjData` (e.g. `cvel`, `cacc`, `cdot`) are spatial vectors combining:
- 3D rotational component (first)
- 3D translational component (second)

This is Featherstone's spatial algebra convention (rotation before translation). No utility functions are provided for these.

## Notes

- `mj_fullM` converts the sparse inertia to dense — only use for debugging, never in a simulation loop.
- `mj_isSparse(model)` returns `True` if constraint Jacobian is stored sparse (depends on `model.opt`).
- `mj_isPyramidal(model)` returns `True` if pyramidal friction cones are active.

## See Also

[[MjData]], [[MjModel]], [[mujoco_simulation_functions]], [[mujoco_contacts]]
