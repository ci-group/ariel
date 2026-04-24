---
type: api_reference
tags: [mujoco, python, functions, inertia, dynamics]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MuJoCo Inertia Matrix Functions

Functions for computing and operating on the joint-space inertia matrix M (size `nv × nv`).

## Functions

### `mj_crb`

```python
mujoco.mj_crb(model, data)
```

Composite Rigid Body (CRB) algorithm. Fills `data.crb` (composite inertias, `(nbody, 10)`). Must be called after `mj_comPos`.

---

### `mj_factorM`

```python
mujoco.mj_factorM(model, data)
```

Sparse L^T D L factorization of the inertia matrix. Results stored in `data.qLD` and `data.qLDiagInv`. Required before `mj_solveM` or `mj_solveM2`.

---

### `mj_solveM`

```python
mujoco.mj_solveM(model, data, x, y, n)
```

Solve `M x = y` using the existing factorization.

**Parameters:**
- `x` — output array, shape `(nv, n)`
- `y` — input (RHS) array, shape `(nv, n)`
- `n` — number of RHS vectors

---

### `mj_solveM2`

```python
mujoco.mj_solveM2(model, data, x, y, n)
```

Half solve: `x = sqrt(D^-1) * (L^T)^-1 * y`. Same signature as `mj_solveM`.

---

### `mj_mulM`

```python
mujoco.mj_mulM(model, data, res, vec)
```

Multiply a vector by the inertia matrix: `res = M * vec`.

**Parameters:**
- `res` — output, shape `(nv,)`
- `vec` — input, shape `(nv,)`

---

### `mj_mulM2`

```python
mujoco.mj_mulM2(model, data, res, vec)
```

Multiply by the square root of M: `res = sqrt(M) * vec`. Same shape as `mj_mulM`.

---

### `mj_fullM`

```python
mujoco.mj_fullM(model, dst, M)
```

Convert sparse inertia matrix to dense format.

**Parameters:**
- `dst` — output dense matrix, shape `(nv, nv)`
- `M` — sparse input (from `data.qM`)

## Example

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('model.xml')
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)    # computes everything including crb

# Factorize and solve M*x = qacc
mujoco.mj_factorM(model, data)
rhs = data.qacc.copy().reshape(-1, 1)
result = np.zeros_like(rhs)
mujoco.mj_solveM(model, data, result, rhs, 1)

# Get full dense inertia matrix
M_dense = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M_dense, data.qM)
```

See [[mujoco_kinematics_functions]], [[mujoco_simulation_functions]].
