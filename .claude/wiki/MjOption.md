---
type: api_reference
tags: [mujoco, python, class, physics-options]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MjOption

Physics solver options. Accessed as `model.opt` on an [[MjModel]] instance. Pre-initialized by `mj_defaultOption()`.

## Access

```python
model.opt.timestep = 0.002
model.opt.gravity[:] = [0, 0, -9.81]
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `timestep` | float | simulation timestep (seconds) |
| `apirate` | float | update rate for remote API |
| `impratio` | float | ratio of friction to normal impedance |
| `tolerance` | float | constraint solver tolerance |
| `noslip_tolerance` | float | no-slip solver tolerance |
| `gravity` | `(3,)` | gravitational acceleration vector |
| `wind` | `(3,)` | wind velocity vector |
| `magnetic` | `(3,)` | global magnetic flux vector |
| `density` | float | medium density |
| `viscosity` | float | medium viscosity |
| `integrator` | int | `mjtIntegrator`: Euler=0, RK4=1, implicit=2, implicitfast=3 |
| `collision` | int | `mjtCollision` mode |
| `cone` | int | friction cone type (`mjtCone`) |
| `jacobian` | int | Jacobian type |
| `solver` | int | constraint solver: PGS=0, CG=1, Newton=2 |
| `iterations` | int | max constraint solver iterations |
| `noslip_iterations` | int | max no-slip iterations |
| `disableflags` | int | bitmask of disabled features (`mjtDisableBit`) |
| `enableflags` | int | bitmask of enabled features (`mjtEnableBit`) |

## Integrator Values (`mjtIntegrator`)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `mjINT_EULER` | semi-implicit Euler (default) |
| 1 | `mjINT_RK4` | 4th-order Runge-Kutta |
| 2 | `mjINT_IMPLICIT` | implicit integration |
| 3 | `mjINT_IMPLICITFAST` | faster implicit variant |

## Solver Values

| Value | Name |
|-------|------|
| 0 | PGS (Projected Gauss-Seidel) |
| 1 | CG (Conjugate Gradient) |
| 2 | Newton |

See also [[MjStatistic]] for model statistics.
