---
type: api_reference
tags: [mujoco, python, simulation, diagnostics, debugging]
source: https://mujoco.readthedocs.io/en/stable/programming/simulation.html
date_ingested: 2026-04-13
---
# mujoco_diagnostics

MuJoCo's built-in diagnostics: warnings, profiling timers, solver monitoring, forward/inverse dynamics quality check, and memory usage tracking.

## Warnings

MuJoCo registers warnings in `data.warning` without stopping simulation. Triggered by numerical inaccuracies or suspicious model states.

```python
# data.warning is an array of mjWarningStat, one per mjtWarning type
for wtype in mujoco.mjtWarning:
    stat = data.warning[wtype]
    if stat.number > 0:
        print(f"Warning {wtype.name}: triggered {stat.number} times, last info: {stat.lastinfo}")
```

Counters are cleared on reset. The first trigger of each type also prints via `mju_warning`.

## Timers (Profiling)

MuJoCo accumulates timing per top-level function in `data.timer`. Requires installing the time callback `mjcb_time`.

```python
import mujoco, time

# Install timer callback
mujoco.set_mjcb_time(time.perf_counter)

# After simulation:
step_stat = data.timer[mujoco.mjtTimer.mjTIMER_STEP]
avg_ms = 1000 * step_stat.duration / max(1, step_stat.number)
print(f"Average mj_step time: {avg_ms:.3f} ms")
```

Available timers (in `mjtTimer`): `mjTIMER_STEP`, `mjTIMER_FORWARD`, `mjTIMER_INVERSE`, `mjTIMER_POSITION`, `mjTIMER_VELOCITY`, `mjTIMER_ACTUATION`, `mjTIMER_ACCELERATION`, `mjTIMER_CONSTRAINT`, plus collision and other sub-timers.

## Solver Diagnostics

```python
# Number of solver iterations on the last step
print(data.solver_niter)

# Per-iteration solver state (mjSolverStat structs)
for i in range(data.solver_niter):
    print(data.solver[i].improvement, data.solver[i].gradient)
```

`solver_niter` is usually well below the maximum because the solver has early-termination tolerance. If it hits the maximum, consider increasing `model.opt.iterations` or tightening the tolerance.

## Forward/Inverse Dynamics Quality (`fwdinv`)

When `model.opt.enableflags` has the `fwdinv` flag set, MuJoCo computes:

```
data.solver_fwdinv[0]  # L2 norm of joint-force discrepancy (fwd vs inv)
data.solver_fwdinv[1]  # L2 norm of constraint-force discrepancy
```

These quantify solver convergence quality. Near-zero values mean the forward dynamics solution is exact (matching analytical inverse dynamics).

Enable in Python:
```python
model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_FWDINV
```

## Energy Monitoring

```python
model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
mujoco.mj_step(model, data)

kinetic_energy   = data.energy[0]
potential_energy = data.energy[1]
total            = data.energy[0] + data.energy[1]
```

For fully conservative systems (no contacts, no actuators, no dissipation), energy should be constant. Temporal drift in total energy indicates integration inaccuracy — use RK4 integrator for better conservation.

## Memory Diagnostics

```python
# Maximum arena memory used since last reset (bytes)
print(data.maxuse_arena)

# Use to tune <size memory="..."/> in MJCF
# Strategy: set memory large, run typical simulations, then reduce to maxuse_arena
```

If `maxuse_arena` exceeds the `memory` attribute, MuJoCo triggers a terminal error. Monitor this field during development.

## Error and Warning Callbacks

```python
# Install custom error handler (must not return)
def my_error(msg: str) -> None:
    raise RuntimeError(f"MuJoCo error: {msg}")
mujoco.set_mju_user_error(my_error)

# Install custom warning handler
def my_warning(msg: str) -> None:
    logging.warning(f"MuJoCo warning: {msg}")
mujoco.set_mju_user_warning(my_warning)
```

Errors are also logged to `MUJOCO_LOG.TXT` in the program directory.

## Notes

- `mj_loadXML` and `mj_loadModel` return `None` on failure — they do not call `mju_error`. Check return value.
- The `mjtWarning.mjWARN_INERTIA` warning fires when inertia values are degenerate — a common issue in evolved bodies.
- Timer durations accumulate across steps and are only cleared on `mj_resetData`. Compute per-step averages by dividing `duration` by `number`.

## See Also

[[mujoco_simulation_functions]], [[MjOption]], [[MjData]]
