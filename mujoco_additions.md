# Porting MuJoCo-Drones-Gym Components into ARIEL

Instructions for a coding agent (Claude Code) working inside the `ariel`
repository at `/home/user/Desktop/EvoDevo/ariel/`. Goal: port three
self-contained components from the sibling project
`/home/user/Desktop/EvoDevo/MuJoCo-drones-gym/` (referred to below as
`DRONES_GYM`) into ARIEL.

The three components, in recommended implementation order:

1. **Domain Randomization wrapper** — easiest, fully self-contained.
2. **Obstacle generation** — XML-injection based, also self-contained.
3. **Cascaded PID controller** — most coupled; needs an adapter layer.

Read the entire document before starting. Each component has a
"Verify before using" subsection — execute those checks first.

---

## Project conventions (ARIEL)

These come from existing memory and `CLAUDE.md` for this repo. Follow
them or your code will fail in CI / be rejected:

- Use `uv run filename.py` to run scripts, **never** bare `python`.
- The current working branch is `ec_prototype`, not `main`. Published
  ARIEL docs describe `main` — when in doubt, read source directly.
- `airevolve` lives at `src/airevolve/` and is installed via
  `uv pip install -e src/airevolve`.
- Lynx-arm joint damping must be `2.0` (not `0.1`) — relevant if any
  ported component touches joint definitions.
- **Never** add `from __future__ import annotations` in files that
  define `@EAOperation`.
- Before using any attribute on an ARIEL/MuJoCo object that isn't
  already used elsewhere in the codebase, verify with `dir()` or
  `help()`. Don't guess names.

## Where to put new code

The three components are **library code**, not EA operations. Suggested
locations (verify these paths exist before writing — adjust if the
layout differs):

```
src/ariel/control/pid.py             # component 3
src/ariel/wrappers/domain_random.py  # component 1
src/ariel/wrappers/obstacles.py      # component 2
src/ariel/wrappers/__init__.py       # exports
```

If `src/ariel/wrappers/` or `src/ariel/control/` doesn't exist, create
it with an `__init__.py`. Mirror the package style already used in
`src/ariel/`.

---

## 1. Domain Randomization wrapper

### Source

`DRONES_GYM/multi_drone_mujoco/wrappers/__init__.py` — defines
`DomainRandomizationConfig` (dataclass) and `DomainRandomizationWrapper`
(gymnasium.Wrapper subclass). Single file, ~195 lines, no internal deps
beyond `numpy` and `gymnasium`.

### What it does

Wraps a gym env and on each `reset()`:

- Multiplicatively perturbs the underlying env's physics constants
  (mass `M`, inertia, thrust coefficient `KF`, torque coefficient `KM`,
  arm length `L`, max RPM).
- Recomputes derived quantities (`HOVER_RPM`, `MAX_THRUST`, `WEIGHT`).
- Samples a per-episode actuator delay (in control steps) and motor
  time constant (first-order lag).
- Samples constant per-episode IMU biases.
- Optionally perturbs initial position and velocity.

On `step()`: applies the first-order motor lag filter to the action
before passing it to the wrapped env.

### Verify before using

Run these in an `uv run python -c '...'` session to confirm the ARIEL
env exposes the attributes the wrapper relies on. The wrapper currently
reads/writes:

```
env.M, env.KF, env.KM, env.L, env.MAX_RPM
env.HOVER_RPM, env.MAX_THRUST, env.WEIGHT
env.G, env.NUM_DRONES, env.CTRL_TIMESTEP
env.pos[i], env.vel[i]
```

ARIEL morphologies are **not** drones with these exact constants. So
**do not port the wrapper as-is** — instead, abstract it:

1. Define a small `RandomizableEnv` protocol (or duck-typed) that lists
   the float/array attributes the wrapper is allowed to perturb.
2. Make `DomainRandomizationConfig` carry a `dict[str, tuple[float, float]]`
   of attribute name → (lo, hi) multiplicative ranges, rather than
   hard-coded `mass_range`, `kf_range`, etc.
3. Keep the actuator-delay and first-order-lag logic — those are
   generally useful for any robot.
4. Drop the IMU bias logic for the first pass (drone-specific).

### Tasks

- [ ] Read the source file end-to-end.
- [ ] Create `src/ariel/wrappers/domain_random.py` with the
  generalised version described above.
- [ ] Write a unit test that wraps a trivial ARIEL env and checks
  perturbed attributes fall inside the requested ranges across 100
  resets with a fixed seed.
- [ ] Run `uv run pytest path/to/test_domain_random.py -v`.

---

## 2. Obstacle generation

### Source

`DRONES_GYM/multi_drone_mujoco/wrappers/obstacles.py` — ~345 lines.

Defines:

- `ObstacleType` enum (FOREST, URBAN, INDOOR, RANDOM, GATES, CUSTOM)
- `Obstacle` dataclass (`geom_type`, `position`, `size`, `rgba`, `euler`)
- `ObstacleConfig` dataclass (arena bounds, count, spacing, safe zones)
- `generate_obstacles(config) -> list[Obstacle]`
- `obstacles_to_xml(obstacles) -> str` — produces MuJoCo `<geom>` XML
- Private `_generate_forest`, `_generate_urban`, `_generate_indoor`,
  `_generate_random`, `_generate_gates`, and `_try_place` (spacing /
  safe-zone check).

### Why it's portable

It is pure Python: produces MJCF XML fragments. Zero dependency on
the drone env. You inject the returned string into the `<worldbody>`
of whatever XML ARIEL is already building.

### Verify before using

ARIEL builds its world XML somewhere — find that path first. Likely
candidates to grep for:

```bash
grep -rn "worldbody" src/ariel/ | head
grep -rn "compile.*MjModel\|from_xml_string" src/ariel/ | head
```

Identify the function that produces the world string. The injection
point is between `<worldbody>` and the closing `</worldbody>` tag,
**after** the existing morphology body.

### Tasks

- [ ] Copy `obstacles.py` to `src/ariel/wrappers/obstacles.py` verbatim
  as a starting point.
- [ ] Remove anything drone-specific (none visible — confirm).
- [ ] Locate ARIEL's world-XML builder. Add a parameter
  `obstacles: list[Obstacle] | None = None` and inject
  `obstacles_to_xml(obstacles)` before `</worldbody>`.
- [ ] Add an example script `examples/obstacles_demo.py` that
  generates a FOREST and renders one frame (uses the existing ARIEL
  viewer — see `project_phase15_complete` memory).
- [ ] Sanity test: `uv run python examples/obstacles_demo.py` should
  produce a window (or saved PNG) with visible cylinders.

---

## 3. Cascaded PID controller

### Source

- `DRONES_GYM/multi_drone_mujoco/control/pid_control.py` (~226 lines)
- `DRONES_GYM/multi_drone_mujoco/control/dsl_pid_control.py` (~104 lines,
  enhanced version with anti-windup)

### What it does

A cascaded position → attitude → motor-mixer PID for a quadrotor:

- Position PID outputs desired acceleration `target_acc`.
- Gravity is added; desired thrust = `M * |target_acc|`.
- Desired roll/pitch/yaw derived from thrust direction.
- Attitude PID outputs body torques.
- `_thrustTorquesToRPM` solves a 4×4 linear system (`A @ rpm^2 = b`)
  to back out per-motor RPMs in X-configuration, clamps to motor
  envelope.

### Why this one is harder

It assumes a **quadrotor** with four motors in X-configuration and
specific physical constants (`M`, `KF`, `KM`, `L`, `MAX_RPM`,
`GRAVITY`). ARIEL morphologies are evolved — variable arm count, geometry,
actuators. So:

- The high-level position-loop logic (PID on position → desired
  acceleration → desired thrust direction) is **portable as a
  reference baseline** for any flying ARIEL morphology with a
  well-defined "body up" axis.
- The attitude PID is similarly reusable if you can compute roll/pitch/yaw
  from the morphology's body quaternion.
- The motor mixer (`_thrustTorquesToRPM`) is **specific to 4-rotor X**.
  Do **not** port it as-is. Replace it with a per-morphology
  mixer/allocator, or stop the port at "produce desired body wrench
  (thrust + 3 torques)" and let the morphology decide allocation.

### Verify before using

Before writing, determine:

```bash
grep -rn "quat\|getQuaternion\|body_quat" src/ariel/ | head
grep -rn "ctrl_timestep\|CTRL_TIMESTEP" src/ariel/ | head
```

Confirm:

1. What ARIEL calls the control timestep.
2. How ARIEL exposes body position, velocity, quaternion, angular
   velocity for a morphology.
3. Whether there's already a "motor allocator" or similar.

### Tasks

- [ ] Read both PID source files end-to-end.
- [ ] Create `src/ariel/control/pid.py` containing:
  - `CascadedPID` class with `compute_wrench(dt, pos, quat, vel,
    ang_vel, target_pos, target_yaw)` returning
    `(thrust_scalar, body_torques_xyz, pos_error, yaw_error)`.
  - All gains exposed as constructor args (don't hard-code the
    Crazyflie numbers — leave them as defaults but documented).
  - A `reset()` that clears integral accumulators.
  - **No motor mixer.** Mixing is the morphology's responsibility.
- [ ] Add a quick smoke test: drive `CascadedPID` with a synthetic
  hovering state vector and confirm thrust ≈ `M * G` and torques are
  small.
- [ ] Document in a docstring that this is a baseline reference
  controller, not a morphology-aware optimal controller.

---

## Working order & PR strategy

Do the three components as **separate commits / PRs** in this order:

1. PR 1: obstacles (lowest risk, no API changes).
2. PR 2: domain randomization (small API additions, no breaking changes).
3. PR 3: PID baseline (touches the control layer).

Each PR should include the unit test described in its section. Run
`uv run pytest` at the repo root before opening the PR.

## When in doubt

- The DRONES_GYM source files are the authoritative spec — re-read
  them rather than guessing.
- If an ARIEL attribute name is wrong, run `dir(env)` in
  `uv run python -i ...` and confirm before writing more code.
- If a port requires invasive changes to an ARIEL base class, **stop
  and ask the user** rather than refactoring around it.
