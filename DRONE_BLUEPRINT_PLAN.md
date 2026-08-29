# Drone Blueprint Plan — ARIEL for the SPEAR Drone Project

**Audience:** SPEAR pre-meeting (Friday 9:30) and consortium integration event.
**Author:** Aron (VUA), drone-evolution port in ARIEL.

---

## 1. Value proposition

ARIEL already proves a **blueprint-mediated** pipeline for modular robots: any
genotype (tree, CPPN, vector) is decoded to a common intermediate
representation, which a single phenotype builder turns into a MuJoCo robot
(`src/ariel/body_phenotypes/robogen_lite/decoders/_blueprint.py`). The same
architecture is what the drone consortium needs: many partners, many
encodings, divergent simulators (MuJoCo, Aerial Gym, Isaac Lab). A shared
**Drone Blueprint** lets each group keep their genotype and their preferred
backend while reusing the rest of the pipeline.

## 2. Current standing

- Jed Muff's `airevolve` is ported into ARIEL under
  `src/ariel/body_phenotypes/drone/` and `src/ariel/ec/drone/`.
- Three encodings already coexist: `spherical_angular`, `cartesian_euler`,
  `cppn_neat` (plus a `hybrid_cppn`). Each currently wires *directly* to a
  MuJoCo phenotype — bypassing the blueprint layer that is ARIEL's actual
  contribution.
- MuJoCo-only sim. No USD/Isaac Lab path yet.
- Examples in `examples/d_drones/` (evolution, NEAT, Lee controller, RL
  figure-8, STL export, repair) demonstrate end-to-end runs.

**Gap:** drone encodings can't yet share a downstream pipeline, and there
is no architectural seam for adding an Isaac Lab backend.

## 3. Proposed Drone Blueprint

An **ARIEL-native abstract IR**: a typed tree (NetworkX `DiGraph`, JSON
serialisable) mirroring `robogen_lite`'s blueprint, **lifted into continuous
space**. Drones aren't cubes-on-a-grid; nodes therefore carry continuous
SE(3) transforms and physical parameters rather than enum'd faces.

### v1 node types

| Node | Carries |
| --- | --- |
| `CorePlate` | mass, inertia tensor, mounting frame, geometry primitive (disc/box) |
| `Arm` | length, SE(3) attachment pose to parent, material/inertia |
| `Motor` | pose on parent `Arm`, thrust/torque coefficients, spin direction (CW/CCW), max rpm |
| `Rotor` | radius, pitch, blade count, drag coefficient |
| `Sensor` | type (IMU/camera/range), pose, intrinsic params |

Edges encode parent → child attachment. The root is always a `CorePlate`.

### Example (sketch)

```json
{
  "nodes": [
    {"id": 0, "type": "CorePlate", "mass": 0.4, "geometry": {"shape": "disc", "r": 0.05}},
    {"id": 1, "type": "Arm", "length": 0.18, "pose": {"xyz": [0.05,0,0], "rpy": [0,0,0.785]}},
    {"id": 2, "type": "Motor", "pose": {"xyz": [0.18,0,0], "rpy": [0,0,0]}, "spin": "CW", "kf": 1.1e-5, "km": 1.9e-7},
    {"id": 3, "type": "Rotor", "radius": 0.0635, "pitch": 0.045, "blades": 2}
  ],
  "edges": [[0,1], [1,2], [2,3]]
}
```

### Decoders (genotype → blueprint)

- **Spherical-angular → Blueprint** *(v1, must-have)* — each `(mag, az, el,
  motor_az, motor_pitch, spin)` arm row becomes `Arm → Motor → Rotor`.
- **Cartesian-Euler → Blueprint** *(v1, must-have)* — proves portability:
  two independent encodings producing the *same* blueprint format.
- **CPPN-NEAT → Blueprint** *(v1 stretch)* — generative mapping; sample the
  CPPN over a polar grid, discretise into N arms, then emit the same tree.
  Non-trivial — flag as the v1 stretch goal, with a clean v2 fallback.

### Backends (blueprint → phenotype)

- **MJCF / `mjSpec`** *(v1)* — replaces today's direct genome→mjSpec path
  in `phenotype_assembly/generator.py`. Single source of truth for the
  drone body in MuJoCo.
- **STL / STEP** *(v1, free)* — `generate_stl_files` already exists; rewire
  it to consume a Blueprint instead of a `SphericalAngularDroneGenomeHandler`.
- **USD / Isaac Lab** *(designed-for, deferred)* — sketched interface:
  ```python
  def blueprint_to_usd(bp: Blueprint, out: Path) -> Path: ...
  # → loaded by isaaclab.sim.UsdFileCfg(usd_path=...)
  ```
  Each node maps to a USD `Xform` prim with appropriate `RigidBodyAPI` /
  `ArticulationRootAPI`; ImplicitActuatorCfg wiring lives in a small
  per-backend adapter, not the blueprint itself. Implementation deferred
  pending consortium agreement on Isaac Lab adoption.

## 4. Refactor: align drone subsystem with ARIEL conventions

**Current layout** under `src/ariel/body_phenotypes/drone/` (shipped):

```
body_phenotypes/drone/
├── blueprint.py             # DroneBlueprint + Pose, *Node dataclasses, JSON I/O
├── decoders.py              # spherical_angular_to_blueprint, cartesian_euler_to_blueprint
├── backends.py              # blueprint_to_propellers, blueprint_to_mjspec, blueprint_to_usd (stub)
└── phenotype_assembly/      # legacy airevolve port (STL/STEP, generator.py) — not yet rewired
```

**Target layout** (still aspirational; matches `robogen_lite/`):

```
body_phenotypes/drone/
├── blueprint.py
├── modules/                 # CorePlate, Arm, Motor, Rotor, Sensor as separate modules
├── decoders/                # one file per encoding (spherical, cartesian, cppn-neat)
├── backends/                # one file per backend (mjspec, cad, usd)
└── prebuilt_drones/         # quad / hex / x8 reference blueprints
```

`ec/drone/` keeps the EA-side operators (mutation, crossover, evaluation,
inspection); they continue to operate on **genomes**, but downstream calls
now go through `DroneBlueprint` rather than the phenotype generator directly.

## 5. Roadmap

**Phase 1 — Blueprint scaffolding. ✅ Done.**
- `src/ariel/body_phenotypes/drone/blueprint.py` — `DroneBlueprint`
  (NetworkX `DiGraph`, typed node dataclasses, JSON save/load, summary).
- `src/ariel/body_phenotypes/drone/decoders.py` — `spherical_angular_to_blueprint`,
  `cartesian_euler_to_blueprint`.
- `src/ariel/body_phenotypes/drone/backends.py` —
  - `blueprint_to_propellers(convention="z_up"|"ned")` →
    `DroneSimulator` / `DroneConfiguration` (Python physics stack).
  - `blueprint_to_mjspec(...)` → `mujoco.MjSpec` with one welded body per
    Arm, motor cylinders + thin rotor discs, and one
    `mjTRN_SITE` thrust actuator per rotor
    (`ctrl ∈ [0,1] → F ∈ [0, max_thrust]` along the rotor's spin axis).
    Physical parameters (`arm.mass`, `arm.inertia_diag`, `motor.mass`,
    `motor.radius`, `motor.thickness`) are read from the blueprint nodes
    themselves; only `core_mass_override` survives as a tuning kwarg.
  - `blueprint_to_urdf` stub on `blueprint-to-urdf` branch; intermediate
    for the Isaac Lab path via `isaaclab.sim.converters.UrdfConverter`.
  - `blueprint_to_usd` still a stub (direct-USD path deferred behind URDF).
- Examples:
  - `examples/d_drones/11_blueprint_demo.py` — two encodings → same
    blueprint format → identical `DroneConfiguration` mass/inertia.
  - `examples/d_drones/12_visualize_from_blueprint.py` — blueprint flown
    through a circle course with the Lee controller (Python sim) +
    `sameAxisAnimation` MP4.
  - `examples/d_drones/13_mujoco_blueprint_quad.py` — blueprint compiled
    to MuJoCo; open-loop hover-thrust quad, MP4 or `mujoco.viewer` (`--view`).
  - `examples/d_drones/14_mujoco_lee_figure8.py` — figure-8 flown by Lee
    (NED Python sim, validated path), then kinematically replayed inside
    MuJoCo using the blueprint's MJCF body. NED→ENU bridge: position
    `z` negated, attitude `qy` negated.

**Phase 2 — Multi-encoding portability. Partial.**
- Two decoders shipped (spherical-angular, Cartesian-Euler) and validated
  through both backends. CPPN-NEAT decoder still TODO.
- STL/STEP backend not yet rewired through Blueprint (still consumes the
  legacy `SphericalAngularDroneGenomeHandler` directly in
  `phenotype_assembly/generator.py`).
- Open issue surfaced while building example 14: the Lee controller's
  ENU motor-allocation path has a sign error
  (`_wrench_to_motor_commands` applies `-self.Bf[2:3,:]` unconditionally,
  which is correct for NED but inverts the relationship for ENU and
  drives `w_cmd` to the motor floor). NED is validated; ENU needs the
  upstream fix before closed-loop MuJoCo Lee control is possible without
  the bridge.

**Phase 3 — Isaac Lab spike. ✅ Landed end-to-end for rigid drones (branch `blueprint-to-urdf`, with the pluggable architecture on `pluggable-simulator`).**

The full Blueprint → URDF → USD → DirectRLEnv → `rl_games` PPO path
runs cleanly in the unified `ariel-isaaclab-train` conda env as of
2026-05-27. All seven Option A acceptance boxes (see Phase 2.5
section in "Still pending" below) tick empirically.
Approach: Blueprint → URDF → USD via Isaac Lab's `UrdfConverter`, not a
direct `blueprint_to_usd`. Direct `blueprint_to_usd` remains stubbed and
now sits *behind* URDF in the roadmap.

What's landed (in order of commit):

1. `Add blueprint_to_urdf stub.` (also on `drones`) — sketches the
   planned URDF backend alongside the existing `blueprint_to_usd` stub.
2. `Derive arm and motor physical params from blueprint nodes.` —
   schema refactor: cross-section sub-objects (`CylindricalCrossSection`
   / `HollowTubeCrossSection` / `RectangularCrossSection`) and derived
   `mass` / `inertia_diag` properties on `ArmNode` and `MotorNode`,
   sourced from a `propsize`-keyed lookup in `PROPELLER_LIBRARY`
   (extended with `motor_radius` / `motor_thickness` per entry, values
   from the APC propeller-motor pairing table). Default arm cross-
   section is `HollowTube(outer=4mm, inner=3mm)` which with
   `density=1500 kg/m³` reproduces existing `BEAM_DENSITY = 0.034 kg/m`.
3. `Implement blueprint_to_urdf for rigid-drone Isaac Lab path.` —
   walks the Blueprint tree, emits one `<link>` per CorePlate/Arm/Motor
   (diagonal `<inertial>` from node-derived properties), one
   `<joint type="fixed">` per parent-child edge, dispatches arm
   geometry on cross-section type (`<cylinder>` for solid/hollow-tube,
   `<box>` for rectangular). No actuators emitted (Isaac Lab applies
   thrust at runtime via force on the motor link's local +Z).
4. `Add Blueprint → URDF → USD pipeline scripts.` — split across two
   envs because ariel needs Python 3.12 (PEP 695 `type` statements,
   25+ files) and the local isaaclab conda env runs Python 3.11. URDF
   file is the boundary:
   * `scripts/blueprint_to_urdf.py` — ariel env, builds blueprint,
     writes URDF.
   * `scripts/urdf_to_usd.py` — isaaclab env, takes URDF, writes USD
     via `isaaclab.sim.converters.UrdfConverter` (with `target_type=
     "none"` + zero PD gains for the all-fixed-joints case).
   End-to-end smoke test: quad URDF → USD via the standard Isaac Lab
   layered layout (`quad.usd` + `configuration/{quad_base,
   quad_physics, quad_robot, quad_sensor}.usd`).
5. `Add example 16: blueprint_to_urdf demo with cross-section
   dispatch.` — `examples/d_drones/16_blueprint_to_urdf.py`. Default
   produces a HollowTube X-quad URDF; `--rect-arms` flips arms to
   solid boxes and verifies the URDF box-geometry path (both variants
   round-trip cleanly to USD via the converter).

**Follow-on branch `python-311-compat` (off `blueprint-to-urdf`):**

6. `Make ariel importable on Python 3.11.` — dropped all PEP 695
   syntax (31 type-alias statements + one generic class) and
   downgraded `requires-python` to `>=3.11`. Motivation: unify ariel
   and Isaac Lab into one conda env (Isaac Sim 5.1 pins Python 3.11).
   After `pip install -e ariel` into the existing `isaaclab` conda
   env, the full Blueprint → URDF → USD pipeline runs in one
   process. See §6 entry 15 for details.

**Follow-on branch `pluggable-simulator` (off `python-311-compat`):**

7. `Add pluggable simulator backend Protocol + tutorial.` — defines
   the `BlueprintGateEnv` Protocol so ariel's EA+PPO loop is
   simulator-backend-agnostic. Ships `NumpyBlueprintGateEnv` (thin
   shim over the existing `DroneGateEnv` declared as a Protocol
   implementation) and an `IsaacLabBlueprintGateEnv` stub showing
   the next plug-in target. Tutorial at
   `tutorials/pluggable_simulator/` walks a collaborator through the
   architecture and the "add your own simulator" recipe. Includes a
   fix to a pre-existing `DroneGateEnv` bug where `self.seed`
   shadowed the inherited `VecEnv.seed()` method (broke
   stable-baselines3's `set_random_seed`). See §6 entry 17.

8. `Implement IsaacLabBlueprintHoverEnv (Phase 2).` —
   `src/ariel/simulation/tasks/isaaclab_hover_env.py` adapted from
   Isaac Lab's reference `QuadcopterEnv`; spawns N parallel drones
   from a Blueprint-derived USD, applies a thrust+moment wrench at
   the root body each step, computes `-distance × step_dt` rewards.
   `train.py --simulator isaaclab` exercises the full path
   end-to-end via a random-action stepping loop (72 steps in 1.6 s
   on 16 envs). Real PPO training via `rl_games.torch_runner.Runner`
   is wired through `make_rl_games_agent_cfg` but currently calls a
   random-action loop instead, gated on Phase 2.5. Architectural
   choice change from Phase 1: the Isaac Lab backend follows the
   `DirectRLEnv` shape rather than the `BlueprintGateEnv` Protocol —
   "two-Protocols-one-trainer" as picked during planning. See §6
   entry 17.

**Still pending — to pick up next session:**

* **Phase 2.5 of pluggable simulator: stabilize the Isaac Lab
  training stack, then run PPO.** The `rl_games` Runner trips on a
  transitive import chain (Isaac Sim's bundled
  `torch.utils.tensorboard.writer` → lazy `tensorflow` → `jax` →
  `numpy.dtypes.StringDType`) because Isaac Sim's `pip_prebundle/`
  ships an older numpy than the conda env's numpy 2.4.6. We should
  stop treating the three remediations as peers; the recommended
  path is a dedicated Isaac-Lab-specific training env pinned to the
  Isaac Sim / Isaac Lab stack, with ariel installed into it without
  upgrading the simulator-owned binaries.

  **Concrete packaging / env plan:**
  1. Create a clean `ariel-isaaclab-train` conda env from Isaac
    Lab's published env / bundled versions first, not from
    ariel's top-level `pyproject.toml`.
  2. Install Isaac Lab / Isaac Sim into that env, verify the stock
    `rl_games` examples run, then install ariel editable with
    `pip install -e . --no-deps` so pip does not replace the
    simulator-owned torch / gymnasium / numpy stack.
  3. Trim ariel's top-level dependencies down to simulator-agnostic
    requirements. In particular, move `torch`, `torchvision`, and
    `gymnasium` out of `[project.dependencies]`; do not let the
    base package force versions of libraries Isaac Lab already owns.
  4. Pin shared binary deps conservatively in the base package:
    `numpy>=1.26,<2` is the target unless a code path is proven to
    require a numpy-2-only API. The current `numpy>=2.0.2` default
    is the wrong bias for this integration.
  5. If we keep an optional `isaaclab` extra, it should be a
    glue-only extra, not a second owner of the low-level runtime.
    That means: no `torch`, no `torchvision`, no `gymnasium`, and
    no attempt to pip-install Isaac Lab itself from ariel. At most,
    it carries tiny compatibility pins that are not already owned by
    Isaac Lab's env.
  6. Treat shadowing torch on `PYTHONPATH` as a diagnostic hack, not
    a supported solution. Treat `jax` / `tensorboard` downgrades as
    a probe to confirm the import chain, not the primary fix.
  7. Phase-2.5 exit criterion: one reproducible env-creation recipe,
    one documented install command, and one short `rl_games` PPO
    smoke run from `tutorials/pluggable_simulator/train.py`.

  **Option A execution recipe (copy/paste):**

  ```bash
  # 0) Paths (adjust if your checkout differs)
  export ARIEL_ROOT="$HOME/Documents/sandbox/ariel"
  export ISAACLAB_ROOT="$HOME/Documents/sandbox/IsaacLab"
  export ENV_NAME="ariel-isaaclab-train"

  # 1) Create a clean env from a *vendored* copy of Isaac Lab's pinned stack.
  #    Vendoring the env spec inside ariel (rather than reading
  #    $ISAACLAB_ROOT/environment.yml directly) makes this recipe
  #    reproducible across upstream Isaac Lab evolution — see the SHA
  #    documented in the file header.
  conda env remove -n "$ENV_NAME" -y || true
  conda env create -n "$ENV_NAME" \
      -f "$ARIEL_ROOT/tutorials/pluggable_simulator/isaaclab-env.yml"
  conda activate "$ENV_NAME"

  # 2) Install Isaac Lab into that env (owns torch/gymnasium/numpy ABI layer)
  cd "$ISAACLAB_ROOT"
  ./isaaclab.sh -i

  # 3) Sanity-check Isaac Lab python path first
  ./isaaclab.sh -p -c "import isaaclab; print('isaaclab import OK')"

  # 3b) Source Isaac Sim's conda-env setup so bare `python` invocations
  #     that follow can find `isaacsim` / `omni.*` / `pxr` on PYTHONPATH.
  #     `./isaaclab.sh -p` (above) handles this internally; bare `python`
  #     does not. The recipe uses bare `python` from step 7 onward.
  source "$ISAACLAB_ROOT/_isaac_sim/setup_conda_env.sh"

  # 4) Snapshot simulator-owned binary versions BEFORE ariel install.
  #    The next step (pip install -e . --no-deps) MUST leave these
  #    untouched; the post-install diff confirms it.
  pip list --format=freeze \
      | grep -iE "^(torch|torchvision|gymnasium|numpy)==" \
      | sort > /tmp/ariel_phase25_binaries_before.txt
  cat /tmp/ariel_phase25_binaries_before.txt

  # 5) Install ariel WITHOUT dependency resolution side effects.
  #    --no-deps is the load-bearing flag: pip MUST NOT replace the
  #    simulator-owned torch/gymnasium/numpy stack here.
  cd "$ARIEL_ROOT"
  pip install -e . --no-deps

  # 6) Guardrail check: verify ariel install did not bump binaries.
  #    Expected output: "BINARIES UNCHANGED" with empty diff.
  pip list --format=freeze \
      | grep -iE "^(torch|torchvision|gymnasium|numpy)==" \
      | sort > /tmp/ariel_phase25_binaries_after.txt
  if diff -u /tmp/ariel_phase25_binaries_before.txt \
              /tmp/ariel_phase25_binaries_after.txt; then
      echo "BINARIES UNCHANGED ✓"
  else
      echo "ERROR: simulator-owned binaries were bumped by ariel install" >&2
      echo "       inspect pyproject.toml [project.dependencies] for leaks" >&2
      exit 1
  fi

  # 6b) Install ariel's pure-Python deps that --no-deps skipped, but
  #     constrain the simulator-owned binaries against accidental
  #     upgrade. The before-snapshot from step 4 doubles as a pip
  #     constraints file. Skip evotorch and mujoco-mjx — they bring
  #     torch / jax / numpy deps that fight Isaac Lab's stack.
  pip install --constraint /tmp/ariel_phase25_binaries_before.txt \
      "networkx>=3.2.1" \
      "rich>=14.1.0" \
      "pydantic>=2.11.9" \
      "pydantic-settings>=2.10.1" \
      "sqlalchemy>=2.0.43" \
      "sqlmodel>=0.0.25" \
      "numpy-quaternion>=2023.0.3" \
      "matplotlib>=3.9.4" \
      "mujoco>=3.3.6"

  # 7) Quick import smoke for ariel (Blueprint chain — what the Isaac
  #    Lab path actually uses). `DroneGateEnv` would test the NumPy
  #    backend's chain, which transitively needs EA orchestration deps
  #    we intentionally don't pull in here.
  python -c "
  from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
  from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
  from ariel.body_phenotypes.drone.backends import blueprint_to_urdf
  print('ariel Blueprint chain: OK')
  "

  # 8) Isaac Lab backend smoke (env stepping; PPO training still Phase 2.5 gate)
  python tutorials/pluggable_simulator/train.py \
      --simulator isaaclab \
      --headless \
      --num-envs 16 \
      --max-iterations 3
  ```

  **Option A acceptance checklist** — *all boxes ticked on
  2026-05-27 during pluggable-simulator @ 7dfaea4 + lazy-init refactor
  in progress; execution log immediately below:*

  - [x] New clean env created from the vendored
        `tutorials/pluggable_simulator/isaaclab-env.yml` (NOT directly
        from upstream `$ISAACLAB_ROOT/environment.yml`) with no manual
        pin surgery.
  - [x] Isaac Lab import sanity check passes before ariel is installed.
  - [x] Pre-install binary snapshot captured to
        `/tmp/ariel_phase25_binaries_before.txt`. Captured values:
        `torch==2.7.0+cu128`, `torchvision==0.22.0+cu128`,
        `gymnasium==1.2.1`, `numpy==1.26.4`.
  - [x] `pip install -e . --no-deps` completes and does not replace
        simulator-owned torch/gymnasium/numpy (guardrail diff is
        empty — `BINARIES UNCHANGED ✓`).
  - [x] Blueprint chain import smoke passes in the Option A env
        (`from ariel.body_phenotypes.drone.{blueprint,decoders,backends}`).
  - [x] `train.py --simulator isaaclab --headless --max-iterations 3`
        completes successfully — 72 env-steps in 0.4 s @ 189 steps/sec,
        16 parallel Isaac Sim envs, mean reward `-0.0295`.
  - [x] One short `rl_games` PPO smoke run completes in this env.
        Executed on 2026-05-27 with
        `python tutorials/pluggable_simulator/train.py
        --simulator isaaclab --headless --num-envs 16
        --max-iterations 3` (the new default `--mode train` exercises
        the `rl_games.torch_runner.Runner` path). Result: 3 PPO
        epochs in 0.9 s, fps_total climbing 719 → 2787 → 3318 across
        epochs, ep3 reward `-1.2708529` (consistent with
        `-distance × step_dt` summed over a 24-step horizon × 16 envs
        with a freshly-initialized policy). Checkpoint saved to
        `runs/ariel_blueprint_hover_27-15-14-30/nn/`.
  - [x] Re-running the same commands in a fresh env reproduces the
        same result (caveat: the in-progress lazy-init refactor and
        the recipe-side additions documented here — sourcing
        `setup_conda_env.sh`, the safe-deps install with the
        constraint file — were needed to get to a clean run on the
        first pass; a fresh env now starts from a known-good doc).

  **Option B (diagnostic only): shadow bundled torch with env torch**

  **Purpose:** quickly test whether the `rl_games` failure is mainly an
  import-precedence issue (`pip_prebundle/torch` winning over the env's
  torch).

  **Why this is NOT a primary fix:** this changes import order rather than
  making the environment reproducible. It is sensitive to launch scripts,
  shell state, and path ordering.

  **Minimal runbook (for diagnosis):**
  1. Start from a known-good Option A env.
  2. Export `PYTHONPATH` so the env's site-packages come before Isaac Sim's
     bundled paths.
  3. Run the shortest failing `rl_games` smoke command.
  4. Capture `sys.path`, `torch.__file__`, and `numpy.__version__` in logs.

  **Stop criteria:**
  - If this makes PPO smoke pass once but the result is not reproducible in
    a fresh shell, treat as a confirmed diagnostic only.
  - If this still fails on the same tensorboard/jax path, abandon Option B
    and return to Option A.

  **Rollback:** unset `PYTHONPATH`, reopen shell, and re-run baseline Option A
  smoke to verify environment behavior is unchanged.

  **Option C (diagnostic fallback): targeted jax/tensorboard downgrade**

  **Purpose:** test whether the `numpy.dtypes.StringDType` failure is caused by
  a specific transitive version edge in `jax` / `tensorboard`.

  **Why this is NOT a primary fix:** package surgery in this subtree can hide
  deeper ABI mismatches and often drifts over time.

  **Minimal runbook (for diagnosis):**
  1. Clone Option A env to an isolated test env.
  2. Apply only the smallest jax/tensorboard version change needed for the
     failing import path.
  3. Re-run the same short `rl_games` smoke command.
  4. Record before/after versions for `jax`, `tensorboard`, `tensorflow`,
     `numpy`, and `torch`.

  **Stop criteria:**
  - If PPO smoke passes, treat this as evidence about the failure mechanism,
    not production config, until reproduced from a locked env spec.
  - If additional downgrades cascade beyond this subtree, stop and revert;
    this is no longer a targeted probe.

  **Rollback:** discard the test env and recreate from Option A recipe rather
  than attempting in-place undo.

  Independent of this, `rsl_rl` integration via
  `isaaclab_rl.rsl_rl` was a separate dead end during Phase 2 — the
  adapter sends kwargs (`optimizer`, `share_cnn_encoders`) that do
  not match the installed `rsl-rl-lib` 3.0.1 PPO signature (and 5.x
  has its own config shape mismatch). `rl_games` is the path we
  picked forward; `make_rl_games_agent_cfg` in
  `isaaclab_hover_env.py` is the matching config helper.
* **Compliant joints (URDF revolute + sidecar stiffness + USD
  `ariel:*` attrs).** Schema landed on `compliant-joint-schema`;
  the emission half is what's still pending. The two-layer pattern
  is documented in §6 entry 10 — zero-stiffness `revolute` joint in
  URDF, non-linear `τ(θ)` law stashed as sidecar attrs for a
  runtime controller.
* Pytest integration test for the Blueprint → URDF → USD pipeline
  (now feasible in-process under the unified env; just needs a
  `pytest.mark.isaaclab` skip-if-not-available guard).
* **~~Targeted dep pins in pyproject.toml~~ — implemented this
  session as a dependency-ownership split.**
  `pyproject.toml` now keeps simulator-agnostic requirements in
  `[project.dependencies]` and moves simulator-/workflow-specific
  stacks to extras (`rl-sb3`, `torch`, `viz`, `fabrication`,
  `experimental`). Base `numpy` is pinned conservatively
  (`>=1.26,<2`) and `rl-sb3` pins `gymnasium==1.2.1` to match
  Isaac Lab expectations. `README.md` now documents the
  extra-specific install commands and the Isaac-Lab-safe
  `pip install -e . --no-deps` path.

## 6. Design decisions (Phase 3 — URDF / USD backends)

Each entry: **decision** — *why*; alternatives considered.

1. **URDF as the intermediate to USD, not direct `pxr`.** URDF is plain
   XML, no new dependency, and verifiable without an Isaac Sim install;
   the Blueprint tree maps 1:1 to URDF `<link>`/`<joint>`. The
   soft_airframe project already exercises the URDF→USD half via
   `isaaclab.sim.converters.UrdfConverter` (see
   `soft_airframe_optimization/scripts/convert_xconfig_urdf.py`). Direct
   USD emission with `pxr` is feasible but harder to author and verify
   from outside an Isaac Lab environment; deferred behind the URDF route.

2. **Hybrid mass model: density-volumetric for arms, propsize lookup for
   motors, inertia always derived.** Arms have homogeneous geometry the
   EA can evolve (length, radius/width), so `mass = density × area ×
   length` is the right physical model. Motors are catalog parts whose
   mass correlates with `propsize`, not arbitrary cylinder volume, so a
   `propsize → mass` lookup matches reality and the existing
   `propeller_data.PROPELLER_LIBRARY` pattern (which already keys `k_f`,
   `k_m`, `wmax` on propsize). Inertia tensors are always derived from
   shape + mass — nobody hand-authors `Ixx/Iyy/Izz` on a genotype.
   Alternatives: store `mass` directly on every node (no shape↔mass
   coupling, lets EA evolve impossibly heavy 1m arms); density-only for
   motors (meaningless — motor isn't a homogeneous cylinder).

3. **Derived properties live on the Blueprint node dataclasses, not in
   backends.** `@property mass`, `@property inertia_diag` on `ArmNode` /
   `MotorNode`. Backends (MJCF, URDF, USD) just call `arm.mass`. Single
   source of truth; formulas tested once; backends can't drift.
   Alternatives: a helper module (extra indirection, no enforcement);
   per-backend duplication (drift risk).

4. **`CorePlateNode.mass` stays a primary field.** The core is a lumped
   controller + battery mass (cf. `propeller_data.CONTROLLER_MASS +
   BATTERY_MASS = 0.0568 kg` baseline; ariel's default `0.4 kg` is the
   larger ref build), not a homogeneous geometric solid. Deriving it
   from `density × disc volume` would be physically meaningless. Also
   keeps existing `decoders.py` `CorePlateNode(mass=core_mass)` call
   working unchanged.

5. **Cross-section is a tagged sub-object on `ArmNode`, not flat
   fields.** Three classes: `CylindricalCrossSection`,
   `HollowTubeCrossSection`, `RectangularCrossSection`. Each exposes
   `area` (property) and `principal_inertia(length, mass)` (method).
   `arm.mass` and `arm.inertia_diag` work uniformly regardless of shape.
   Adding a new section (e.g. hollow box, I-beam) is one new dataclass,
   no backend changes for mass/inertia. Alternatives: flat fields with
   a `shape` discriminator (half-empty fields are footguns); subclasses
   of `ArmNode` (forces backends to dispatch on more types).

6. **Default cross-section is `HollowTubeCrossSection(outer=0.004,
   inner=0.003)` with `density=1500 kg/m³`.** With these defaults,
   `mass per metre ≈ 0.033 kg/m`, matching ariel's existing
   `propeller_data.BEAM_DENSITY = 0.034 kg/m` (8mm OD / 6mm ID
   carbon-fiber tube). Switching to the volumetric model is
   behavior-preserving for the canonical small-drone build.
   Alternatives considered: solid cylinder with a hand-tuned tiny radius
   (awkward defaults); solid cylinder at 5 mm radius (~2× mass change
   on existing builds, rejected).

7. **Backends dispatch geometry emission on `isinstance(cross_section,
   …)`. The Blueprint module has no MuJoCo / URDF / `pxr` imports.**
   Cross-section classes carry physics formulas (area, inertia) but no
   backend-specific emission code. Each backend knows how to render
   each section type. Alternatives: cross-section classes carry
   `urdf_geometry()` / `mjcf_kwargs()` methods (couples Blueprint to
   every backend it'll ever serve; rejected).

8. **Motor visual dimensions (`motor_radius`, `motor_thickness`) added to
   `PROPELLER_LIBRARY`, sourced from the APC propeller-motor pairing
   table.** Outer-can radius plateaus at ≈ 14 mm for prop5+ (all use
   22mm-class stators — `2204`–`2212`); only motor *height* grows with
   propsize beyond that. EA-evolved `propsize` automatically picks the
   matching visual size. Alternatives: keep dims as `blueprint_to_mjspec`
   defaults (single propsize only); heuristic radius ∝ propsize (wrong
   physics — was the v1 rough estimate, replaced).

9. **MJCF arm visuals: capsule for cylindrical/hollow, box for
   rectangular. URDF will use cylinder/box (URDF has no capsule).**
   Visual proxies only — true inertia comes from
   `arm.inertia_diag`, not the geom-derived MuJoCo formula. Capsule is
   MuJoCo's `fromto` primitive for "rounded cylinder along an arbitrary
   axis" and is already the existing behavior for cylindrical arms.

10. **Compliant joints (deferred for v1) will use a two-layer pattern.**
    Emit a `revolute` joint with zero physics stiffness in URDF / MJCF /
    USD; store the non-linear `τ(θ)` law as sidecar custom attributes
    (proposed `ariel:*` namespace, mirroring soft_airframe's `morphy:*`
    on USD root prims); a runtime controller reads the params and
    applies the torque each step. Rationale: URDF / MJCF / USD only
    support linear torsional springs natively. The soft_airframe X-config
    drone already uses this exact pattern and it is proven.

11. **No actuators in URDF.** URDF has no first-class actuator
    primitive equivalent to MuJoCo's `mjTRN_SITE`. Isaac Lab applies
    thrust at runtime via Python by force-on-link; the URDF only needs
    to expose named motor link prims. This is simpler than MJCF's
    actuator-per-motor wiring, not harder.

12. **~~Pipeline split into two scripts across two Python envs~~ — superseded by entry 15.**
    Originally documented: ariel used PEP 695 `type X = …` syntax (25+
    files) requiring Python 3.12, while the local isaaclab conda env
    runs Python 3.11, so `scripts/blueprint_to_urdf.py` ran in the
    ariel env and `scripts/urdf_to_usd.py` in the isaaclab env with
    the URDF file as the boundary. **As of branch `python-311-compat`
    (commit `7f678f3`), this constraint is gone** — ariel is now 3.11-
    compatible and pip-installable into the isaaclab env. The two
    scripts still exist and are useful as separable building blocks,
    but they can both run in the single unified env.

13. **`scripts/urdf_to_usd.py` logs progress to stderr, not stdout.**
    Isaac Sim's launcher captures stdout (probably for its Carb
    logger), so `print(...)` from inside the script can be lost. We
    use a small `_log(msg)` that writes to `sys.stderr` and flushes,
    so the only "where did my prints go?" question is answered up
    front. Discovered by burning a debug cycle on it.

14. **`UrdfConverterCfg.joint_drive` is required even for all-fixed
    drones.** The configclass validator treats
    `joint_drive.gains.stiffness` as a required field. For our v1
    rigid drone we don't have any actuated joints, so we set
    `target_type="none"` with zero PD gains — the values are unused
    but the field has to be present to pass validation. (Same dance
    soft_airframe does in `convert_xconfig_urdf.py`.)

15. **ariel is now Python 3.11-compatible (branch `python-311-compat`).**

    **Motivation:** ARIEL must be simulator-agnostic, and the use case
    that forced this was *EA objective functions that simulate each
    morphology* (e.g., fitness = gates passed in N seconds). With
    thousands of individuals per evolution, the previous two-env
    subprocess split paid the Isaac Sim launch cost (~60–90 s) once
    per evaluation — catastrophic. The only practical way to host
    Isaac Lab inside the same conda env as ariel was to drop ariel's
    Python 3.12-only constructs, because Isaac Sim 5.1 ships its own
    Python 3.11 in `_isaac_sim/kit/python/` and NVIDIA pins it (we
    can't bring Isaac to 3.12, so ariel comes down to 3.11). With
    both in one process, EA workers can convert + simulate
    in-memory; per-individual launch overhead disappears.

    **Audit:** 31 PEP 695 `type X = ...` statements across 14 files
    plus one PEP 695 generic class (`class EAOperation[**P]:`); no
    3.12-only stdlib calls. Alternatives considered: (a) rewrite
    Isaac to support 3.12 — not actually controllable by us; (b)
    long-running Isaac Lab worker process talking to ariel over IPC
    — viable but ~hundreds of lines of plumbing and harder to debug;
    (c) keep MuJoCo as the only EA-loop simulator and use Isaac Lab
    only for top-K validation — a fine v1 default but doesn't give
    us GPU-parallel envs.

    **Rewrite:** each `type X = ...` → `X: TypeAlias = ...` (PEP 613);
    the generic class → `class EAOperation(Generic[P]):` using the
    existing module-level `P = ParamSpec("P")`. One subtlety:
    `src/ariel/ec/individual.py` had self-referential aliases
    (`JSONType` refers to itself); PEP 695's lazy evaluation made
    this transparent, but PEP 613 is eager, so we dropped the
    recursion (nested values now `Any`, matching `pydantic.JsonValue`
    practice for runtime use). pyproject.toml dropped from `>=3.12`
    to `>=3.11`.

    **Validation:** blueprint_to_urdf + urdf_to_usd both run in one
    `isaaclab` conda env after `pip install -e ariel`. The pip
    resolver reports conflicts (numpy 2 vs `isaaclab`'s `<2`,
    gymnasium 1.3 vs 1.2.1, torch CUDA build swap from `+cu128` to
    `+cu130`) but the URDF→USD pipeline still works. Phase 2.5 makes
    this no longer a vague future cleanup item: ariel's base package
    should stop owning Isaac-Lab-sensitive binary deps. The concrete
    split is:

    * keep simulator-agnostic deps in `[project.dependencies]`;
    * pin base `numpy` conservatively (`>=1.26,<2` unless proven
      otherwise);
    * move `torch`, `torchvision`, and `gymnasium` out of top-level
      deps;
    * keep any future `isaaclab` extra glue-only, so the env created
      by Isaac Lab remains the sole owner of torch / gymnasium /
      low-level CUDA-coupled packages.

17. **Simulator backends are a plug point, not a hard-coded choice;
    the contract is two complementary shapes — one per ecosystem.**

    **Motivation:** the ARIEL consortium has collaborators using
    divergent simulators (MuJoCo, Aerial Gym, Isaac Lab, IsaacGym,
    in-house stacks). ARIEL's contribution is the *evolutionary +
    learning loop* — decoders, EA operators, repair, inspection,
    training boilerplate. If the simulator is hard-coded, every
    collaborator either swaps to ariel's simulator (won't happen) or
    re-implements the loop themselves (defeats the point). The seam
    has to be deliberate.

    **Architecture (revised in Phase 2): two Protocols, one trainer
    per backend.** Initially we tried a single `BlueprintGateEnv`
    Protocol (gymnasium VecEnv). The Phase 2 attempt to wrap Isaac
    Lab's `DirectRLEnv` to that shape via `isaaclab_rl.sb3` collided
    with the numpy-2 ABI issues that sb3's compiled deps have in the
    unified env. The honest fix is to accept heterogeneity: numpy
    backend exposes a gymnasium VecEnv and pairs with sb3 PPO;
    Isaac Lab backend exposes a `DirectRLEnv` and pairs with one of
    Isaac Lab's native RL libraries (`rl_games` is the v1 target).
    Each backend brings its own simulator AND its own trainer — same
    as how heterogeneous simulator ecosystems work in the wild.

    **Contract 1 (gymnasium-VecEnv path):** `BlueprintGateEnv` is a
    `typing.Protocol` declaring `.blueprint`, `.num_envs`, plus the
    standard VecEnv methods (inherited from
    `stable_baselines3.common.vec_env.VecEnv`). `@runtime_checkable`,
    so `isinstance(env, BlueprintGateEnv)` works for sanity checks.
    `NumpyBlueprintGateEnv` is the v1 reference implementation.

    **Contract 2 (Isaac Lab DirectRLEnv path):** subclass
    `isaaclab.envs.DirectRLEnv` with a config built from a
    `DroneBlueprint` (Blueprint → URDF → USD generated at construction
    time via the helpers in `isaaclab_hover_env.py`). The standard
    `_setup_scene` / `_pre_physics_step` / `_apply_action` /
    `_get_observations` / `_get_rewards` / `_get_dones` / `_reset_idx`
    methods implement the task.

    **What ships in v1:**
    - **NumPy backend (Phase 1):** `NumpyBlueprintGateEnv` —
      Protocol-conforming subclass of existing `DroneGateEnv` with a
      Blueprint→propellers adapter. Trains a gate-passing policy
      with sb3 PPO end-to-end. Smoke-tested with PPO 2k steps in
      4.7 s (3.12 venv); also works in the unified 3.11 env.
    - **Isaac Lab backend (Phase 2):** `IsaacLabBlueprintHoverEnv` —
      adapted from Isaac Lab's reference `QuadcopterEnv`; the
      articulation USD is generated at runtime from a Blueprint via
      `blueprint_to_urdf` + `UrdfConverter`. Spawns N parallel drones,
      computes `-distance × step_dt` rewards. v1 smoke-tests via a
      random-action stepping loop (PPO training deferred to
      Phase 2.5; see below). 72 steps in 1.6 s on 16 envs through
      Isaac Sim PhysX.
    - `tutorials/pluggable_simulator/{README.md, train.py}` — the
      tutorial doc + a unified entry point with `--simulator
      {numpy,isaaclab}` dispatch and the five-step "add your own
      backend" recipe.

    **Phase 2.5 deferred — three layered env-stack issues
    encountered:**
    1. `sb3 + numpy 2`: sb3's compiled deps segfault in the unified
       isaaclab env (numpy ABI mismatch with bundled torch). Avoided
       by NOT routing Isaac Lab through sb3.
    2. `isaaclab_rl.rsl_rl` adapter ↔ `rsl-rl-lib`: the adapter
       sends config keys (`optimizer`, `share_cnn_encoders`) that
       don't match the installed PPO signature on either 3.0.1 or
       5.3.0. Tried both; both fail differently. The
       `handle_deprecated_rsl_rl_cfg` shim doesn't catch the
       relevant fields on these versions.
    3. `rl_games` (the v1 target): the runner's tensorboard import
       transitively pulls in Isaac Sim's bundled `pip_prebundle/torch`,
       which imports an older `tensorflow` → `jax` →
       `numpy.dtypes.StringDType`. Bundle ships older numpy than the
       conda env's 2.4.6, so the attribute lookup fails. Architectural
       fix likely requires either a clean conda env with pinned-to-
       bundle versions, or shadowing the bundle's torch.

    **Pre-existing bug fixed in passing:** `DroneGateEnv.__init__`
    stored `self.seed = seed` (an int), shadowing the inherited
    `VecEnv.seed()` method that stable-baselines3's `PPO` calls in
    `set_random_seed`. Renamed to `self._seed` and updated the one
    other internal reader (`reset_seed`). Any PPO user of
    `DroneGateEnv` would have hit this; nothing in the existing
    examples exercised it because they didn't pass `seed=` to PPO.

18. **Session update (dependency split + smoke validation):
    two import-chain blockers were fixed to make the new extras
    usable in clean envs.**

    **Why:** after moving dependencies behind extras, clean-env smoke
    tests surfaced two failures that made the split incomplete:
    (a) importing `DroneGateEnv` through the `rl-sb3` path
    transitively imported `fcl` from an unrelated operator module;
    (b) `fabrication` imports failed because
    `phenotype_assembly/parts/` did not exist even though the
    assembly code imported it.

    **What was changed this session:**
    - `src/ariel/ec/drone/genome_handlers/__init__.py` was converted
      from eager imports to lazy `__getattr__`-based imports. This
      prevents optional heavy deps from being imported just by touching
      helper modules under `genome_handlers`.
    - Added missing
      `src/ariel/body_phenotypes/drone/phenotype_assembly/parts/`
      package with API-compatible factories:
      `create_core_plate`, `create_arm_mount`, `create_motor_arm`,
      `create_motor_mount`.

    **Validation:** reran smoke tests in fresh temporary uv envs.
    `rl-sb3 + torch` now imports `DroneGateEnv` successfully; and
    `fabrication` now imports
    `ariel.body_phenotypes.drone.phenotype_assembly.generator`
    successfully. `viz` and `experimental` extra smoke checks also
    passed.

19. **Lazy `__init__.py` is required across ariel's package surface —
    eager imports leak heavy deps into light-weight import paths.**

    **Why:** during the Phase 2.5 Option A execution pass we hit a
    cascade of `ModuleNotFoundError`s on every retry of the import
    smoke. Each one was an eager `__init__.py` import that pulled in
    a transitive dep the `--no-deps` ariel install intentionally
    skipped — `sympy` (via `simulation/__init__.py` →
    `mujoco_worker`), `sqlalchemy` / `sqlmodel` (via `ec/__init__.py`
    → `archive`), `pydantic_settings` (via
    `body_phenotypes/drone/__init__.py` → `operations` →
    `ec/ea.py`), and so on. The pattern was identical to the
    `genome_handlers/__init__.py` and `phenotype_assembly/parts/`
    fixes we landed earlier in the session.

    **The rule:** any package whose `__init__.py` re-exports symbols
    from submodules should defer those imports via `__getattr__`. The
    submodule (with its own heavy deps) is only loaded when the
    symbol is actually accessed — not when an unrelated sibling
    module is imported.

    **Pattern (after the fix, ec/__init__.py example):**

    ```python
    from __future__ import annotations
    from typing import TYPE_CHECKING, Any

    if TYPE_CHECKING:
        from ariel.ec.archive import Archive
        # … etc.

    __all__ = ["Archive", ...]

    _LAZY_IMPORTS: dict[str, str] = {
        "Archive": "ariel.ec.archive",
        # … one entry per re-exported symbol …
    }

    def __getattr__(name: str) -> Any:
        if name in _LAZY_IMPORTS:
            import importlib
            return getattr(importlib.import_module(_LAZY_IMPORTS[name]), name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    ```

    The package's external API (`from ariel.ec import Archive`) is
    unchanged. Existing examples and tests don't need updates.

    **Files this session:** converted `ariel/simulation/__init__.py`,
    `ariel/ec/__init__.py`,
    `ariel/body_phenotypes/drone/__init__.py`,
    `ariel/simulation/drone/__init__.py`. Adds to the prior
    `ariel/ec/drone/genome_handlers/__init__.py` (§6 entry 18) on the
    same pattern.

    **Future maintenance:** *new* `__init__.py` files that re-export
    submodule symbols should follow this pattern from the start, not
    be added eagerly and refactored later. Eager re-export should be
    reserved for `__init__.py` files whose source submodules are
    guaranteed to use only stdlib or core-dep imports.

20. **Operational gotcha: failed Isaac Sim smoke runs leak processes.**

    **Symptom:** a `python tutorials/pluggable_simulator/train.py …`
    process running at 100–120% CPU for hours even though the shell
    returned its prompt to you ages ago. We observed five such
    orphans across the Phase 2 and 2.5 sessions, each from a
    different failed smoke attempt; cumulatively ~25 core-hours
    burned silently.

    **Cause:** when Isaac Sim's `AppLauncher` or
    `SimulationContext` errors mid-init (a dependency mismatch, a
    config-class validation error, etc.), the Python interpreter's
    main thread re-raises and exits, but Isaac Sim's own app
    threads (Carb, USD, kit) don't always co-operatively shut down.
    The process continues spinning. This is an Isaac Sim behavior,
    not an ariel bug; the official Isaac Lab `train.py` is
    susceptible to the same issue.

    **Detection (run after every failed smoke):**

    ```bash
    ps -u $USER -o pid,etime,pcpu,cmd \
        | grep -E "tutorials/pluggable_simulator/train\.py" \
        | grep -v grep
    ```

    **Cleanup:**

    ```bash
    pkill -KILL -f "tutorials/pluggable_simulator/train.py"
    ```

    **Recommendation for the recipe:** add this `ps` check as a
    no-op step the user is expected to run after any non-zero-exit
    smoke. We're not baking it into the `train.py` script itself
    because the issue is the process not exiting; nothing
    in-process can guarantee the cleanup. Documented in
    `tutorials/pluggable_simulator/README.md` §3c.

## 7. Asks for the meeting

1. Confirm Isaac Lab as the simulator target the consortium will converge
   on (so the USD backend is worth investing in).
2. Decide whether partners' encodings should target this Blueprint, or
   whether we adopt a different community-standard IR if one is emerging.
3. Identify one partner-encoding (besides spherical/cartesian) to onboard
   as a third decoder — proves the portability claim with non-VUA code.
4. Agreement that ARIEL hosts the Blueprint spec; partners contribute
   decoders.
