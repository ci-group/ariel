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

**Phase 3 — Isaac Lab spike. Largely landed for rigid drones (branch `blueprint-to-urdf`).**
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

**Still pending — to pick up next session:**

* **Phase 2 of pluggable simulator** — implement
  `IsaacLabBlueprintGateEnv` for real (URDF→USD in-process, parallel
  envs, per-motor thrust model lifted from soft_airframe). The
  Protocol contract is now stable, so this is purely simulator-side
  work.
* **Compliant joints (URDF revolute + sidecar stiffness + USD
  `ariel:*` attrs).** Schema landed on `compliant-joint-schema`;
  the emission half is what's still pending. The two-layer pattern
  is documented in §6 entry 10 — zero-stiffness `revolute` joint in
  URDF, non-linear `τ(θ)` law stashed as sidecar attrs for a
  runtime controller.
* Pytest integration test for the Blueprint → URDF → USD pipeline
  (now feasible in-process under the unified env; just needs a
  `pytest.mark.isaaclab` skip-if-not-available guard).
* Targeted dep pins in pyproject.toml to keep numpy < 2 and
  gymnasium == 1.2.1 so isaaclab's pins aren't violated. **Promoted
  from stretch:** the smoke test for `pluggable-simulator` segfaulted
  in the unified isaaclab env on PPO startup (likely a numpy 2 ABI
  conflict with stable-baselines3's compiled deps). Same smoke test
  passed cleanly in the ariel 3.12 venv where numpy 1.x is intact,
  confirming the seam itself is correct.

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
    `+cu130`) but the URDF→USD pipeline still works. If those
    conflicts bite elsewhere (RL training, dex_retargeting, etc.)
    they'll need targeted pins in ariel's pyproject.toml; deferred
    until something actually breaks.

17. **Simulator backends are a plug point, not a hard-coded choice;
    the contract is the `BlueprintGateEnv` Protocol.**

    **Motivation:** the ARIEL consortium has collaborators using
    divergent simulators (MuJoCo, Aerial Gym, Isaac Lab, IsaacGym,
    in-house stacks). ARIEL's contribution to those groups is the
    *evolutionary + learning loop* — decoders, EA operators, repair,
    inspection, PPO training. If the simulator is hard-coded, every
    collaborator either swaps to ariel's simulator (won't happen) or
    re-implements the loop themselves (defeats the point). The
    seam has to be deliberate.

    **Contract:** a `BlueprintGateEnv` is a `gymnasium.VecEnv` (stable-
    baselines3 style) constructed from a `DroneBlueprint`, exposing
    `.blueprint` and `.num_envs` attributes. Implemented as
    `typing.Protocol` with `@runtime_checkable` (per the
    pluggable-backend question we deliberated): no inheritance
    forced on collaborators, but `isinstance(env, BlueprintGateEnv)`
    works for runtime sanity-checks. Concrete VecEnv methods come
    via the standard `stable_baselines3.common.vec_env.VecEnv` base
    class.

    **What ships in v1:**
    - `NumpyBlueprintGateEnv` — declares the existing `DroneGateEnv`
      (NumPy + SymPy physics) as a Protocol implementation by
      subclassing it and adding a Blueprint → propellers-list
      adapter at the constructor. Smoke-tested with PPO 2k steps in
      4.7 s.
    - `IsaacLabBlueprintGateEnv` — stub with the planned constructor
      signature and a `NotImplementedError` body. Phase 2 work.
    - `tutorials/pluggable_simulator/{README.md, train.py}` —
      walks a collaborator through the architecture, the
      `--simulator {numpy,isaaclab}` dispatch, and the five-step
      "add your own backend" recipe.

    **Pre-existing bug fixed in passing:** `DroneGateEnv.__init__`
    stored `self.seed = seed` (an int), shadowing the inherited
    `VecEnv.seed()` method that stable-baselines3's `PPO` calls in
    `set_random_seed`. Renamed to `self._seed` and updated the one
    other internal reader (`reset_seed`). Any PPO user of
    `DroneGateEnv` would have hit this; nothing in the existing
    examples exercised it because they didn't pass `seed=` to PPO.

## 7. Asks for the meeting

1. Confirm Isaac Lab as the simulator target the consortium will converge
   on (so the USD backend is worth investing in).
2. Decide whether partners' encodings should target this Blueprint, or
   whether we adopt a different community-standard IR if one is emerging.
3. Identify one partner-encoding (besides spherical/cartesian) to onboard
   as a third decoder — proves the portability claim with non-VUA code.
4. Agreement that ARIEL hosts the Blueprint spec; partners contribute
   decoders.
