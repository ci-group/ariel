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
    Accepts `motor_mass`, `arm_mass`, `core_mass_override` so the MuJoCo
    body mass can be matched to whatever `DroneConfiguration` auto-computes.
  - `blueprint_to_usd` still a stub.
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

**Phase 3 — Isaac Lab spike. Not started.** USD backend remains stubbed;
the MJCF backend is structured so a USD compiler maps node-for-node from
the same blueprint tree.

## 6. Asks for the meeting

1. Confirm Isaac Lab as the simulator target the consortium will converge
   on (so the USD backend is worth investing in).
2. Decide whether partners' encodings should target this Blueprint, or
   whether we adopt a different community-standard IR if one is emerging.
3. Identify one partner-encoding (besides spherical/cartesian) to onboard
   as a third decoder — proves the portability claim with non-VUA code.
4. Agreement that ARIEL hosts the Blueprint spec; partners contribute
   decoders.
