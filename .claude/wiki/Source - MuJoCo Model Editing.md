---
type: source_summary
tags: [source, mujoco, mjspec, model-building]
source: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html
author: DeepMind Technologies
date_ingested: 2026-04-13
---
# Source - MuJoCo Model Editing

MuJoCo's programmatic model construction and editing API (`mjSpec`, `mjs_*` functions). Covers the full C API for building models without XML, including the attachment system for assembling modular subtrees — directly relevant to ariel's `lynx_mjspec` and `robogen_lite` body phenotype pipelines.

Source included content from `mujoco/mjspec.h` and `mujoco/mujoco.h` (local install), plus the RST documentation from the MuJoCo GitHub repository.

## Entity Pages Created

- [[mujoco_model_editing_c_api]] — Complete C API: all `mj_*` lifecycle/compile functions and `mjs_*` constructors, finders, getters/setters, type-casts, defaults, and attachment functions
- [[mjsBody]] — `mjsBody` and `mjsFrame` struct fields; body frame/inertial frame conventions
- [[mjsGeom]] — `mjsGeom` and `mjsSite` struct fields; geom size convention table for all shape types
- [[mjsJoint]] — `mjsJoint` struct fields; `mjtLimited` and `mjtAlignFree` enums
- [[mjsActuator]] — `mjsActuator` struct fields; `mjs_setTo*` shorthand functions
- [[mjsSensor]] — `mjsSensor` struct fields; common sensor type table
- [[mujoco_mjspec_enums]] — All `mjspec.h` enums: `mjtGeomInertia`, `mjtMeshInertia`, `mjtLimited`, `mjtOrientation`, `mjtInertiaFromGeom`, etc.
- [[MjSpec]] — Updated: added Default Classes section, Attachment section, In-Place Recompilation section, `mjSpec`/`mjsCompiler` struct field tables
