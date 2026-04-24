---
type: source_summary
tags: [source, mujoco, simulation]
source: https://mujoco.readthedocs.io/en/stable/programming/simulation.html
author: DeepMind / Google
date_ingested: 2026-04-13
---
# Source - MuJoCo Programming Simulation

MuJoCo's programming reference for the Simulation chapter, covering initialization, simulation loop, state model, forward/inverse dynamics, multi-threading, model changes, data layout, Jacobians, contacts, and diagnostics. Directly relevant to ariel's evaluation pipeline, multi-rollout EA sampling, and runtime model modification.

## Entity Pages Created

- [[mujoco_state_control]] — `mjtState` bitfield, all state components (qpos/qvel/act/ctrl/etc.), `mj_getState`/`mj_setState`/`mj_copyState`/`mj_extractState`; critical for EA rollout state save/restore
- [[mujoco_jacobians]] — `mj_jac` and all `mj_jacXXX` convenience functions; Jacobian-based control, moment arms, constraint Jacobian
- [[mujoco_contacts]] — `mjContact` struct fields, `mj_contactForce`, friction cone types, custom collision function override via `mjCOLLISIONFUNC`
- [[mujoco_data_layout]] — Row-major layout, quaternion convention, sparse matrix (CSR) format, buffer aliasing gotchas, internal stack (`mj_markStack`/`mj_stackAllocNum`/`mj_freeStack`), arena memory monitoring
- [[mujoco_diagnostics]] — Warnings (`data.warning`), timers (`data.timer`, `mjcb_time`), solver diagnostics (`solver_niter`, `solver_fwdinv`), energy monitoring, memory diagnostics (`maxuse_arena`)
- [[mujoco_simulation_functions]] — **Updated**: added `mj_forwardSkip`/`mj_inverseSkip` with skip levels (`mjSTAGE_NONE/POS/VEL`), multi-threading pattern with per-thread `MjData`, `mj_setConst` for safe runtime model modifications
