---
type: source_summary
tags: [mujoco, python, bindings, google-deepmind]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MuJoCo Python Bindings

Official Python bindings for the MuJoCo physics engine, maintained by Google DeepMind. Provides direct access to the MuJoCo C API via pybind11 wrappers. Install with `pip install mujoco`.

**Design principles:**
- C struct names are PEP-8 renamed: `mjData` → `mujoco.MjData`, `mjvCamera` → `mujoco.MjvCamera`
- All array fields are NumPy views backed by C memory (no copying on access)
- Memory management is automatic via Python destructors
- MuJoCo errors surface as `mujoco.FatalError` exceptions

## Pages Created

### Core Classes
- [[MjModel]] — static model (geometry, physics parameters)
- [[MjData]] — dynamic simulation state (positions, velocities, forces)
- [[MjSpec]] — programmatic model builder (compiles to MjModel)
- [[MjOption]] — physics solver options (timestep, gravity, solver)
- [[MjvCamera]] — abstract visualizer camera
- [[MjvOption]] — visualization flags and options
- [[MjvScene]] — abstract geometric scene for rendering
- [[MjrContext]] — OpenGL rendering context
- [[MjrRect]] — viewport rectangle

### Functions & APIs
- [[mujoco_simulation_functions]] — mj_step, mj_forward, substage functions
- [[mujoco_kinematics_functions]] — mj_kinematics, mj_comPos, tendon/transmission
- [[mujoco_inertia_functions]] — mj_crb, mj_factorM, mj_solveM, mj_mulM
- [[mujoco_named_access_api]] — O(1) named element access on MjModel/MjData
- [[mujoco_rendering_pipeline]] — low-level rendering with mjv/mjr functions
- [[mujoco_viewer]] — interactive passive/blocking viewer
- [[mujoco_callbacks]] — mjcb_control, mjcb_sensor, and other callbacks
- [[mujoco_enumerations]] — mjtObj, mjtJoint, mjtGeom, mjtCamera, constants
