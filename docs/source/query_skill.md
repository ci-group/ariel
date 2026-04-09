# Using the `/query` Skill in ARIEL

The `/query` skill lets you ask Claude Code questions that are answered from the project's local wiki (`wiki/`) rather than from general training knowledge. For ARIEL development this is particularly useful because MuJoCo's Python bindings have several non-obvious behaviors that differ from what you might expect based on the C API docs or older tutorials.

## Basic Usage

```
/query <your question>
```

You can optionally attach source context using `@`:

```
/query What timestep should I use? @src/ariel/body_phenotypes/lynx_mjspec/table.py
/query How do I add a position actuator? @src/ariel/body_phenotypes/lynx_mjspec/lynx_arm.py
```

The `@` context tells Claude which file you're currently working in, so answers are framed around your actual code rather than abstract examples.

---

## What's in the Wiki

The `wiki/` directory contains reference pages ingested from the MuJoCo Python bindings docs. Each page covers a specific class or module:

| Wiki file | What it covers |
|---|---|
| `MjSpec.md` | Building models programmatically (`add_body`, `add_geom`, `add_joint`, `add_actuator`, `compile`) |
| `MjOption.md` | Physics solver settings (`timestep`, `gravity`, `integrator`) |
| `MjModel.md` | Compiled model fields (read-only after compile) |
| `MjData.md` | Simulation state (`qpos`, `qvel`, `ctrl`, `xpos`, etc.) |
| `MjData_Actuator_Variables.md` | Actuator-specific fields |
| `MjData_Sensor_Variables.md` | Sensor readout fields |
| `MjData_Spatial_Variables.md` | Body/geom spatial transforms |
| `MjData_Force_Variables.md` | Contact and constraint forces |
| `mujoco_simulation_functions.md` | `mj_step`, `mj_forward`, `mj_resetData`, etc. |
| `mujoco_kinematics_functions.md` | Forward/inverse kinematics helpers |
| `mujoco_named_access_api.md` | `mj_name2id`, `mj_id2name`, `bind()` |
| `mujoco_rendering_pipeline.md` | Rendering setup |
| `mujoco_viewer.md` | `launch_passive`, `launch` viewer API |
| `mujoco_callbacks.md` | `set_mjcb_control` and other callbacks |
| `mujoco_enumerations.md` | `mjtGeom`, `mjtJoint`, `mjtTrn`, etc. |

---

## Good Questions to Ask

### MjSpec / model building
```
/query How do I attach one MjSpec to another (e.g. arm onto table)?
/query What size array does geom.size need for a cylinder?
/query How do I set the simulation timestep before compiling?
```

### Actuators and control
```
/query What do gainprm and biasprm do for a position servo actuator?
/query How do I make a position-controlled joint with velocity damping?
/query What is the difference between dynprm, gainprm, and biasprm?
```

### Simulation state and data access
```
/query How do I read joint position and velocity for a named joint?
/query How do I get the world-space position of a site during simulation?
/query What is the difference between xpos and site_xpos?
```

### Viewer / replay
```
/query How do I run a passive viewer that I can step manually?
/query How do I sync the viewer after each physics step?
```

---

## Why Use `/query` Instead of Just Asking?

The wiki acts as **ground truth for this project**. It captures behaviors that are easy to get wrong:

- `geom.size` always requires exactly 3 elements regardless of shape type (pad unused dims with `0`).
- `spec.option.timestep` must be set on the **spec** before compile, not on `model.opt` after.
- `mujoco.viewer.launch` blocks; `launch_passive` returns a context manager for manual stepping.
- Named access like `mj_name2id` uses `mjtObj` enums, not plain strings.

These details are in the wiki and `/query` will cite the specific file they come from. General knowledge from Claude's training may give plausible but subtly wrong answers for the Python bindings specifically.

---

## Adding to the Wiki

If you find something not covered, use `/ingest` to add it:

```
/ingest <URL or file path>
```

Good candidates for ingestion:
- New MuJoCo release notes that change Python binding behavior
- ARIEL-specific conventions (e.g. how `contype`/`conaffinity` is set for arm vs table)
- Tuning rules (e.g. joint damping vs actuator gain relationships)
