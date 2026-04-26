---
type: api_reference
tags: [mujoco, physics, plugin, actuator, sensor, extension]
source: https://mujoco.readthedocs.io/en/stable/programming/extension.html
date_ingested: 2026-04-13
---
# mujoco_engine_plugins

Engine plugins allow user-defined logic to be inserted into MuJoCo's computational pipeline. Custom sensor types, actuators, passive forces, and signed distance fields can be implemented as plugins referenced in MJCF.

## Core Concepts

**Plugin** — a stateless collection of functions and static attributes bundled into an [[mjpPlugin]] struct. Treated as "pure logic"; often packaged as a C library. Not a model element.

**Plugin instance** — a model element (`mjOBJ_PLUGIN`) representing runtime state operated on by the plugin. `mjModel.nplugin` instances exist with ids in `[0, nplugin-1]`. Supports `mj_name2id` / `mj_id2name`.

Instance relationships:
- **one-to-one** — each instance is referenced by exactly one element (e.g. two separate sensors).
- **one-to-many** — multiple elements (of same or different types) share a single instance. Useful when values are linked to the same physical entity (e.g. motor + thermometer) or when batching is faster (e.g. all N motors through a single neural net forward pass).

## Plugin Capabilities

Capabilities are declared as a bitfield in `mjpPlugin.capabilityflags` using the `mjtPluginCapabilityBit` enum.

| Capability | Description |
|---|---|
| Actuator plugin | Custom actuator dynamics, gain, bias |
| Sensor plugin | Custom sensor value computation |
| Passive force plugin | Custom passive generalized forces |
| Signed distance field (SDF) plugin | Custom mesh-free collision geometry |

## Declaration in MJCF

```xml
<mujoco>
  <extension>
    <plugin plugin="mujoco.test.simple_sensor_plugin"/>
    <plugin plugin="mujoco.test.actuator_sensor_plugin">
      <instance name="explicit_instance"/>
    </plugin>
  </extension>

  <sensor>
    <!-- implicit instance creation (one-to-one) -->
    <plugin name="sensor0" plugin="mujoco.test.simple_sensor_plugin"/>
    <plugin name="sensor1" plugin="mujoco.test.simple_sensor_plugin"/>
    <!-- explicit instance reference (one-to-many) -->
    <plugin name="sensor2" instance="explicit_instance"/>
  </sensor>

  <actuator>
    <plugin name="actuator2" instance="explicit_instance"/>
  </actuator>
</mujoco>
```

`sensor0` and `sensor1` each get an implicit instance. `sensor2` and `actuator2` share `explicit_instance`.

## Configuration in MJCF

Plugins expose custom config attributes via `<config key="..." value="...">` children:

```xml
<extension>
  <plugin plugin="mujoco.test.simple_actuator_plugin">
    <instance name="explicit_instance">
      <config key="resistance" value="1.0"/>
      <config key="inductance" value="2.0"/>
    </instance>
  </plugin>
</extension>

<actuator>
  <!-- references pre-existing instance (no config children allowed) -->
  <plugin name="actuator0" instance="explicit_instance"/>
  <!-- creates implicit instance with inline config -->
  <plugin name="actuator1" plugin="mujoco.test.simple_actuator_plugin">
    <config key="resistance" value="3.0"/>
    <config key="inductance" value="4.0"/>
  </plugin>
</actuator>
```

Adding `<config>` children to `actuator0` would be a compile error since no new instance is being created.

## Plugin State vs Plugin Data

| Concept | Type | Serialized | Description |
|---|---|---|---|
| **Plugin state** | `float[]` | Yes | Time-dependent values evolving with physics (e.g. temperature). Stored in `mjData.plugin_state`. |
| **Plugin data** | Arbitrary | No | Memoized structures reconstructible from config + state (e.g. preloaded NN weights). Managed by plugin via `init`/`destroy`/`copy` callbacks. |

- Plugin declares number of float values needed via `nstate` callback in [[mjpPlugin]].
- `mjModel.plugin_stateadr[i]` gives the offset in `plugin_state` for instance `i`.
- `mj_makeData` calls `init` to create plugin data; `mj_deleteData` calls `destroy`; `mj_copyData` calls `copy`.

## Actuator States and `mjData.act`

Stateful actuator plugins may store state in either `plugin_state` or `mjData.act`:

- **`mjData.act`** — MuJoCo auto-integrates `act_dot` between timesteps. `mjd_transitionFD` works as for native actuators. `mjpPlugin.advance` is called after integration; plugin may overwrite `act` values if the built-in integrator is inappropriate.
- Optional `dyntype` attribute on the actuator introduces a filter/integrator between user inputs and actuator states. The `dyntype` state variable is placed *after* the plugin's state variables in the `act` array.

## Registration

Plugins must be registered before being referenced in MJCF.

### Static linking (one-off / throwaway)

```c
mjpPlugin plugin = {/* ... */};
mjp_registerPlugin(&plugin);
```

### Dynamic library (reusable)

Use `mjPLUGIN_LIB_INIT` macro — auto-registers the plugin when the library is loaded:

```c
// In library source file:
mjPLUGIN_LIB_INIT {
  mjp_registerPlugin(&my_plugin);
}
```

Expands to `__attribute__((constructor))` on GCC or equivalent MSVC CRT injection.

### Loading plugin libraries at runtime

```c
// Load a single dynamic library
mj_loadPluginLibrary(const char* path);

// Scan a directory and load all plugin libraries found
mj_loadAllPluginLibraries(const char* directory, mjfPluginLibraryLoadCallback callback);
```

`mj_loadPluginLibrary` is preferred over raw `dlopen`/`LoadLibraryA` because the expected registration mechanism may evolve. `mj_loadAllPluginLibraries` is used by the interactive viewer to auto-load user-provided plugins placed in a directory.

## SDF Plugin Methods

SDF plugins define custom collision geometry as signed distance fields. The following methods must be implemented:

| Method | Description |
|---|---|
| `sdf_distance(point, plugin_instance, ngeom)` | Returns signed distance of query point in local coordinates |
| `sdf_staticdistance(point, attributes)` | Static version; called by compiler for mesh creation (before instantiation) |
| `sdf_gradient(grad, point, plugin_instance, ngeom)` | Computes SDF gradient at query point in local coordinates |
| `sdf_aabb(aabb, plugin_instance, ngeom)` | Computes axis-aligned bounding box in local coordinates |

Collision algorithm: minimizes `A + B + abs(max(A, B))` (A and B are the two SDFs) via gradient descent. Number of starting points set by `sdf_initpoints`; iterations by `sdf_iterations`. Starting points use the Halton sequence inside the intersection of the AABBs.

`sdf_distance` is also called by the compiler to generate a visual mesh via marching cubes (MarchingCubeCpp).

Exact SDFs are preferred, but any function that vanishes at the surface and grows monotonically away from it (negative interior) can work for collision detection.

## First-Party Plugin Directories

| Directory | Contents |
|---|---|
| `plugin/actuator/` | PID controller |
| `plugin/elasticity/` | 1D cable (rotation-invariant, large deformation) and 2D shell (constant stiffness matrix) passive forces |
| `plugin/sensor/` | Touch grid sensor |
| `plugin/sdf/` | Mesh-free SDF-based collision shapes |

## Notes

- Callbacks are deprecated for stable extended functionality; plugins are the preferred mechanism.
- Thread safety: plugin instances are thread-local, avoiding data races in parallel simulation.
- See [[mujoco_callbacks]] for the legacy global callback system.
- See [[mjpPlugin]] for the full struct definition and callback signatures.
- See [[mujoco_resource_providers]] for asset loading extensions.
