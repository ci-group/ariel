---
type: api_reference
tags: [mujoco, physics, plugin, extension, c-api]
source: https://mujoco.readthedocs.io/en/stable/programming/extension.html
date_ingested: 2026-04-13
---
# mjpPlugin

Struct bundling a plugin's logic: stateless function callbacks and static attributes that define a MuJoCo engine plugin's capabilities and behavior.

## Signature

```c
struct mjpPlugin {
  const char* name;               // globally unique plugin identifier string
  int capabilityflags;            // bitfield of mjtPluginCapabilityBit values

  // Config attribute declaration
  const char** attributes;        // null-terminated list of declared config attribute names
  int nattribute;                 // number of declared attributes

  // State size callback
  int (*nstate)(const mjModel* m, int instance);

  // Lifecycle callbacks
  void (*init)(const mjModel* m, mjData* d, int instance);
  void (*destroy)(mjData* d, int instance);
  void (*copy)(mjData* dest, const mjModel* m, const mjData* src, int instance);

  // Computation callbacks (capability-dependent)
  void (*compute)(const mjModel* m, mjData* d, int instance, int capability_bit);
  void (*advance)(const mjModel* m, mjData* d, int instance);

  // SDF-specific callbacks
  mjtNum (*sdf_distance)(const mjtNum point[3], const mjData* d, int instance);
  void   (*sdf_staticdistance)(const mjtNum point[3], const mjtNum* attributes, mjtNum* distance);
  void   (*sdf_gradient)(mjtNum gradient[3], const mjtNum point[3], const mjData* d, int instance);
  void   (*sdf_aabb)(mjtNum aabb[6], const mjData* d, int instance);
};
```

> **Note:** The exact struct definition is not fully documented in public docs; the above is derived from first-party plugin source and documentation. Use `#include <mujoco/mjplugin.h>` for the authoritative definition.

## Capability Flags (`mjtPluginCapabilityBit`)

```c
typedef enum mjtPluginCapabilityBit_ {
  mjPLUGIN_ACTUATOR = 1 << 0,   // custom actuator
  mjPLUGIN_SENSOR   = 1 << 1,   // custom sensor
  mjPLUGIN_PASSIVE  = 1 << 2,   // passive force
  mjPLUGIN_SDF      = 1 << 3,   // signed distance field
} mjtPluginCapabilityBit;
```

## Key Callbacks

| Callback | When Called | Purpose |
|---|---|---|
| `nstate(m, instance)` | Model compilation | Returns number of float state values needed for this instance |
| `init(m, d, instance)` | `mj_makeData` | Allocate and initialize plugin data; store pointer in `d->plugin_data[instance]` |
| `destroy(d, instance)` | `mj_deleteData` | Free plugin data allocated in `init` |
| `copy(dest, m, src, instance)` | `mj_copyData` | Set up plugin data for newly copied `mjData` |
| `compute(m, d, instance, cap_bit)` | During step | Main computation (sensor value, actuator force, passive force) |
| `advance(m, d, instance)` | After `act_dot` integration | Post-integration update for stateful actuators using `mjData.act` |
| `sdf_distance` | Collision detection | Signed distance at query point (local coords) |
| `sdf_staticdistance` | Compiler (mesh gen) | Static version of distance — called before instantiation |
| `sdf_gradient` | Collision detection | SDF gradient at query point |
| `sdf_aabb` | Collision detection | Axis-aligned bounding box for the SDF shape |

## Registration

```c
// Static / inline registration
mjpPlugin plugin = { .name = "myorg.mylib.myplugin", ... };
mjp_registerPlugin(&plugin);

// Auto-register when dynamic library is loaded (preferred for reusable plugins)
mjPLUGIN_LIB_INIT {
  mjp_registerPlugin(&plugin);
}
```

## Data Fields in `mjModel` / `mjData`

| Field | Location | Description |
|---|---|---|
| `plugin_state` | `mjData` | Contiguous float array holding all plugin state values |
| `plugin_stateadr[i]` | `mjModel` | Start offset in `plugin_state` for instance `i` |
| `plugin_data[i]` | `mjData` | Opaque pointer to plugin-managed data for instance `i` |
| `nplugin` | `mjModel` | Total number of plugin instances |

## Notes

- Plugin code is **stateless** — all mutable state must go through `plugin_state` or `plugin_data`; never use global/static variables in callbacks.
- `plugin_data` is not serialized by MuJoCo; must be reconstructible from config + `plugin_state` + MuJoCo state variables.
- For actuator plugins that use `mjData.act`, `advance` runs after MuJoCo integrates `act_dot`; plugin may overwrite `act` at that point.
- See [[mujoco_engine_plugins]] for the full plugin system overview, MJCF declaration, and SDF collision details.
