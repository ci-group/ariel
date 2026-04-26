---
type: source_summary
tags: [source]
source: https://mujoco.readthedocs.io/en/stable/programming/extension.html
author: Google DeepMind
date_ingested: 2026-04-13
---
# Source - MuJoCo Extensions

MuJoCo's official documentation page on the two extensibility mechanisms: engine plugins (custom actuators, sensors, passive forces, SDF shapes) and resource providers (custom asset loading). Relevant to ariel for implementing custom sensor types, novel actuator dynamics, and alternative asset pipelines.

## Entity Pages Created

- [[mujoco_engine_plugins]] — Full engine plugin system: capabilities, MJCF declaration, configuration, plugin state vs plugin data, actuator act integration, registration (static + dynamic), SDF plugin interface, first-party plugin directory overview
- [[mjpPlugin]] — C struct definition, `mjtPluginCapabilityBit` enum, all callback signatures (nstate, init, destroy, copy, compute, advance, sdf_*), mjModel/mjData fields for plugin state
- [[mujoco_resource_providers]] — `mjpResourceProvider` and `mjResource` structs, all callback typedefs (open/read/close/getdir/modified), registration API, complete data-URI provider example with MJCF usage
