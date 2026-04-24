---
type: api_reference
tags: [mujoco, python, mjdata, sensors, mocap, ariel]
source: https://ci-group.github.io/ariel/source/Mujoco_docs/mujoco_docs.html
---
# MjData — Sensor, Motion Capture & Utility Variables

Miscellaneous [[MjData]] fields for sensor readings, motion capture overrides, energy, the inertia matrix, and the `bind` helper.

---

## `sensordata` — Sensor Measurements

**Definition:** Flat array containing the output of all sensors defined in the model, in declaration order.

**Type:** `numpy.ndarray`, shape `(nsensordata,)`

```python
# Read a specific sensor by name
sensor_id = model.sensor('imu_accel').id
adr = model.sensor_adr[sensor_id]
dim = model.sensor_dim[sensor_id]
reading = data.sensordata[adr : adr + dim]
```

---

## `mocap_pos` — Motion Capture Position

**Definition:** Position of motion capture bodies. Write to these to kinematically drive `mocap` bodies defined in the XML.

**Type:** `numpy.ndarray`, shape `(nmocap, 3)`

```python
# Move mocap body 0 to a target position
data.mocap_pos[0] = [1.0, 0.0, 0.5]
```

---

## `mocap_quat` — Motion Capture Orientation

**Definition:** Quaternion orientation of motion capture bodies `[qw, qx, qy, qz]`.

**Type:** `numpy.ndarray`, shape `(nmocap, 4)`

```python
# Set identity orientation for mocap body 0
data.mocap_quat[0] = [1, 0, 0, 0]
```

---

## `energy` — Simulation Energy

**Definition:** Potential and kinetic energy of the system. Only populated when `model.opt.enableflags` includes energy computation.

**Type:** `numpy.ndarray`, shape `(2,)` — `[potential_energy, kinetic_energy]`

```python
potential, kinetic = data.energy
total = potential + kinetic
```

---

## `M` — Inertia Matrix

**Definition:** Joint-space inertia (mass) matrix. Shape `(nv, nv)`, symmetric positive-definite.

**Type:** `numpy.ndarray`, shape `(nv, nv)`

**Note:** Call `mujoco.mj_fullM(model, M, data.qM)` to populate the dense form from MuJoCo's sparse internal representation.

```python
M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M, data.qM)
```

---

## `bind` — Step-by-Step Data Binding

**Definition:** Class-level helper for binding named model objects so their runtime data can be accessed at each simulation step.

**Example:**

```python
import mujoco

# Find all geoms in the world body
geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)

# Bind them to track per-step positions
bound_geoms = data.bind(geoms)
# Access position of first bound geom
pos = bound_geoms.xpos[0]
```

---

## See Also

- [[MjData]]
- [[MjData_Spatial_Variables]]
- [[MjData_State_Variables]]
