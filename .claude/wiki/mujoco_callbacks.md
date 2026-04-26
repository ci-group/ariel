---
type: api_reference
tags: [mujoco, python, callbacks, control]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MuJoCo Callbacks

MuJoCo supports installing Python callables as C-level callback hooks via `set_mjcb_*` / `get_mjcb_*` functions.

## Callback Management

```python
mujoco.set_mjcb_foo(callable)   # install callback
mujoco.set_mjcb_foo(None)       # remove callback
cb = mujoco.get_mjcb_foo()      # get current callback (or None)
```

## Available Callbacks

| Callback name | Trigger point | Typical use |
|---------------|--------------|-------------|
| `mjcb_control` | after `mj_step1`, before `mj_step2` | custom control laws; write to `data.ctrl`, `data.qfrc_applied`, `data.xfrc_applied` |
| `mjcb_sensor` | after sensor positions/velocities computed | populate `data.sensordata` for user-defined sensors |
| `mjcb_time` | at simulation start | custom time source |
| `mjcb_act_dyn` | actuator dynamics | custom activation dynamics |
| `mjcb_act_gain` | actuator gain computation | custom gain |
| `mjcb_act_bias` | actuator bias computation | custom bias |
| `mjcb_contactfilter` | collision detection | filter/reject contact pairs |
| `mjcb_passive` | passive forces | custom passive dynamics |

## Callback Signatures

```python
def control_callback(model, data):
    # model: MjModel (read-only recommended)
    # data: MjData (read/write)
    data.ctrl[0] = -0.1 * data.qpos[0]

def sensor_callback(model, data):
    # Populate user sensor data
    data.sensordata[user_sensor_adr] = compute_sensor(data)
```

## Example: Control Callback

```python
import mujoco

model = mujoco.MjModel.from_xml_path('model.xml')
data  = mujoco.MjData(model)

def my_controller(model, data):
    data.ctrl[0] = -0.1 * data.qpos[0]   # proportional control

mujoco.set_mjcb_control(my_controller)

for _ in range(500):
    mujoco.mj_step(model, data)

mujoco.set_mjcb_control(None)  # clean up
```

## Performance Note

The GIL is acquired and released on each callback entry/exit. This can be expensive at high simulation frequencies (e.g., many solver iterations with `mjcb_contactfilter`).

## MjSpec Recompile Warning

Always remove callbacks before recompiling a [[MjSpec]] to avoid dangling references:

```python
mujoco.set_mjcb_control(None)
model = spec.compile()
mujoco.set_mjcb_control(my_controller)
```

See [[mujoco_simulation_functions]], [[MjSpec]].
