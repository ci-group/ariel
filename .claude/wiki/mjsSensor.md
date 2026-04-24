---
type: api_reference
tags: [mujoco, mjspec, model-building, c-api]
source: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html
date_ingested: 2026-04-13
---
# mjsSensor

C struct representing a sensor element in an [[MjSpec]] model specification. Corresponds to the MJCF `<sensor>` element.

## Signature

```c
typedef struct mjsSensor_ {
  mjsElement* element;             // element type (do not modify)

  // sensor definition
  mjtSensor type;                  // type of sensor
  mjtObj objtype;                  // type of sensorized object
  mjString* objname;               // name of sensorized object
  mjtObj reftype;                  // type of referenced object
  mjString* refname;               // name of referenced object
  int intprm[mjNSENS];             // integer parameters

  // user-defined sensors
  mjtDataType datatype;            // data type: mjDATATYPE_REAL, mjDATATYPE_POSITIVE, mjDATATYPE_AXIS, mjDATATYPE_QUATERNION
  mjtStage needstage;              // compute stage: mjSTAGE_POS, mjSTAGE_VEL, mjSTAGE_ACC
  int dim;                         // number of scalar outputs

  // output post-processing
  double cutoff;                   // cutoff for real and positive datatypes
  double noise;                    // noise standard deviation

  // other
  mjDoubleVec* userdata;           // user data
  mjsPlugin plugin;                // sensor plugin
  mjString* info;                  // message appended to compiler errors
} mjsSensor;
```

## Parameters

| Field | Type | Description |
|---|---|---|
| `type` | `mjtSensor` | Sensor type (e.g., `mjSENS_JOINTPOS`, `mjSENS_ACTUATORFRC`, `mjSENS_FRAMEPOS`, etc.) |
| `objtype` | `mjtObj` | Type of the object being sensed |
| `objname` | `mjString*` | Name of the sensorized object (joint, site, body, etc.) |
| `reftype` | `mjtObj` | Type of reference object (for relative sensors) |
| `refname` | `mjString*` | Name of reference object |
| `intprm` | `int[mjNSENS]` | Integer parameters (sensor-type-specific) |
| `datatype` | `mjtDataType` | Output data type for user-defined sensors |
| `needstage` | `mjtStage` | Required compute stage for user sensors |
| `dim` | `int` | Output dimensionality for user-defined sensors |
| `cutoff` | `double` | Clamp output magnitude to this value (0 = no cutoff) |
| `noise` | `double` | Standard deviation of additive Gaussian noise |

## Common Sensor Types (`mjtSensor`)

| Constant | Object type | Dim | Description |
|---|---|---|---|
| `mjSENS_JOINTPOS` | joint | 1 | Joint position |
| `mjSENS_JOINTVEL` | joint | 1 | Joint velocity |
| `mjSENS_ACTUATORFRC` | actuator | 1 | Actuator force |
| `mjSENS_ACTUATORPOS` | actuator | 1 | Actuator position (transmission length) |
| `mjSENS_ACTUATORVEL` | actuator | 1 | Actuator velocity |
| `mjSENS_FRAMEPOS` | site/body | 3 | Frame position in world |
| `mjSENS_FRAMEQUAT` | site/body | 4 | Frame orientation (quaternion) |
| `mjSENS_FRAMEXAXIS` | site/body | 3 | Frame x-axis in world |
| `mjSENS_FRAMEYAXIS` | site/body | 3 | Frame y-axis in world |
| `mjSENS_FRAMEZAXIS` | site/body | 3 | Frame z-axis in world |
| `mjSENS_FRAMELINVEL` | site/body | 3 | Linear velocity |
| `mjSENS_FRAMEANGVEL` | site/body | 3 | Angular velocity |
| `mjSENS_SUBTREECOM` | body | 3 | Center of mass of body subtree |
| `mjSENS_TOUCH` | site | 1 | Touch sensor (contact normal force) |
| `mjSENS_ACCELEROMETER` | site | 3 | Accelerometer |
| `mjSENS_GYRO` | site | 3 | Gyroscope (angular velocity) |
| `mjSENS_FORCE` | site | 3 | Force at site |
| `mjSENS_TORQUE` | site | 3 | Torque at site |
| `mjSENS_USER` | any | `dim` | User-defined sensor |

## Examples

```c
// C: joint position sensor
mjsSensor* sensor = mjs_addSensor(spec);
mjs_setName(sensor->element, "q0");
sensor->type = mjSENS_JOINTPOS;
sensor->objtype = mjOBJ_JOINT;
mjs_setString(sensor->objname, "joint_0");

// Python:
sensor = spec.add_sensor()
sensor.name = "q0"
sensor.type = mujoco.mjtSensor.mjSENS_JOINTPOS
sensor.objtype = mujoco.mjtObj.mjOBJ_JOINT
sensor.objname = "joint_0"
```

## Notes

- Sensor output is stored in `mjData.sensordata` after simulation; index via `mjModel.sensor_adr[id]`.
- `noise` adds Gaussian noise to the output on each step.
- `cutoff > 0` clamps output to `[-cutoff, cutoff]` for real/positive types.
- For `mjSENS_USER`, set `datatype`, `needstage`, and `dim` explicitly.
- See [[MjData_Sensor_Variables]] for reading sensor output at runtime. See [[mujoco_model_editing_c_api]] for constructor.
