---
type: api_reference
tags: [mujoco, mjspec, model-building, c-api]
source: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html
date_ingested: 2026-04-13
---
# mjsActuator

C struct representing an actuator element in an [[MjSpec]] model specification. Corresponds to the MJCF `<actuator>` element.

## Signature

```c
typedef struct mjsActuator_ {
  mjsElement* element;             // element type (do not modify)

  // gain, bias
  mjtGain gaintype;                // gain type
  double gainprm[mjNGAIN];         // gain parameters
  mjtBias biastype;                // bias type
  double biasprm[mjNGAIN];         // bias parameters

  // activation state
  mjtDyn dyntype;                  // dynamics type
  double dynprm[mjNDYN];           // dynamics parameters
  int actdim;                      // number of activation variables
  mjtByte actearly;                // apply next activations to qfrc

  // transmission
  mjtTrn trntype;                  // transmission type
  double gear[6];                  // length and transmitted force scaling
  mjString* target;                // name of transmission target
  mjString* refsite;               // reference site (for site transmission)
  mjString* slidersite;            // site defining cylinder (for slider-crank)
  double cranklength;              // crank length (for slider-crank)
  double lengthrange[2];           // transmission length range
  double inheritrange;             // automatic range for position/intvelocity

  // input/output clamping
  int ctrllimited;                 // control limits defined (mjtLimited)
  double ctrlrange[2];             // control range [min, max]
  int forcelimited;                // force limits defined (mjtLimited)
  double forcerange[2];            // force range [min, max]
  int actlimited;                  // activation limits defined (mjtLimited)
  double actrange[2];              // activation range [min, max]

  // other
  int group;                       // group
  mjDoubleVec* userdata;           // user data
  mjsPlugin plugin;                // actuator plugin
  mjString* info;                  // message appended to compiler errors
} mjsActuator;
```

## Parameters

| Field | Type | Description |
|---|---|---|
| `gaintype` | `mjtGain` | `mjGAIN_FIXED`, `mjGAIN_AFFINE`, `mjGAIN_MUSCLE`, `mjGAIN_USER` |
| `gainprm` | `double[mjNGAIN]` | Gain parameters (depends on `gaintype`) |
| `biastype` | `mjtBias` | `mjBIAS_NONE`, `mjBIAS_AFFINE`, `mjBIAS_MUSCLE`, `mjBIAS_USER` |
| `biasprm` | `double[mjNGAIN]` | Bias parameters |
| `dyntype` | `mjtDyn` | `mjDYN_NONE`, `mjDYN_INTEGRATOR`, `mjDYN_FILTER`, `mjDYN_FILTEREXACT`, `mjDYN_MUSCLE`, `mjDYN_USER` |
| `dynprm` | `double[mjNDYN]` | Dynamics parameters |
| `actdim` | `int` | Number of activation variables (≥1 if dyntype != NONE) |
| `trntype` | `mjtTrn` | `mjTRN_JOINT`, `mjTRN_JOINTINPARENT`, `mjTRN_SLIDERCRANK`, `mjTRN_TENDON`, `mjTRN_SITE`, `mjTRN_BODY` |
| `gear` | `double[6]` | Length and force scaling |
| `target` | `mjString*` | Name of joint/tendon/site/body being actuated |
| `ctrllimited` | `int` | Control clamping active (`mjtLimited`) |
| `ctrlrange` | `double[2]` | Control input range `[min, max]` |
| `forcelimited` | `int` | Force output clamping active (`mjtLimited`) |
| `forcerange` | `double[2]` | Force output range `[min, max]` |
| `actlimited` | `int` | Activation clamping active (`mjtLimited`) |
| `actrange` | `double[2]` | Activation range `[min, max]` |

## Actuator Shorthand Setters

These functions configure `gaintype`, `biastype`, `dyntype`, and `trntype` in one call:

```c
// Motor: direct force control
const char* mjs_setToMotor(mjsActuator* actuator);

// Position servo (PD-like)
const char* mjs_setToPosition(mjsActuator* actuator, double kp,
                               double kv[1],          // optional velocity gain
                               double dampratio[1],   // optional damping ratio
                               double timeconst[1],   // optional filter time constant
                               double inheritrange);  // if >0, inherit joint range

// Integrated velocity servo
const char* mjs_setToIntVelocity(mjsActuator* actuator, double kp,
                                  double kv[1], double dampratio[1],
                                  double timeconst[1], double inheritrange);

// Velocity servo
const char* mjs_setToVelocity(mjsActuator* actuator, double kv);

// Damper
const char* mjs_setToDamper(mjsActuator* actuator, double kv);

// Cylinder (hydraulic/pneumatic)
const char* mjs_setToCylinder(mjsActuator* actuator, double timeconst,
                               double bias, double area, double diameter);

// Muscle
const char* mjs_setToMuscle(mjsActuator* actuator,
                             double timeconst[2],  // activation/deactivation
                             double tausmooth,
                             double range[2],      // operating length range
                             double force,         // max isometric force
                             double scale,         // peak force at zero velocity
                             double lmin, double lmax,  // force-length curve
                             double vmax,          // shortening velocity
                             double fpmax,         // peak passive force
                             double fvmax);        // peak active force

// Adhesion
const char* mjs_setToAdhesion(mjsActuator* actuator, double gain);
```

All `setTo*` functions return `NULL` on success or an error string on failure.

## Examples

```c
// C: position servo on a joint
mjsActuator* act = mjs_addActuator(spec, NULL);
mjs_setName(act->element, "servo_0");
mjs_setString(act->target, "joint_0");
act->trntype = mjTRN_JOINT;
const char* err = mjs_setToPosition(act, 50.0, NULL, NULL, NULL, 0.0);
if (err) fprintf(stderr, "actuator error: %s\n", err);

// Python:
actuator = spec.add_actuator()
actuator.name = "servo_0"
actuator.target = "joint_0"
actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
actuator.gainprm[0] = 1.0
actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
actuator.biasprm[1] = -1.0  # position feedback
```

## Notes

- `setTo*` functions are the recommended way to configure actuator types — they handle all the parameter interdependencies.
- For `trntype = mjTRN_JOINT`, set `target` to the joint name.
- `gear[0]` is the transmission ratio; default is 1.0.
- See [[mjsSensor]] for paired sensing. See [[mujoco_model_editing_c_api]] for constructor functions.
