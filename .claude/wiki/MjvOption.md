---
type: api_reference
tags: [mujoco, python, class, rendering, visualization]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MjvOption

Visualization flags and options. Controls what is drawn in the scene. Used with [[mujoco_rendering_pipeline]].

## Signature

```python
opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)   # initialize to defaults
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `flags` | `(mjNVISFLAG,)` bool array | visualization flags indexed by `mjtVisFlag` |
| `geomgroup` | `(mjNGROUP,)` | per-group geom visibility flags |
| `sitegroup` | `(mjNGROUP,)` | per-site-group visibility |
| `jointgroup` | `(mjNGROUP,)` | per-joint-group visibility |
| `tendongroup` | `(mjNGROUP,)` | per-tendon-group visibility |
| `actuatorgroup` | `(mjNGROUP,)` | per-actuator-group visibility |
| `label` | int | label mode (`mjtLabel`) |
| `frame` | int | frame visualization mode (`mjtFrame`) |

## Common Visualization Flags (`mjtVisFlag`)

Toggle via `opt.flags[mujoco.mjtVisFlag.mjVIS_XXX] = True/False`:

`mjVIS_CONVEXHULL`, `mjVIS_TEXTURE`, `mjVIS_JOINT`, `mjVIS_CAMERA`, `mjVIS_ACTUATOR`, `mjVIS_ACTIVATION`, `mjVIS_LIGHT`, `mjVIS_TENDON`, `mjVIS_RANGEFINDER`, `mjVIS_CONSTRAINT`, `mjVIS_INERTIA`, `mjVIS_PERTFORCE`, `mjVIS_PERTOBJ`, `mjVIS_CONTACTPOINT`, `mjVIS_CONTACTFORCE`, `mjVIS_COM`, `mjVIS_SELECT`, `mjVIS_STATIC`, `mjVIS_SKIN`

## Example

```python
opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)

# Show contact points and forces
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

# Hide textures
opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = False
```

See [[MjvCamera]], [[MjvScene]], [[mujoco_enumerations]].
