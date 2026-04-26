---
type: api_reference
tags: [mujoco, python, class, model-building]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MjSpec

Programmatic model builder. Allows constructing and editing MuJoCo models in Python without writing XML. Compiles to [[MjModel]].

## Signature

```python
spec = mujoco.MjSpec()                          # empty spec
spec = mujoco.MjSpec.from_file('/path/to.xml')  # parse existing XML
spec = mujoco.MjSpec.from_string(xml_str)       # parse XML string
```

## Compilation Methods

```python
model = spec.compile()
# returns MjModel, or None on error (check spec.error)

model, data = spec.recompile(old_model, old_data)
# in-place update preserving simulation state; returns new objects

xml_str = spec.to_xml()
# serialize back to XML string
```

**Error handling:**
```python
model = spec.compile()
if model is None:
    print(spec.error)
```

## Body Tree

```python
spec.worldbody                   # root body (MjsBody)

# Adding children to a body:
body   = spec.worldbody.add_body()
geom   = body.add_geom()
joint  = body.add_joint()
site   = body.add_site()
camera = body.add_camera()
light  = body.add_light()
frame  = body.add_frame()

# Free joint shorthand:
joint = body.add_freejoint()
```

## Top-Level Element Addition

```python
actuator = spec.add_actuator()
sensor   = spec.add_sensor()
tendon   = spec.add_tendon()
material = spec.add_material()
mesh     = spec.add_mesh()
texture  = spec.add_texture()
key      = spec.add_key()
```

## Named Access

```python
spec.geom('my_geom')   # returns element or None
spec.body('my_body')
```

## List Properties

All element collections are accessible as plural properties:

`spec.bodies`, `spec.geoms`, `spec.joints`, `spec.sites`, `spec.cameras`, `spec.lights`, `spec.frames`, `spec.materials`, `spec.meshes`, `spec.tendons`, `spec.actuators`, `spec.sensors`, `spec.textures`, `spec.texts`, `spec.tuples`, `spec.keys`, `spec.numerics`, `spec.excludes`, `spec.pairs`, `spec.equalities`, `spec.plugins`, `spec.skins`, `spec.flexes`, `spec.hfields`

## Body Subtree Methods

```python
body.geoms                       # direct children geoms
body.find_all('site')            # or mjtObj.mjOBJ_SITE

spec.delete(spec.geom('my_geom'))  # remove element and all referencing elements
```

## Setting Options

```python
spec.option.timestep = 0.002
spec.option.gravity  = [0, 0, -9.81]
```

## Example

```python
import mujoco

spec = mujoco.MjSpec()
spec.option.timestep = 0.002

body = spec.worldbody.add_body()
body.name = 'box'
body.pos  = [0, 0, 1]

geom = body.add_geom()
geom.type = mujoco.mjtGeom.mjGEOM_BOX
geom.size = [0.1, 0.1, 0.1]
geom.mass = 1.0

joint = body.add_freejoint()

model = spec.compile()
data  = mujoco.MjData(model)
```

## Notes

- ⚠️ **`geom.size` always requires exactly 3 elements**, regardless of geom type. The binding enforces shape `[3, 1]`. Always pad unused dimensions with `0`:
  ```python
  # capsule: [radius, half_length, 0]
  geom.size = [0.02, 0.4, 0]
  # sphere: [radius, 0, 0]
  geom.size = [0.08, 0, 0]
  # box: [x, y, z]
  geom.size = [0.1, 0.1, 0.1]
  # plane: [half_x, half_y, 0]
  geom.size = [5, 5, 0]
  ```

- Remove callbacks before recompiling to avoid dangling references:
  ```python
  mujoco.set_mjcb_control(None)
  model = spec.compile()
  mujoco.set_mjcb_control(my_controller)
  ```
- See [[mujoco_callbacks]] for callback management.
