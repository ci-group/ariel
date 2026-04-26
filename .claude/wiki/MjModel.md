---
type: api_reference
tags: [mujoco, python, class]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MjModel

Represents the static model — geometry, physics parameters, and topology. Loaded once; does not change during simulation. Pairs with [[MjData]] which holds the dynamic state.

## Signature

```python
# No direct constructor — use class methods:
model = mujoco.MjModel.from_xml_string(xml_str, assets=None)
model = mujoco.MjModel.from_xml_path('/path/to/model.xml', assets=None)
model = mujoco.MjModel.from_binary_path('/path/to/model.mjb', assets=None)
model = mujoco.MjModel.from_spec(spec)   # from MjSpec
```

`assets` is an optional `dict[str, bytes]` mapping filenames to file contents (virtual filesystem for meshes, textures, etc.).

## Count Attributes

| Attribute | Description |
|-----------|-------------|
| `nq` | number of generalized coordinates (position DoFs) |
| `nv` | number of DoFs (velocity dimensions) |
| `nu` | number of actuators/controls |
| `na` | number of actuator activations |
| `nbody` | number of bodies |
| `njnt` | number of joints |
| `ngeom` | number of geoms |
| `nsite` | number of sites |
| `ncam` | number of cameras |
| `nlight` | number of lights |
| `nmesh` | number of meshes |
| `nsensor` | number of sensors |
| `ntendon` | number of tendons |
| `nkey` | number of keyframes |

## Key Struct Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `opt` | `MjOption` | physics options (timestep, gravity, solver, etc.) |
| `stat` | `MjStatistic` | model statistics (mean body mass, extent, center) |

## Key Array Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `body_pos` | `(nbody, 3)` | body positions relative to parent |
| `body_quat` | `(nbody, 4)` | body orientations relative to parent (quaternion) |
| `body_mass` | `(nbody,)` | body masses |
| `geom_rgba` | `(ngeom, 4)` | geom RGBA colors |
| `geom_size` | `(ngeom, 3)` | geom sizes |
| `geom_type` | `(ngeom,)` | geom types (int, see `mjtGeom`) |
| `jnt_type` | `(njnt,)` | joint types (int, see `mjtJoint`) |
| `jnt_qposadr` | `(njnt,)` | address in `qpos` for each joint |
| `jnt_dofadr` | `(njnt,)` | address in `qvel`/`qacc` for each joint |
| `actuator_trntype` | `(nu,)` | actuator transmission types |
| `cam_fovy` | `(ncam,)` | camera field-of-view, y-axis (degrees) |

## Named Access

`MjModel` supports O(1) named element access via accessor methods. See [[mujoco_named_access_api]].

```python
model.geom('floor').rgba      # (4,) view into geom_rgba
model.geom('floor').id        # integer index
model.body('upper_arm').mass  # float
model.joint('elbow').type     # int (mjtJoint)
model.actuator('motor1').id
```

## Examples

```python
import mujoco

# Load from XML string
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""")

# Load from file with virtual asset filesystem
with open('mesh.stl', 'rb') as f:
    mesh_bytes = f.read()
model = mujoco.MjModel.from_xml_path('model.xml', assets={'mesh.stl': mesh_bytes})

# Access physics options
print(model.opt.timestep)   # e.g. 0.002
print(model.opt.gravity)    # (3,) array

# Access count attributes
print(model.nbody, model.ngeom, model.nu)
```
