# spear

Build a morphing drone from an airevolve Spherical Angular genome and
simulate it in Isaac Lab.

A genome is a `(narms, 6)` numpy array (`.npy`), one row per rotor, NaN rows = unused arm:

```
[magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
```

Each arm gets a shoulder (Z), elbow (Y) and motor-tilt (X) revolute joint plus a spinning rotor.

## Usage

```bash
# URDF only (plain Python + numpy)
python spear/drone_urdf.py [--genome g.npy]

# URDF -> USD -> Isaac Sim viewer with joint sliders
<IsaacLab>/isaaclab.sh -p spear/view_drone.py [--genome g.npy] [--gravity] [--teleport] [--viz newton]
```

Without `--genome` a built-in 4-rotor sample is used. Output goes to `spear/output/{urdf,usd}/`.
