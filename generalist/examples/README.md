# Generalist Drone Controller Examples

## Run and Visualize in MuJoCo

The `run_and_visualize_mujoco.py` script demonstrates end-to-end trajectory rollout with MuJoCo visualization.

### What It Does

1. **Generates/loads a morphology** (nominal or random)
2. **Rolls out a trajectory** using the geometric expert controller
3. **Converts to ARIEL blueprint** — bridges our φ representation to ARIEL's drone specification
4. **Compiles to MuJoCo** — uses ARIEL's `blueprint_to_mjspec` backend to create a full 3D drone model
5. **Reports tracking metrics** — mean/max/RMS position error vs. reference

### Usage

```bash
cd /home/user/Desktop/EvoDevo/ariel
PYTHONPATH=. uv run python generalist/examples/run_and_visualize_mujoco.py [OPTIONS]
```

### Options

- `--duration FLOAT` — simulation duration in seconds (default: 20.0)
- `--task {circle,figure8}` — reference trajectory (default: circle)
- `--morphology {nominal,random}` — use nominal quad or random sample (default: nominal)
- `--no-expert` — use random actions instead of geometric expert
- `--no-render` — skip visualization (default: render enabled)
- `--output-video PATH` — save video to file (not yet implemented)

### Examples

#### Quick test: 10s circle with nominal quad
```bash
PYTHONPATH=. uv run python generalist/examples/run_and_visualize_mujoco.py \
  --duration 10.0 --task circle --morphology nominal --no-render
```

Output:
```
Morphology: nominal
Task: circle, Duration: 10.0s, Steps: 500
Trajectory: 499 steps
Blueprint: DroneBlueprint (13 nodes)
  #0 CorePlateNode (...)
  #1-#12 ArmNode / MotorNode / RotorNode tree
MuJoCo model compiled: 11 bodies, 0 q dims

Trajectory Summary:
  Mean error: 0.249 m
  Max error: 0.383 m
  RMS error: 0.251 m
```

#### Random morphology, longer episode
```bash
PYTHONPATH=. uv run python generalist/examples/run_and_visualize_mujoco.py \
  --duration 30.0 --task figure8 --morphology random --no-render
```

#### Random baseline (no expert)
```bash
PYTHONPATH=. uv run python generalist/examples/run_and_visualize_mujoco.py \
  --morphology random --no-expert --no-render
```

#### Visualize RL-trained policy
```bash
PYTHONPATH=. uv run python generalist/examples/run_and_visualize_mujoco.py \
  --rl-checkpoint checkpoints/rl_custom/final_policy.pt \
  --duration 10.0 --morphology nominal
```

#### Train RL policy then visualize
```bash
# Train for 10k steps
python -c "
from generalist.training.rl_loop import RLLoop
from generalist.policy import GeneralistDronePolicy
from generalist.envs import DroneTrackingEnv
from generalist.morphology.genome import NOMINAL_PHI

env = DroneTrackingEnv(phi=NOMINAL_PHI)
policy = GeneralistDronePolicy()
loop = RLLoop(policy, env, n_steps=256, batch_size=64)
checkpoint = loop.train(total_timesteps=10000, save_dir='checkpoints/rl_custom')
print(f'Saved to: {checkpoint}')
"

# Visualize the trained policy
PYTHONPATH=. uv run python generalist/examples/run_and_visualize_mujoco.py \
  --rl-checkpoint checkpoints/rl_custom/final_policy.pt
```

### Script Output Breakdown

1. **Morphology Summary**: φ vector shape (16,) and control/tracking parameters
2. **Task Info**: reference trajectory steps, duration
3. **Blueprint Tree**: ARIEL's hierarchical representation of the drone
   - Core (central plate)
   - 4 Arms (with attachment poses)
   - 4 Motors (with rotation directions: cw/ccw)
   - 4 Rotors (propeller geometry)
4. **MuJoCo Compilation**: model size (bodies, q dims) and physics parameters
5. **Tracking Metrics**: error statistics across the rollout

### Phase 1.5 Additions

- **MuJoCo viewer**: ✅ Full viewer support with freejoint state replay (Phase 1.5)
- **RL policy visualization**: ✅ Can load RL-trained checkpoints via `--rl-checkpoint` (Phase 1.5)
- **Video output**: `--output-video` parameter accepted but not yet implemented

### Integration with Generalist Stack

```
Our φ (16D)
    ↓
DroneTrackingEnv (rollout simulator)
    ↓
Trajectory + Error metrics
    ↓
Convert φ → ARIEL genome (6×4 format)
    ↓
spherical_angular_to_blueprint()
    ↓
blueprint_to_mjspec()
    ↓
MuJoCo model (physics simulation ready)
```

### Files

- `run_and_visualize_mujoco.py` — main script
- `README.md` — this file

### See Also

- `generalist/envs/drone_tracking_env.py` — trajectory rollout
- `generalist/expert/geometric_controller.py` — control policy
- `src/ariel/body_phenotypes/drone/backends.py` — blueprint compilation
- `GENERALIST_QUICK_START.md` — quick overview of full stack
