# RL Training Quick Start

Simple parametrizable script for training the generalist drone controller.

## Usage

```bash
cd /home/user/Desktop/EvoDevo/ariel
uv run python train_rl_policy.py [OPTIONS]
```

## Examples

### Quick test (512 steps, ~1 minute)
```bash
uv run python train_rl_policy.py --steps 512
```

### Standard training (10k steps, ~10 minutes)
```bash
uv run python train_rl_policy.py --steps 10000
```

### Longer training (50k steps, ~1 hour)
```bash
uv run python train_rl_policy.py --steps 50000
```

### Random morphology
```bash
uv run python train_rl_policy.py --steps 10000 --morphology random
```

### Custom hyperparameters
```bash
uv run python train_rl_policy.py \
  --steps 20000 \
  --morphology nominal \
  --lr 5e-5 \
  --batch-size 128 \
  --n-epochs 5 \
  --clip-range 0.15
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--steps` | int | 10000 | Total training steps |
| `--morphology` | {nominal, random} | nominal | Drone morphology |
| `--seed` | int | 0 | Random seed |
| `--lr` | float | 1e-4 | Learning rate |
| `--n-steps` | int | 256 | Rollout length per iteration |
| `--batch-size` | int | 64 | Minibatch size |
| `--n-epochs` | int | 3 | Epochs per update |
| `--gamma` | float | 0.99 | Discount factor |
| `--gae-lambda` | float | 0.95 | GAE smoothing |
| `--clip-range` | float | 0.2 | PPO clip range |
| `--ent-coef` | float | 0.001 | Entropy bonus coefficient |
| `--save-dir` | str | auto | Checkpoint save directory |

## What You'll See

Training output:
```
======================================================================
GENERALIST DRONE CONTROLLER — RL TRAINING
======================================================================

📦 Morphology: nominal
   φ shape: (16,)

🌍 Environment
   Observation dim: (73,)
   Action dim: (4,)

🧠 Policy: GeneralistDronePolicy
   Parameters: 276,584

⚙️  Training Configuration
   Total steps: 10000
   ...

🚀 Creating RLLoop...
✓ RLLoop created

======================================================================
TRAINING IN PROGRESS
======================================================================

Step 256: loss=2332.9, mean_return=-208.9, mean_length=51
Step 512: loss=1847.3, mean_return=-195.4, mean_length=49
Step 768: loss=1562.1, mean_return=-185.6, mean_length=48
Step 1024: loss=944.6, mean_return=-170.7, mean_length=49
...

Saved final policy to checkpoints/rl_nom_20260611_112054/final_policy.pt

======================================================================
TRAINING COMPLETE ✓
======================================================================

📊 Results:
   Checkpoint: checkpoints/rl_nom_20260611_112054/final_policy.pt
   Mean episode return: -170.71
   Mean episode length: 49

🎬 To visualize the trained policy:
   PYTHONPATH=. uv run python generalist/examples/run_and_visualize_mujoco.py \
     --rl-checkpoint checkpoints/rl_nom_20260611_112054/final_policy.pt \
     --duration 10.0 \
     --morphology nominal
```

## Visualizing Results

After training completes, the script prints the exact command to visualize:

```bash
uv run python generalist/examples/run_and_visualize_mujoco.py \
  --rl-checkpoint checkpoints/rl_nom_20260611_112054/final_policy.pt \
  --duration 10.0 \
  --morphology nominal
```

To see the MuJoCo viewer, remove `--no-render` (if you have a display). The script includes `--no-render` by default for headless environments.

## Expected Results

### Loss
- Initial: ~2000–3000 (untrained)
- Final: ~500–1000 (after 10k steps)
- Trend: Should generally decrease

### Episode Return
- Initial: -200 to -300 (poor control)
- Final: Improves toward less negative (better control)
- Note: Still negative because reward function includes penalties

### Episode Length
- Initial: 40–60 steps
- Final: 50–100 steps
- Better control = longer episodes before crash

## Troubleshooting

### Out of memory
Reduce batch size or n_steps:
```bash
PYTHONPATH=. uv run python train_rl_policy.py \
  --steps 10000 \
  --n-steps 128 \
  --batch-size 32
```

### Training is slow
- Reduce `n_epochs` (3 → 2)
- Reduce `n_steps` (256 → 128)
- Reduce `--steps` for testing

### Checkpoint not found for visualization
Check the printed path from training. Checkpoints are auto-saved to `checkpoints/rl_*_TIMESTAMP/final_policy.pt`

## Files Generated

```
checkpoints/
└── rl_nom_20260611_112054/
    └── final_policy.pt          # Trained policy weights
```

Checkpoint path is printed at the end of training and ready to use immediately.
