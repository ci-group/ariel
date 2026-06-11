# Custom RL Loop — Phase 1.5 Documentation

## Overview

`rl_loop.py` implements a custom PPO training loop that manages `GeneralistDronePolicy`'s LSTM state correctly across episode boundaries. This avoids SB3's dual-LSTM constraint and enables proper integration with the shared-LSTM + MoE architecture.

## Architecture

### RLLoop Class

```python
from generalist.training.rl_loop import RLLoop
from generalist.policy import GeneralistDronePolicy
from generalist.envs import DroneTrackingEnv
from generalist.morphology.genome import NOMINAL_PHI

# Create loop
policy = GeneralistDronePolicy()
env = DroneTrackingEnv(phi=NOMINAL_PHI)
loop = RLLoop(policy, env, learning_rate=1e-4, n_steps=256, ...)

# Train
checkpoint_path = loop.train(total_timesteps=10000, log_interval=1000)
```

### Key Components

#### 1. **rollout(n_steps)** — Trajectory Collection
- Collects `n_steps` of experience with LSTM state tracking
- Resets LSTM state at episode boundaries (terminal states)
- Returns: `(obs, actions, returns, advantages, values)` as tensors
- Action format: pre-tanh (raw network output) for correct log_prob computation

#### 2. **_compute_gae(rewards, values)** — Advantage Estimation
- Computes returns and advantages via Generalized Advantage Estimation
- Formula: `gae_t = δ_t + γ*λ*gae_{t+1}` where `δ_t = r_t + γ*V(s_{t+1}) - V(s_t)`
- Normalizes advantages for training stability

#### 3. **update(obs, acts, rets, advs)** — PPO Update
- One epoch of minibatch PPO updates
- Loss components:
  - Actor loss: clipped surrogate with advantage
  - Value loss: MSE between predicted value and returns
  - Entropy bonus: encourages exploration
- Total loss: `L = L_actor + 0.5*L_value - 0.001*entropy`
- Gradient clipping: max norm = 1.0

#### 4. **train(total_timesteps, log_interval, save_dir)** — Main Loop
- Runs rollout → update → scheduler step repeatedly
- Logs loss, mean episode return, mean episode length
- Saves final checkpoint at `save_dir/final_policy.pt`
- Cosine annealing learning rate schedule

## LSTM State Management

### Critical Difference from SB3

SB3's RecurrentPPO enforces dual-LSTM (separate actor/critic), but GeneralistDronePolicy uses a single shared LSTM. Our custom loop handles this:

```python
lstm_states = None  # Episode init

for step in range(n_steps):
    if lstm_states is None:
        # Start of episode: None becomes tuple on first forward
        lstm_states = None
    
    action, value, lstm_states = policy(obs, lstm_states=lstm_states)
    # lstm_states is now: (h_fast, c_fast, h_slow, c_slow)
    
    if terminal:
        lstm_states = None  # Reset for next episode
```

### Format

- `lstm_states = None`: Initial state or after episode termination
- `lstm_states = (h_fast, c_fast, h_slow, c_slow)`: 4 tensors, shape (batch=1, hidden_dim)
  - `h_fast`: fast LSTM hidden state (stride=1)
  - `c_fast`: fast LSTM cell state
  - `h_slow`: slow LSTM hidden state (stride=4)
  - `c_slow`: slow LSTM cell state

## Action Format

Critical for PPO: actions are stored **pre-tanh** to enable correct log_prob computation.

```python
# In rollout:
action_mean, value, lstm_states = policy(obs, lstm_states=lstm_states)
action_pre_tanh = action_mean.squeeze(0).cpu().numpy()  # Store this
action = np.tanh(action_pre_tanh)  # For environment

# In update:
log_prob = dist.log_prob(act_batch)  # act_batch is pre-tanh
log_prob -= torch.sum(torch.log(1 - torch.tanh(act_batch) ** 2 + 1e-6), dim=-1)
# ^ This is the tanh Jacobian determinant correction
```

## Hyperparameters

Default values (tunable in constructor):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `learning_rate` | 3e-4 | Adam optimizer LR |
| `n_steps` | 2048 | Rollout length per iteration |
| `batch_size` | 256 | Minibatch size for updates |
| `n_epochs` | 10 | Number of passes over collected data |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing |
| `clip_range` | 0.2 | PPO clipping range |
| `ent_coef` | 0.001 | Entropy bonus coefficient |
| `vf_coef` | 0.5 | Value function loss weight |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |

## Usage Examples

### Example 1: Basic Training

```python
import numpy as np
from generalist.training.rl_loop import RLLoop
from generalist.policy import GeneralistDronePolicy
from generalist.envs import DroneTrackingEnv
from generalist.morphology.genome import NOMINAL_PHI

phi = NOMINAL_PHI
env = DroneTrackingEnv(phi=phi, seed=0)
policy = GeneralistDronePolicy()

loop = RLLoop(
    policy=policy,
    env=env,
    learning_rate=1e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=3,
    gamma=0.99,
    gae_lambda=0.95,
)

checkpoint = loop.train(
    total_timesteps=10000,
    log_interval=256,
    save_dir="checkpoints/rl_experiment_1"
)
print(f"Policy saved to {checkpoint}")
```

### Example 2: Multiple Morphologies (Coming Soon)

```python
from generalist.morphology.genome import sample_valid_morphology

rng = np.random.default_rng(42)
for i in range(10):
    phi = sample_valid_morphology(rng)
    env = DroneTrackingEnv(phi=phi, seed=i)
    
    # ... same training loop
```

### Example 3: Visualize Trained Policy

```bash
PYTHONPATH=. uv run python generalist/examples/run_and_visualize_mujoco.py \
  --rl-checkpoint checkpoints/rl_experiment_1/final_policy.pt \
  --duration 10.0 \
  --morphology nominal
```

## Performance Metrics

### Short Training (512 steps)
- Initial loss: ~2200
- Final loss: ~600
- Episode return improves but remains negative (untrained environment)

### Expected Long-Term (10k+ steps)
- Loss should converge to 200–500 range
- Episode return gradually increases
- Mean episode length should stabilize

## Troubleshooting

### Issue: "The size of tensor a (4) must match the size of tensor b (32)"
**Cause**: Action log_prob shape mismatch in update
**Fix**: Ensure `act_batch` is pre-tanh and sum over action dimension before subtracting tanh correction

### Issue: LSTM states shape mismatch
**Cause**: Incorrect state format or size
**Fix**: Verify policy forward signature returns `(action, value, lstm_states)` where lstm_states is None or 4-tuple of (batch=1, hidden_dim) tensors

### Issue: Episode doesn't terminate, infinite loop
**Cause**: Environment step() not returning `term=True` on crash/timeout
**Fix**: Check DroneTrackingEnv termination conditions (z < 0.05m, pos_error > 10m)

## Architecture Notes

### Why Custom Loop Instead of SB3?

1. **LSTM compatibility**: SB3's RecurrentPPO requires separate actor/critic LSTMs. Our shared LSTM + MoE doesn't fit.
2. **State management**: Direct control over LSTM reset at episode boundaries
3. **Action format**: Pre-tanh storage enables correct log_prob computation
4. **Transparency**: Full visibility into PPO update equations

### Why Pre-Tanh Actions?

The correct PPO formula for continuous control with tanh squashing requires:
- Gaussian distribution over un-squashed actions
- Jacobian determinant correction for the tanh transformation

Storing pre-tanh actions enables this:
```
log_prob = log(N(z; μ, σ)) - Σ log(1 - tanh²(z))
```

## Next Steps

1. **Curriculum callbacks** (Phase 1.5.5):
   - Ramp b2 weight (body rate penalty) after convergence
   - Randomize morphology during training
   - Mix in quintic forest tasks

2. **Dataset scaling**:
   - Train on 200+ morphologies × 20+ trajectory types
   - Measure generalization across unseen morphologies

3. **Physics-based control**:
   - Use mujoco.mj_step() for forward dynamics (not just state replay)
   - Command rotor thrusts directly instead of CTBR

4. **GPU vectorization**:
   - SubprocVecEnv with 64–256 parallel environments
   - Vectorized LSTM state management

## Files

- `rl_loop.py` — RLLoop class implementation
- `test_step9_rl_custom.py` — Unit tests
- `test_phase15_integration.py` — Integration test (train → visualize)
- `run_and_visualize_mujoco.py` — Visualization script with --rl-checkpoint support
