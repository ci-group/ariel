# Drone Morphology Evolution + PPO Gate Racing

Evolve drone body shapes (number of arms, arm geometry, motor angles) and
simultaneously train a PPO flight controller for each candidate body on a
quintic gate-racing circuit.

---

## Quick start

```bash
# Default settings — small run to verify everything works (~5 min on GPU)
uv run examples/spear/9_drone_evo_configurable.py --device cuda:0

# Longer research run
uv run examples/spear/9_drone_evo_configurable.py --device cuda:0 --seed 7

# Then visualise
uv run examples/e_drones_ec/6_visualize_evo_results.py __data__/drone_evo_configurable/<RUN_ID>
uv run examples/spear/8_visualize_gate_track.py  --run-dir __data__/drone_evo_configurable/<RUN_ID>
```

---

## Scripts in this directory

| Script | Purpose |
|--------|---------|
| `9_drone_evo_configurable.py` | **Main entry point** — all config at the top |
| `7_drone_evo_rl_quintic.py` | Original script (CLI-only, no inline config) |
| `6_visualize_evo_results.py` | Fitness plot + drone blueprint + MuJoCo rollout video |
| `8_visualize_gate_track.py`  | Visualise the gate circuit in MuJoCo (no drone) |
| `colab_drone_evo.ipynb`      | Ready-to-run Google Colab notebook |

---

## What gets evolved

Each **individual** is a drone body described by a genome of arm parameters:

| Parameter | Range | Meaning |
|-----------|-------|---------|
| Arm length | 0.055 – 0.17 m | Physical reach of each arm |
| Arm azimuth | −π … +π | Horizontal angle around the body |
| Arm elevation | −π/2 … +π/2 | Tilt up or down |
| Motor disc azimuth | −π … +π | Rotation of the motor head around the arm |
| Motor disc pitch | −π … +π | Tilt of the motor thrust axis |
| Spin direction | 0 or 1 | CCW (0) or CW (1) |

Each body is **evaluated by training a PPO controller** from scratch (or warm-started
from the best known policy for the same motor count) and measuring the mean
episode reward over a deterministic rollout.

---

## CONFIG 1 — Evolutionary Algorithm

**Where:** top of `9_drone_evo_configurable.py`, section labelled `CONFIG 1 — EA`

### Population & generations

```python
EA_POP_SIZE = 10   # individuals alive at any time
EA_GENS     = 50   # number of EA generations
```

A generation consists of: select parents → crossover → mutate → evaluate → select survivors.
Larger population = more diversity, slower per-generation wall-clock time.

### Number of arms

```python
N_ARMS_MIN = 6
N_ARMS_MAX = 6   # set > N_ARMS_MIN to allow variable arm count
```

Keeping min == max fixes the arm count so evolution focuses on arm geometry.
Setting `N_ARMS_MAX = 8` and `APPEND_ARM_CHANCE = 0.1` lets evolution also
discover the right number of arms — but makes evaluation slower because
policies don't transfer between different arm counts.

### Genome parameter ranges (`PARAMETER_LIMITS`)

Widen a range to give evolution more room; narrow it to constrain designs.
Example — restrict arm lengths to a narrow band for compact drones:
```python
PARAMETER_LIMITS[0] = [0.06, 0.09]   # arm length: 6–9 cm only
```

### Bilateral symmetry

```python
BILATERAL_SYMMETRY = None   # no constraint
BILATERAL_SYMMETRY = "xz"   # force left-right symmetric designs
```

Symmetric designs are easier to fly and converge faster, but may miss
asymmetric optima. See `SphericalAngularDroneGenomeHandler` for details.

### Mutation step sizes

```python
MUTATION_SCALES = None   # library default (~5 % of each parameter range)

# Custom: tight arm-length mutations, larger angle mutations
MUTATION_SCALES = np.array([0.03, 0.10, 0.10, 0.10, 0.10, 0.50])
```

### Custom generation pipeline

Replace the default `(parent_tag → crossover → mutate → evaluate → truncate)`
pipeline by setting `CUSTOM_GENERATION_OPS`. Use `None` as a placeholder where
`eval_op` should be inserted:

```python
CUSTOM_GENERATION_OPS = [
    parent_tag(n=EA_POP_SIZE // 2),
    crossover_drones(template_handler=genome_handler),
    mutate_drones(template_handler=genome_handler),
    None,                            # ← eval_op is spliced in here
    truncation_select(n=EA_POP_SIZE),
]
```

**Available operations** (from `ariel.body_phenotypes.drone`):

| Factory | Effect |
|---------|--------|
| `parent_tag(n=k)` | Mark top-k alive individuals as parents for crossover |
| `crossover_drones(template_handler=...)` | NEAT crossover → one offspring per parent pair |
| `mutate_drones(template_handler=...)` | Gaussian perturbation of all unevaluated individuals |
| `truncation_select(n=k)` | Kill everyone outside the top-k by fitness |

**Source:** `src/ariel/body_phenotypes/drone/operations.py`

---

## CONFIG 2 — Gate Track

**Where:** `CONFIG 2 — GATE TRACK` section

```python
GATE_PATH_STEPS = 15    # number of gates on the circuit
GATE_PATH_SCALE = 5.0   # track spread in metres (gates at ≈ ±5 m)
GATE_Z_HEIGHT   = -1.5  # gate altitude: 1.5 m above ground (NED convention)
GATE_MODE       = "naive"  # "naive" = fixed track | "online" = infinite stream
```

### Difficulty tips

| Goal | Change |
|------|--------|
| Easier task to bootstrap | Fewer gates (`GATE_PATH_STEPS = 6`), smaller scale (`GATE_PATH_SCALE = 2.0`) |
| Harder / longer circuit | More gates (`GATE_PATH_STEPS = 25`), larger scale (`GATE_PATH_SCALE = 8.0`) |
| Infinite randomised track | `GATE_MODE = "online"` |

### Visualise the track first

```bash
# Generate and view a fresh track with these settings
uv run examples/spear/8_visualize_gate_track.py \
    --path-steps 15 --path-scale 5.0 --seed 42

# Or load the saved track from a previous run
uv run examples/spear/8_visualize_gate_track.py \
    --run-dir __data__/drone_evo_configurable/<RUN_ID>
```

**Source:** `src/ariel/simulation/tasks/torch_drone_gate_env.py` (reward function, gate passing logic)

---

## CONFIG 3 — PPO / RL

**Where:** `CONFIG 3 — PPO / RL` section

### Training budget

```python
PPO_STEPS    = 1_000_000   # timesteps per individual (~9 s on RTX 5070 Ti + 2000 envs)
PPO_NUM_ENVS = 2000        # parallel environments (increase on better GPUs)
```

| Hardware | Recommended `PPO_NUM_ENVS` | Per-individual time |
|----------|---------------------------|---------------------|
| CPU only | 64–256 | ~60–120 s |
| T4 (Colab free) | 1000–2000 | ~20–30 s |
| RTX 5070 Ti Laptop | 2000–4000 | ~9–16 s |
| A100 (Colab Pro+) | 8000–16000 | ~4–8 s |

### Network architecture

```python
PPO_NET_ARCH   = dict(pi=[256, 256], vf=[256, 256])  # two hidden layers of 256
PPO_ACTIVATION = torch.nn.SiLU
```

Wider / deeper networks are more expressive but train slower. For hover
tasks `[64, 64]` is often sufficient; for gate racing `[256, 256]` is a good
default.

### PPO hyperparameters

```python
PPO_N_EPOCHS   = 20      # gradient passes per rollout (more = slower but stabler)
PPO_GAMMA      = 0.999   # discount (high = long-horizon)
PPO_LR         = 1e-3    # learning rate
PPO_CLIP_RANGE = 0.2     # clipping (lower = more conservative updates)
PPO_ENT_COEF   = 0.01    # entropy bonus (higher = more exploration)
```

**SB3 documentation:** https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

---

## Repository map — where to find things

```
ariel/
├── examples/spear/
│   ├── 9_drone_evo_configurable.py   ← YOU ARE HERE
│   ├── 7_drone_evo_rl_quintic.py     ← original script (CLI-only)
│   ├── 8_visualize_gate_track.py     ← visualise the gate circuit
│   └── (colab_drone_evo.ipynb lives in examples/e_drones_ec/)
│
├── examples/e_drones_ec/
│   ├── 6_visualize_evo_results.py    ← visualise a finished run
│   └── colab_drone_evo.ipynb         ← run on Google Colab
│
├── src/ariel/
│   ├── ec/
│   │   ├── ea.py                     ← EA loop implementation
│   │   ├── individual.py             ← Individual / Population dataclasses
│   │   └── drone/
│   │       ├── genome_handlers/
│   │       │   └── spherical_angular_genome_handler.py  ← mutation & crossover
│   │       └── strategies/
│   │           └── evolution_components.py              ← low-level EA ops
│   │
│   ├── body_phenotypes/drone/
│   │   ├── operations.py             ← EAOperation factories (crossover_drones etc.)
│   │   ├── decoders.py               ← genome → DroneBlueprint
│   │   └── backends.py               ← DroneBlueprint → propeller list / MuJoCo spec
│   │
│   └── simulation/tasks/
│       ├── torch_drone_gate_env.py   ← GPU-accelerated gate environment
│       └── drone_gate_env.py         ← CPU reference environment
│
└── goal_generator_ltu/
    └── polynomial_goal_generator/
        └── planner_generator.py      ← quintic path generation
```

---

## Typical fitness progression

| Generation range | What's happening |
|-----------------|-----------------|
| 0 – 10 | Most individuals crash immediately; a few learn to hover |
| 10 – 25 | Stable hoverers emerge; best fitness crosses 0 |
| 25 – 40 | Gate approach learned; reward climbs steadily |
| 40 – 50 | Fine-tuning; diminishing returns |

A final reward of **> 5** is good; **> 10** is excellent for 50 generations.
Run 100+ generations and increase `PPO_STEPS` to push further.

---

## Output files

All outputs are written to `__data__/drone_evo_configurable/<RUN_ID>/`:

| File | Contents |
|------|---------|
| `database_*.db` | SQLite DB — every individual's genotype, fitness, and policy |
| `best_blueprint_*.json` | Best drone body as a DroneBlueprint JSON |
| `best_policy_*.zip` | Best PPO policy (SB3 format) |
| `gate_pos_*.npy` | Gate positions used during the run (NED, shape N×3) |
| `gate_yaw_*.npy` | Gate yaw angles (radians, shape N) |
