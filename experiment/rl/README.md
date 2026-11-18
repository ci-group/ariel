# PPO on ARIEL morphologies (mujoco-playground + brax)

GPU-accelerated PPO training for ARIEL modular robots, built on top of the
vendored [mujoco-playground](../mujoco_playground/) stack. The training script
wraps an ariel-built MJCF as a `mujoco_playground._src.mjx_env.MjxEnv` and
runs brax PPO on it with MuJoCo-Warp.

- `undirected_locomotion_ppo_jax.py` — flat-MLP policy; rewards displacement
  in any direction (undirected locomotion).
- `undirected_locomotion_ppo_gnn_jax.py` — same task with a Graph Neural
  Network policy; morphology-independent weights enable transfer across
  `insect_n` sizes (see §5).
- `rerun_undirected_locomotion_ppo_jax.py` — reload a trained run and render a
  fresh rollout video + trajectory plot without retraining.

Supported morphologies: `insect` (fixed), `centipede`, `insect_n`, and
`insect_longspine_n` (all parameterised by segment count).

## 1. Install

PPO uses a dedicated virtualenv living inside the vendored playground, **not**
the top-level ariel `uv sync` venv. Ariel is wired in as an editable path
dependency, so edits to `src/ariel/` are picked up without reinstalling.

```bash
cd experiment/mujoco_playground
uv venv --python 3.12                 # must be 3.12 — playground pins <3.13
source .venv/bin/activate
uv sync --all-extras                  # installs playground + ariel (editable) + CUDA jax
```

Verify the GPU backend is live before you train:

```bash
uv --no-config run python -c "import jax; print(jax.default_backend(), jax.devices())"
# → gpu [CudaDevice(id=0)]
```

If it prints `cpu`, run `unset LD_LIBRARY_PATH` and retry — a stale
`LD_LIBRARY_PATH` from a system CUDA shadows the jax wheels.

### Shell gotcha

A sibling checkout at `~/Desktop/EvoDevo/mujoco_playground/.venv` (outside this
repo) has the same name. If `echo $VIRTUAL_ENV` points outside `ariel/`,
deactivate and re-source the local `.venv/bin/activate` — otherwise
`uv pip install` writes to the wrong env.

### Fresh-clone checklist

Things that trip people up on a brand-new machine, in one place:

1. **Python must be 3.12.** The playground pins `<3.13`. If `uv venv` picks a
   different interpreter, pass `--python 3.12` explicitly.
2. **Use the playground venv, not the top-level ariel one.** `uv sync` at the
   repo root creates a different venv that will not have brax / mujoco-warp /
   jax[cuda]. All PPO commands assume
   `experiment/mujoco_playground/.venv` is active.
3. **`ariel` is an editable path dep** (`{path = "../..", editable = true}` in
   `experiment/mujoco_playground/pyproject.toml`). This resolves correctly
   only because you ran `uv sync` from inside `experiment/mujoco_playground/`.
   Edits to `src/ariel/` are picked up without reinstalling.
4. **`jax.default_backend()` must print `gpu`.** If it prints `cpu`, run
   `unset LD_LIBRARY_PATH` and retry — a stale system-CUDA `LD_LIBRARY_PATH`
   shadows the jax wheel's bundled CUDA libs.
5. **Check `$VIRTUAL_ENV` points inside `ariel/`** before any `uv pip install`
   — see the sibling-checkout gotcha above.

## 2. Train

All commands assume you've activated the venv at
`experiment/mujoco_playground/.venv`. Run from the repo root.

### Insect (6 legs, fixed topology)

```bash
python experiment/rl/undirected_locomotion_ppo_jax.py \
    --morphology insect \
    --num-timesteps 20000000
```

### Centipede (variable leg pairs)

```bash
python experiment/rl/undirected_locomotion_ppo_jax.py \
    --morphology centipede \
    --n-pairs 2 \
    --num-timesteps 20000000
```

`--n-pairs` is the number of leg-bearing spine segments; total legs = 2 ×
`n_pairs`. `n_pairs=2` fits in the default budget. `n_pairs>=3` roughly
doubles geom count and needs the halved `--num-envs` recipe below.

### Full CLI

| flag                     | default                                               | meaning                                                    |
|--------------------------|-------------------------------------------------------|------------------------------------------------------------|
| `--num-timesteps`        | `1_000_000`                                           | total env steps across all workers                         |
| `--seed`                 | `0`                                                   | PRNG seed                                                  |
| `--outdir`               | `experiment/__data__/undirected_locomotion_ppo_jax/`  | parent dir; each run gets its own timestamped subdir       |
| `--episode-length`       | `500`                                                 | post-training rollout length (steps)                       |
| `--morphology`           | `insect`                                              | `insect` or `centipede`                                    |
| `--n-pairs`              | `2`                                                   | centipede leg-pair count                                   |
| `--num-envs`             | `6144`                                                | parallel MJX worlds — the main VRAM lever                  |
| `--naconmax-per-env`     | `320`                                                 | contact-pool slack per world                               |
| `--naccdmax-per-env`     | `100`                                                 | CCD workspace per world                                    |

## 3. VRAM and solver budget

The three numbers that decide whether training fits on your GPU are
`NUM_ENVS`, `naconmax_per_env`, and `naccdmax_per_env`. The defaults target a
~12 GiB card (RTX 5070 Ti class) running `insect`. Rough budget guide:

| GPU VRAM | morphology            | suggested `--num-envs` |
|----------|-----------------------|------------------------|
| 24 GiB   | insect / centipede≤3  | 12288                  |
| 12 GiB   | insect                | 6144 (default)         |
| 12 GiB   | centipede, n_pairs≥3  | 3072                   |
| 8 GiB    | insect                | 3072, maybe 2048       |

The brax batch size is derived automatically — `batch_size * num_minibatches
== BATCH_RATIO * num_envs` is enforced, so changing `--num-envs` can never
desync the PPO assertion.

### If training OOMs

The failure almost always points at the WARP contact or CCD pool. Typical
knobs, in order of effectiveness:

1. **Halve `--num-envs`** — frees linearly, no quality cost beyond slower
   wall-clock. First thing to try for a new morphology.
2. **Lower `--naccdmax-per-env`** — CCD workspace is the single biggest
   per-world buffer. `100` is enough for walking gaits; drop to `80` only if
   you confirm no CCD overflows in the log.
3. **Lower `--naconmax-per-env`** — only safe if contact logs stay well under
   the limit. The ~10 % safety margin in the default is intentional.

The `nefc` / contact-pool peaks that justify the defaults (nefc ≈144 for
random control, CCD ≈135 observed mid-training) are per-env; **total**
`naconmax` / `naccdmax` scale with `--num-envs`. Reducing `--num-envs` already
reduces total pool size, so you rarely need to touch the per-env knobs.

### If eval OOMs after training completes

Eval runs with `num_eval_envs=128` worlds, way below `NUM_ENVS`, and training
buffers stay pinned in the cuda_malloc_async pool during handover. The script
calls `jax.clear_caches()` + `gc.collect()` between phases, but if the
single-env rollout still trips OOM you can rerun it offline (see §5).

## 4. Outputs

Each run creates `experiment/__data__/undirected_locomotion_ppo_jax/<morph>_undirected-<timestamp>/`
containing:

- `env_config.json`, `eval_env_config.json`, `ppo_config.json` — the exact
  configs used (consumed by the rerun script).
- `training_history.json` — eval reward curve (steps / mean / std).
- `reward_curve.png` — plotted eval reward.
- `params.pkl` — final PPO params (brax `model.save_params` pickle).
- `params_snapshots.pkl` — list of `(step, params)` tuples captured each eval;
  consumed by `experiment/plotting/fitness_landscape_ppo_pca.py`.
- `rollout.mp4` + `trajectory.png` — single-env replay of the trained policy
  (unless render skipped due to OOM; see §3).

## 5. GNN controller (`undirected_locomotion_ppo_gnn_jax.py`)

A drop-in alternative to the flat-MLP baseline that represents the robot as a
graph and passes messages along kinematic edges.  The key property: every Dense
layer operates per-node or per-edge and has no dimension that depends on
morphology size, so the **same 51k parameter set applies to any
`insect_n(n_segments)`** — enabling zero-shot transfer or warm-start
fine-tuning from a checkpoint trained on a smaller robot.

### How it works

**Graph structure**

Each joint becomes a node. Node 0 is the torso; nodes 1…nu are the actuators,
in MuJoCo actuator order:

```
node 0            torso
nodes 1..n_seg    spine hinges H0..H_{n-1}
node n_seg+1      core right leg
node n_seg+2      core left leg
node n_seg+3+2k   spine-brick k right leg   (k = 0..n_seg-1)
node n_seg+4+2k   spine-brick k left leg
```

Edges (bidirectional):

- Spine chain: torso → H0 → H1 → … → H_{n-1}
- Core legs: torso ↔ core_right ↔ core_left
- Each spine hinge ↔ its own segment's leg pair (bilateral coordination)
- Left-right siblings within each leg pair

**Node features (24-dim, fixed regardless of morphology)**

```
[:3]   local   joint_angle, joint_vel, last_act   (zeros for torso)
[3:]   global  height(1), rot(9), linvel(3), angvel(3), up(3), phase(2)
```

The global state is broadcast to every node, so each actuator sees both its own
local state and the full body context.

**Message passing (K rounds)**

Each round per node i:

1. For every edge (j→i): compute `msg = MLP([h_j, h_i])` — the sender's and
   receiver's current hidden states.  The MLP learns *what information is
   relevant to pass*, conditioned on who is receiving.
2. Aggregate: `agg_i = Σ_{j→i} msg_{j→i}` (sum over neighbors).
3. Update: `h_i ← h_i + MLP([h_i, agg_i])` — residual connection keeps
   gradient flow healthy across K rounds.

After K rounds, `h_i` encodes i's local state *plus* K hops of neighborhood
context.  A final `Dense(2)` per actuator node outputs `[mean, raw_scale]` for
the action distribution (no separate std parameter, so the whole policy is
morphology-independent).  The value network mean-pools all node hidden states
then passes through a small MLP → scalar baseline.

**Why transfer works**

| layer | input | output | depends on nu? |
|-------|-------|--------|----------------|
| embed | 24 | 64 | no |
| msg_k | 128 | 64 | no |
| update_k | 128 | 64 | no |
| policy_out | 64 | 2 | no |
| value MLP | 64 | 64 → 1 | no |

Verified: n_segments = 1, 2, 3, 4 all produce **identical param shapes** and
the same 51,266-parameter count.  When you scale to a larger morphology the
graph has more nodes, but each node runs the same learned MLPs.  Only the
obs-normalizer (which tracks per-dimension running statistics for the flat
obs vector of length `21 + 3*nu`) must be re-initialised, since `nu` changes.

### Train from scratch

```bash
# activate the mujoco_playground venv first (see §1)
python experiment/rl/undirected_locomotion_ppo_gnn_jax.py \
    --morphology insect_n \
    --n-pairs 2 \
    --num-timesteps 20000000
```

### Warm-start from a smaller morphology

```bash
python experiment/rl/undirected_locomotion_ppo_gnn_jax.py \
    --morphology insect_n \
    --n-pairs 4 \
    --transfer-from experiment/__data__/undirected_locomotion_ppo_gnn_jax/insect_n_n2_gnn_.../params.pkl
```

The script calls `create_transfer_params()`, which builds a fresh normalizer
for the target `obs_dim = 21 + 3*nu_target` and copies the GNN body weights
verbatim, then passes the result to brax's native `restore_params` argument.
Training warm-starts immediately.  The hypothesis to test: a GNN trained on
`n=2` reaches the `n=4` asymptote in fewer steps than training from scratch,
because the per-node communication rule (gait coordination between adjacent
segments) is morphology-invariant.

### GNN-specific CLI flags

| flag | default | meaning |
|------|---------|---------|
| `--gnn-hidden-dim` | `64` | hidden dimension for all Dense layers |
| `--gnn-mp-steps` | `3` | number of message-passing rounds K |
| `--transfer-from` | `None` | path to `params.pkl` from a smaller-morphology GNN run |

All other flags (`--num-envs`, `--naccdmax-per-env`, etc.) are the same as the
MLP baseline.

### Outputs

Same structure as the MLP baseline (§4), but under
`experiment/__data__/undirected_locomotion_ppo_gnn_jax/`.  An extra
`gnn_config.json` records `hidden_dim`, `n_mp_steps`, and the transfer source
path so runs are fully reproducible.

---

## 6. Rerun from a saved run

Replay a trained policy with a different seed or episode length, or
regenerate video artefacts if the training run skipped the render step.

```bash
python experiment/rl/rerun_undirected_locomotion_ppo_jax.py \
    --run-dir experiment/__data__/undirected_locomotion_ppo_jax/insect_undirected-20261008-165537 \
    --seed 42 \
    --episode-length 1000
```

Morphology and `n_pairs` are read from the run's `env_config.json`, so you
can't accidentally run a policy on a mismatched morphology. Output lands in
`<run-dir>/rerun-<timestamp>/` by default; override with `--outdir`.
