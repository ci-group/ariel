# PPO on ARIEL morphologies (mujoco-playground + brax)

GPU-accelerated PPO training for ARIEL modular robots, built on top of the
vendored [mujoco-playground](../mujoco_playground/) stack. The training script
wraps an ariel-built MJCF as a `mujoco_playground._src.mjx_env.MjxEnv` and
runs brax PPO on it with MuJoCo-Warp.

- `undirected_locomotion_ppo_jax.py` — train a policy that rewards displacement
  in any direction (undirected locomotion).
- `rerun_undirected_locomotion_ppo_jax.py` — reload a trained run and render a
  fresh rollout video + trajectory plot without retraining.

Supported morphologies: `insect` (fixed) and `centipede` (parameterised by
number of leg pairs).

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

## 5. Rerun from a saved run

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
