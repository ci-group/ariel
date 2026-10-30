# Generalist Hex Drone Controller — Implementation Plan (v2)

Supersedes the v1 plan (BC distillation of per-morph hover gains).

## Goal

A single PPO actor-critic that performs the five tasks
{hover, figure8, slalom, shuttle-run, circle} on **any feasible hexacopter
morphology**, leveraging our 35c CMA-ES hover controller as an
analytical, morphology-conditional inductive prior.

End-state success criterion: given a previously unseen hex morphology,
producing a competent multi-task controller should take
**~60s of CMA-ES (for the prior) + ≤a few minutes of optional PPO
residual fine-tune**, compared to the many-hours full PPO retrain it
would take from scratch.

## Methodology summary — residual policy on top of CMA prior

For each environment step:
```
action_t  =  π_hover(state_t ; θ_cmaes(morph))      ← analytical, no NN
          +  α · π_residual(state_t, task_oh, morph_feat ; φ)
```
- `π_hover` is the closed-form PD + yaw-damping controller from
  `35c_hover_cmaes_minimal.py`. Per-morph parameters `θ_cmaes` (the 11d
  vector `[trim×N, k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate]`) are
  obtained once per morph by running 35c offline and stored in the
  library.
- `α = 0.4` (constant) — same magnitude as 35c's action-scale clamp.
  The residual can override the prior when needed but the prior
  dominates near hover.
- `π_residual` is a PPO policy mirroring 27's MTRL actor-critic. Its
  inputs are state + task one-hot + morphology features. **The 11d
  cmaes vector is not in the obs** — it acts only through the prior,
  keeping the residual's input space clean and morphology-shape-only.
- The prior + residual addition happens **inside the env** (`Variant
  A`), so PPO is unchanged from 27 except for obs-space dimensions.

### Why this architecture

The 35c controller already solves "how to fly this body." A from-scratch
PPO has to rediscover stable hover before it can learn any task — the
notorious "policy spins/falls for 10M steps" failure mode. With the
prior baked in:
- Hover task is essentially solved on day zero by the prior alone.
- Trajectory tasks become "track this setpoint on top of a stable
  platform" rather than "learn to fly AND track."
- Credit assignment is much cleaner: the residual's contribution to
  reward is small-magnitude and well-scoped.
- Morphology dependence is *inside* the prior, so the residual only
  has to learn morphology-specific *task-tracking corrections*.

### Why hide the cmaes vector from the actor

Two reasons:
1. **Cleaner observation space.** The actor sees only what's structurally
   relevant to *its* job (morph shape, current state, current task);
   it doesn't need to know how the prior is parameterised.
2. **Prevents the residual from collapsing back to "just memorise the
   prior."** If the actor could see θ_cmaes, it might learn to undo
   the prior selectively rather than producing meaningful corrections.

## File layout

```
examples/spear/library/
├── IMPLEMENTATION_PLAN.md          (this file)
├── morphology_features.py          shared 22d featurizer (permutation-invariant)
├── hex_sampler.py                  constrained-hex morphology generator + feasibility
├── prior_controller.py             portable PyTorch impl of 35c's analytical π_hover
├── 36_build_hover_library.py       Stage 1: sample morphs + run 35c per morph
├── 37_train_residual_mtrl.py       Stage 3: PPO on residual, extends 27 v4
├── 38_finetune_morph.py            Stage 4: per-morph CMA prior + short PPO residual
│                                            fine-tune for a fresh held-out morph
└── envs/
    └── residual_drone_env.py       Stage 2: TorchDroneGateEnv wrapper that applies
                                              prior + residual scaling internally
```

## Stage 1 — Library build (100 hex morphs)

### Deliverable
`__data__/hex_library/v1/library.parquet` (or jsonl), one row per morph:

| column         | shape         | source |
|----------------|---------------|--------|
| morph_id       | str           | sampler |
| genome         | float32 (6,6) | spherical_angular row format |
| propellers     | json          | decoded propeller list |
| morph_features | float32 (22,) | `morphology_features.morph_features()` |
| cmaes_params   | float32 (11,) | 35c best params (`[trim×6, k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate]`) |
| hover_score    | float         | 35c final reward |
| feasibility    | bool          | always True (rejected morphs not saved) |

### Hex sampler (`hex_sampler.py`)

Hex space = `spherical_angular` genome with **exactly 6 active rows**,
ranges below. Each draw runs a feasibility filter; reject and resample
until target count.

Diversity ranges:

| Axis                | Range / distribution                       |
|---------------------|--------------------------------------------|
| arm magnitude (per motor) | U(0.08, 0.20) m, then optional ±30% per-motor jitter |
| arm azimuth          | sorted around 2π, **min gap ≥ 30°** enforced |
| arm pitch            | mostly 0, ±10° with 20% probability        |
| motor pitch          | locked at 0 (vertical thrust)              |
| motor azimuth        | = arm azimuth (collinear)                  |
| spin pattern         | exactly **3 ccw + 3 cw**, assignment patterns vary |
| core mass            | U(0.20, 0.60) kg                           |
| prop size            | choice({4, 5, 6, 7} inch)                  |

Feasibility filter (reject if any fails):
1. `|sum(spin_i)| ≤ 0` (perfect balance for hex — already enforced by sampler)
2. Adjacent arm-azimuth gap ≥ 15° (no prop collision)
3. `TWR = total_max_thrust / (mass·g) ≥ 1.5`
4. `derive_reference_params(...)` completes without raising (catches degenerate inertia)
5. `_compute_u_hover` returns a value strictly in (-1, 1) (drone can hover)

### Procedure
1. Sample 100 feasible morphs with stratification across:
   `(mean arm length × prop size × max-arm-asymmetry)` = 3×3×3 = 27 cells,
   ≥3 morphs per cell when achievable.
2. For each morph, run 35c with budget=400, λ=128. Save `cmaes_params`
   and `hover_score`.
3. Persist library.

### Gates / kill criteria
- **Coverage**: scatter plot of `(mean_arm_len, twr, spin_pattern_id)`
  should show even fill across stratification cells.
- **Per-morph hover success**: ≥90% of sampled morphs reach
  `hover_score ≥ 400/600`. Below that, sampler is producing
  borderline-unflyable morphs — tighten the filter (TWR threshold,
  arm asymmetry cap).
- **Library size**: 100 morphs final. If gates eliminate too many,
  loosen sampling ranges before adding compute.

### Wall-clock estimate
100 × ~60s = ~1.5h sequential; embarrassingly parallel across morphs,
~10 min on 8 cores.

## Stage 2 — Residual env (`envs/residual_drone_env.py`)

### Concept
Thin wrapper around the existing `TorchDroneGateEnv` (from
`src/ariel/simulation/tasks/torch_drone_gate_env.py`). At env init,
take a morph descriptor including `cmaes_params`. Each step:

```
obs   = (state, task_one_hot, morph_features)
action_residual = policy(obs)                         # PPO actor output
action_prior    = analytical_prior(state, cmaes_params)
action_total    = clamp( prior + α · action_residual )
returns wrapped_env.step(action_total)
```

`analytical_prior` is a vectorised PyTorch implementation of 35c's
controller — same formulae for `alt_cmd`, `att_cmd`, `rate_cmd`,
`yaw_cmd`, including the corrected `-cos(phi)` pitch mixer and the
`-spin` yaw mixer. Lives in `prior_controller.py` so 35c, 35d, and the
residual env all import the same implementation (single source of truth
for the sign convention).

### Observation / action spaces
- **Obs**: `(state_12 + task_oh_5 + morph_feat_22) = 39d`
- **Action**: residual in `[-1, 1]^6`, scaled by `α = 0.4` before
  addition. PPO sees the residual; the env owns the prior.

### Reward
Reuse 27 v4's per-task shaping. For hover specifically, consider
*tightening* the reward (since the prior already nails it) so the
residual receives gradient pressure to do something meaningful for
hover too rather than collapsing to zero. Concretely: hover reward
peaks at 1.0 only when XY drift < 5 cm (tight) rather than the looser
existing threshold.

### Implementation note
`TorchDroneGateEnv` already accepts per-env propellers and is batched
across VecEnv workers, so per-worker morph selection is natural:
each worker loads one morph from the library at episode start, the
wrapper applies the prior. No changes to the dynamics layer.

## Stage 3 — MTRL-PPO training (`37_train_residual_mtrl.py`)

### Architecture
Direct extension of `27_train_rl_hex_mtrl_v4.py`. Differences:

1. **Five tasks** (one-hot dim 5): hover, figure8, slalom, shuttle-run,
   **circle**. Circle gate config: add to `GATE_CONFIGS` if not already
   present; copy from `22_circle.py` / `27_eval_v4_on_circle.py`.
2. **Morphology rotation per env**: each VecEnv worker draws a morph
   from the library on `reset()` (uniform with replacement). Across
   training, every morph is seen many times by many workers.
3. **Obs adds `morph_features` (22d)**; observed space grows from 27's
   `(state, task_oh)` to `(state, task_oh, morph_features)`.
4. **VecNormalize**: keep per-task reward normalisation from 27 v4;
   obs normalisation still global.
5. **PPO config**: same as 27 v4 (lr schedule, ent_coef anneal, n_steps,
   etc.). Network MLP size unchanged.

### Training budget
Target ~80M env steps as in 27 v4. With the prior absorbing the
"don't fall" subproblem, may converge sooner; monitor per-task mean
reward and cut early if all five tasks plateau.

### Gates / kill criteria
- **Hover regression**: per-task mean hover reward at start of training
  should already be high (prior alone hovers). If hover reward at
  step 0 is low, the prior is being misapplied → debug
  `residual_drone_env` before continuing.
- **Per-task progress**: by 10M steps, every task should be improving
  vs the prior-only baseline. Stagnant task → adjust per-task reward
  weighting or VecNormalize per-task running std.
- **Held-out morph generalisation**: at eval time, take 10 morphs not
  in the training rotation and roll out the policy on each task.
  Acceptable if mean reward is within 25% of training-distribution
  morphs **without any fine-tune**.

## Stage 4 — Per-morph fine-tune for OOD (`38_finetune_morph.py`)

For a fresh morphology the policy has never seen:

1. **Prior tune** (always): run 35c on the new morph → 60s, get
   `cmaes_params`. This alone makes hover task work; trajectory
   tasks inherit a stable platform but may track poorly.
2. **Residual fine-tune** (small budget): with the new morph plugged
   into the residual env, run PPO from the pretrained checkpoint for
   a brief budget (default: 200k env steps, ~2–5 min).
3. Evaluate on all five tasks; compare to no-fine-tune baseline.

### Gates / kill criteria
- **In-distribution-shape OOD morph** (held out but within sampler
  ranges): 200k PPO steps should recover ≥90% of training-mean
  performance per task.
- **Out-of-distribution shape OOD morph** (e.g. asymmetric, off the
  stratification grid): even after 200k steps, gracefully degrade
  rather than crash. If crashes, drop α (rely on prior more) or
  expand library coverage.

## Cross-cutting design decisions

### Single source of truth for prior controller
`prior_controller.py` exports `apply_prior(state, cmaes_params, mix,
yaw_mix, u_hover) → action`. Imported by `35c` (training), `35d`
(replay), and `residual_drone_env` (Stage 2/3). The sign-convention
audit we did once should never need to be repeated.

### Mixer sign-convention unit test
Add `tests/test_mixer_signs.py`:
- Construct a regular hex blueprint.
- Apply a known disturbance (e.g. `pitch = +5°`).
- Assert the prior's action vector reduces `My` (rather than
  increases it) under the dynamics.
Run in CI; this is the test that would have caught the original
sign bug.

### Morphology featurizer (`morphology_features.py`)
22d hand-crafted, permutation-invariant in motor order, normalised to
~O(1):
- motor count, mass, normalised inertia diag (3)
- arm-length stats: mean, std, max (3)
- spin balance: `sum(spin)/N` (1)
- azimuth-gap stats: mean, max, std (3)
- xy-position stats: x mean/std, y mean/std (4)
- analytical `u_hover` (1)
- TWR at full throttle (1)
- mean prop size (1)
- zero padding (3)

For hex-only training, motor count is always 6, but keeping it in the
descriptor means the same featurizer generalises to other rotor counts
later without retraining.

### Reproducibility
All scripts take `--seed` and persist it. Library entries record their
own 35c seed. PPO checkpoint records library version + seed + 35c
seed-rotation pattern.

## Order of operations (with gates)

1. Write `morphology_features.py` + unit tests. ~2h.
   *Gate*: features identical under motor-permutation; values in O(1)
   range for sampled morphs.
2. Write `hex_sampler.py` + feasibility filter + coverage viz. ~2h.
   *Gate*: 100 feasible morphs in <5 min sampler time; coverage plot
   even.
3. Write `prior_controller.py` extracted from 35c. ~1h.
   *Gate*: rerun 35c with the imported prior — identical fitness curve.
4. Refactor `35d_replay_cmaes_minimal.py` to use the shared prior
   (regression check). ~30 min.
5. Run Stage 1 in full (~10–90 min depending on parallelism).
   *Gate*: ≥90% of 100 morphs reach `hover_score ≥ 400`.
6. Write `envs/residual_drone_env.py` wrapper + sign-convention test.
   ~3h.
   *Gate*: with `α = 0`, env behaves identically to the prior alone
   (numerical match). With `α = 0.4` and a zero residual policy, prior
   alone still hovers.
7. Write `37_train_residual_mtrl.py` extending 27 v4. ~4h.
   *Gate*: step-0 per-task reward shows hover already strong (prior
   working). First 1M steps don't crash or diverge.
8. Train (target 80M steps, ~hours/days depending on hardware).
   *Gate*: by 20M steps, all five tasks > 27 v4's same-task numbers
   on the training-distribution morphs; held-out morphs within 25%.
9. Write `38_finetune_morph.py`. ~2h.
   *Gate*: held-out morph + 200k PPO steps recovers ≥90% of training
   performance.
10. Final demo: pick 5 OOD morphs, fine-tune each, screen-record all
    five tasks on each. That's the generalist claim.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Prior dominates so much the residual learns nothing useful | Monitor residual action magnitude during training. If consistently ~0, raise α to 0.6 or tighten per-task reward to demand more from the residual. |
| Library too small (100) → poor OOD generalisation | If Stage 4 gates fail systematically, double library to 200 and retrain; cheap compared to architectural changes. |
| Prior diverges on extreme morphs (high asymmetry) | Tighten Stage 1 feasibility filter; expose `cmaes_params` validity check in `38_finetune_morph.py` (re-run 35c with larger budget). |
| Per-task reward normalisation interacts badly with prior | Each task's prior-alone baseline should set the floor; track baseline vs PPO reward delta rather than absolute reward to monitor learning. |
| Circle task gate config not in `GATE_CONFIGS` | Add it in Stage 3 step 1; copy from `22_circle.py` or `27_eval_v4_on_circle.py`. |
| α = 0.4 wrong for some tasks | Variant: per-task α as a (frozen) hyperparameter; tune via a 5-point grid after Stage 3 converges. |

## Out of scope

- JEPA-style representation pretraining (revisit only if Stage 3 gates
  consistently fail across multiple task types and the failure is
  attributable to representation quality, not dynamics).
- Variant B (prior as differentiable layer inside the actor) — only if
  Variant A's value-function training is unstable.
- Non-hex morphologies (4-rotor, octo, asymmetric). Defer; the
  featurizer is already general enough to support them later.
- Sim-to-real, camera/lidar observations, adversarial morph generation,
  co-evolution of morph + controller.

## Locked decisions (per user 2026-06-24)

- Variant **A** (prior-as-env)
- Library size: **100** hex morphs
- α: **constant 0.4**
- `cmaes_params`: **hidden from actor**, applied only via prior
- Tasks: **5** (hover, figure8, slalom, shuttle-run, **circle**)
- Per-morph fine-tune at eval: **small budget** (default 200k PPO steps;
  CMA prior always re-run)
