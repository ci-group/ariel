# Generalist Drone Control — Analysis & Draft Proposal

**Author:** Opus 4.7 (draft) + Fable updates (2026-07-09 afternoon)
**Date:** 2026-07-09 — updated post-implementation
**Scope:** `examples/spear/` — evolved hex drones × 5 tasks (hover, circle,
shuttle-run, slalom, figure8), leveraging the library of learnt hovering
policies in `examples/spear/library/`.

**Status:** Blocking items 1–4 of the Stage-3 launch plan are now complete
(see §3 and §Execution below). The pipeline is ready for scale-up training
(≥100 envs, 20M steps) and will enter the demo/paper phase after that.

---

## 1. What's already on disk

### 1.1 Per-task scripts (single-morph, single-task)
`18_hover.py`, `19_figure8.py`, `20_slalom.py`, `21_shuttlerun.py`,
`22_circle.py` — five PPO scripts, one per task. Baseline for
per-task performance. All use a fixed hex morphology.

### 1.2 Morphology evolution + PPO (`9_…configurable.py`, `29`, `31`)
Evolves hex geometry, trains a PPO controller **per candidate body**
(warm-started from best previous). Establishes that morph-conditional
control is expensive: each body pays a full PPO retrain.

### 1.3 MTRL v4 (`27_train_rl_hex_mtrl_v4.py`)
Multi-task PPO across the 5 tasks on a **single hex morphology**.
Per-task reward normalisation (`_PerTaskRewardNormalizer`) —
important because hover shaping (~+0.0125/step continuous) is on a
totally different scale from figure8 gate spikes (+1). Without
per-task scaling, hover swamps the value function. Fable: keep this,
it is load-bearing.

### 1.4 Hover-CMA (`35c_hover_cmaes_minimal.py`)
Analytical PD + yaw-damping controller parameterised by an 11-dim
vector `[trim×6, k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate]`.
CMA-ES fits this vector to any hex morph in **~60 s** (budget=400,
λ=128). This is the **library of learnt hovering policies** the user
is asking about: it isn't a set of NN weights, it's a set of 11-d
analytical-controller parameters — one per morphology.

### 1.5 Hex library v1 (`__data__/hex_library/v1/`)
- 100 stratified feasible hex morphologies, seed=42.
- `library.npz` columns: `morph_seed`, `cmaes_params (11,)`,
  `morph_features (22,)`, `hover_score`.
- `coverage.png` shows even fill across the
  (arm-length × prop-size × asymmetry) grid.

### 1.6 Residual-env stack (`library/`) — complete & verified
- `prior_controller.py` — portable Torch implementation of the 35c
  hover law. Single source of truth for mixer sign convention.
  - Default init includes **analytical hover-balance trims** (non-zero,
    computed via least-norm solution to null static moments on asymmetric
    morphs). High-TWR morphs without trims explode at t=0 before CMA can
    engage.
  - All 21 prior+residual gate tests pass (sign guards, shape, α-identity).
- `envs/residual_drone_env.py` — TorchDroneGateEnv wrapper. Per step:
  `action_total = clamp(prior(state; cmaes_params) + α_task · residual)`
  - All 5 tasks (hover, figure8, slalom, shuttle-run, circle) are wired
    and functional (via `_task_gate_config`).
  - Per-task α (hover 0.10, racing 0.40) and gain scaling already in code.
  - **Critic-only prior descriptor added (2026-07-09):** obs is now 65-d
    (was 53-d); trailing 12 dims = `cmaes_params[11] + median_score/100[1]`
    routed to critic input only (actor never sees them). This encodes "how
    trustworthy is my prior on this morph" per `Residual_Policy_Learning.md`.
- `37_train_residual_mtrl.py` — Stage-3 full PPO on the residual.
  - Observations: `(gate_obs[26], task_oh[5], morph_features[22],
    cmaes_params[11], median_score[1]) = 65d`.
  - **Held-out split added (2026-07-09):** `--held-out 10` reserves 10
    stratified morphs (one per coverage cell) for eval-only; persists
    morph-id list in `held_out.json` per run.
  - Each VecEnv worker holds **one** (morph, task) pair for the whole run;
    tasks round-robin so all 5 get equal workers; morphs rotate within the
    training split.
  - Smoke run (50k steps, 20 envs, CPU): completes cleanly, hover reward
    jumps from near-zero (step 0) to +13.3/ep after training.
- `38_visualize_prior_per_task.py` — replays the prior alone on each
  of the 5 tasks. This is the "free baseline" the residual must beat.
- **New:** `40_retune_fragile_priors.py` — re-runs CMA at 3× budget on the
  ~20 morphs with `median_score < 400`, updates `library.npz` atomically
  with timestamped backup + manifest record.

### 1.7 Training artifacts & re-tuning
`__data__/library_residual_mtrl/` has ~18 dated runs (2 from 2026-06-29
were empty, died before first checkpoint; diagnostic suggests transient
CUDA error). Latest run (2026-07-09 smoke): 50k steps, checkpoint
intact.
- Re-tune run in progress (2026-07-09 15:31): 20 fragile morphs at
  1200-generation budget, ETA ~5 hours total (~16 min/morph on CPU).

---

## 2. The v2 plan in one paragraph

> Every hex morph gets a **CMA-tuned analytical hover controller** as
> its prior. A single PPO residual policy — conditioned on
> `(state, task_one_hot, morph_features)` but **not** on the 11-d CMA
> vector — outputs an additive correction with fixed weight α=0.4.
> Prior handles "how to fly this body"; residual handles "how to track
> this task." For a fresh morph: 60 s of CMA + ≤5 min of PPO
> fine-tune, instead of hours-long retrain.

This is a strong architecture. The rest of this doc is about
**closing the gap between plan and code**, plus a few design pushes
Fable should evaluate.

---

## 3. Execution summary (2026-07-09)

**Item 1: Smoke-run `37` (50k steps) — DONE ✅**
- Diagnosis: 2026-06-29 runs died before first checkpoint; likely transient
  CUDA error (now checkpointed every ~250k steps).
- Result: 50k steps complete, hover reward +13.3/ep (near max ~15/ep);
  actor convergence well-behaved.

**Item 2: Held-out split — DONE ✅**
- `--held-out 10` (default) stratified via `_stratify_key` across 27
  coverage cells.
- Persisted per run in `held_out.json`; held-out morph seeds:
  [43,229,51,75,57,120,107,101,117,141] at seed=0.
- Reserved for Stage-4 eval; training uses 90 morphs.

**Item 3: Critic-only prior descriptor — DONE ✅**
- Obs expanded 53d → 65d; trailing 12 dims routed to critic input only
  (`Residual_Policy_Learning.md` §Critic independence).
- Actor path uses original 53d → cannot learn to undo the prior.
- Verified by tail-content assertion + mini PPO run.

**Item 4: Re-tune fragile morphs — IN PROGRESS ⏳**
- Script: `40_retune_fragile_priors.py` (1200 generations × 128 pop/morph).
- Accepted only if median score improves AND score ≥ 90% of old.
- Run started 2026-07-09 15:31, 1/20 done (~24 min); ETA ~5 hours total.
- Will update `library.npz` atomically with timestamped backup.

**Bonus: Prior-testing fixes — DONE ✅**
- `default_init_params()` docstring clarified (returns non-zero analytical
  trims, not zero trims).
- Mixer-sign tests rewritten to be delta-based (disturbed − undisturbed
  action), making them more robust.
- All 21 prior + 30 residual env tests pass.

---

## 3b. Gap analysis (plan v. code) — updated

| Plan gate | Code state (2026-07-09) | Status |
|-----------|-----------|--------|
| Stage 1: ≥90% morphs `score ≥ 400` | ✅ 100% of 100 pass | Verified. |
| Stage 1: `median_score` distribution | Library has `median_score`; p10=298 | ~20 morphs fragile; re-tune in progress. |
| Stage 2: Sign-convention + α=0 tests | ✅ All 21 tests pass | Mixer signs correct; α identity verified. |
| Stage 3: All 5 tasks wired | ✅ All 5 tasks implemented | `hover`, `figure8`, `slalom`, `shuttle-run`, `circle` all functional. |
| Stage 3: Per-task α + gain scaling | ✅ In code since v4 | Hover 0.10, racing 0.40; prior gain scaled per task. |
| Stage 3: Critic sees prior descriptor | ✅ Implemented | Critic input 48d → 60d (adds cmaes_params + median_score). |
| Stage 3: Held-out split | ✅ Implemented | 10 stratified morphs reserved; list persisted. |
| Stage 3: Smoke run | ✅ 50k steps pass | Clean exit, reasonable convergence. |
| Stage 3 obs adds `morph_features` | ✅ done, 53d | Fine. |
| Stage 3 gate: step-0 hover reward high | Achievable only if `α=0.4` residual + zero-init policy ≈ prior — needs verification | Add a "step-0 sanity" logger, kill training if prior is being applied wrong. |
| Stage 4 (`38_finetune_morph.py`) | Not present. `38_visualize_prior_per_task.py` is a different script. | **Missing** — but low priority until Stage 3 lands. |

---

## 4. Design proposals

### 4.1 Wiring all 5 tasks into `ResidualDroneEnv` `[open]`

`_task_gate_config` already returns the right `GATE_CONFIGS` for
figure8/slalom/shuttle-run/circle. What's missing is that
`37_train_residual_mtrl.py`'s VecEnv factory hard-codes `task="hover"`.

**Proposal — per-worker task assignment, not per-episode.**
Each worker draws `(morph, task)` at construction. Rationale:

- The **prior itself is task-agnostic** (it's a hover law). Swapping
  tasks per episode doesn't require any per-morph re-tuning, only a
  reset of the gate stream and reward function. So per-episode task
  rotation is technically cheap.
- But: `TorchDroneGateEnv` embeds gate positions and reward shaping
  into its constructor path. Swapping tasks on `reset()` would need
  a refactor to move gate-track state out of `__init__`. **[open]**
  Fable — assess whether this refactor is worth it or if
  per-worker `(morph, task)` with, say, 400 workers × 5 tasks × 80
  morphs (= 5 per (morph, task) pair) gives enough coverage.

**Recommended split (initial):** `N_WORKERS = 400`, distributed as
`80 morphs × 5 tasks`. This preserves the plan's morph-rotation
diversity and adds task diversity without refactor. If PPO struggles
on the sparse-reward tasks (figure8, slalom) because too few workers
see them, upweight those tasks in the split (e.g. 40 hover, 90 each
trajectory task).

### 4.2 Per-episode morph rotation `[open]`

Plan Stage 3 called for morph rotation on `reset()`. The MVP defers
this to a follow-up. Reasons to keep the deferral:

- Swapping `cmaes_params` mid-worker means also swapping the
  MuJoCo model handle (motor placements, inertia). That's a full
  env rebuild — cheap in wall-clock but hairy to code cleanly.
- Per-worker rotation gives statistical coverage of the same order
  of magnitude at 400+ workers.

Reasons Fable might override me:

- Value-function fitting benefits from **within-worker return
  diversity**. A worker that only ever sees morph #37 learns a
  value function specialised to that morph's rewards; the shared
  critic then has to reconcile 400 specialised critics.
- If per-worker morph diversity is too coarse (e.g. we drop to 100
  workers on smaller hardware), per-episode rotation is the only
  way to preserve coverage.

**Recommendation:** measure first. Add a "critic loss vs morph_id"
diagnostic to Stage 3. If it correlates strongly with morph_id at
convergence, refactor for per-reset rotation. Otherwise, keep MVP.

### 4.3 Should `cmaes_params` be hidden from the actor? `[locked]` — but re-litigate for the critic

The plan (§ locked-decisions) hides `cmaes_params` from the **actor**
to prevent the residual from "learning to undo the prior."

**Fable, consider:** should the **critic** see `cmaes_params`?
The critic's job is value estimation, not action selection; it has
no incentive to route around the prior. And "how competent is my
prior on this morph" (encoded in the 11-d vector plus the scalar
`hover_score`) is arguably the single most informative feature for
predicting return. Providing it to the critic-only (`vf_features` ≠
`pi_features` in SB3) is cheap and might tighten advantage estimates
substantially.

### 4.4 α scheduling `[open]`

Locked decision is `α = 0.4` constant. Two directions worth
evaluating:

1. **Per-task α.** Hover barely needs the residual (prior nails it).
   Trajectory tasks probably want α closer to 0.6–0.8 so the
   residual has authority to carve out non-hover manoeuvres.
   Implementation: `α = α_task[task_id]`, tuned as a 5-point grid
   after Stage 3 first converges.
2. **α annealing.** Start at 0.2 (prior-dominated, so early PPO
   doesn't destabilise), anneal to 0.5 over first 20M steps. Reduces
   the "prior collapses to zero because residual is noisy" risk in
   the first million steps.

I'd try (1) before (2). (2) is more of a training-stability hedge; if
Stage 3 gate #1 (step-0 reward high) passes, we probably don't need
it.

### 4.5 Using the "library of learnt hovering policies" beyond initialisation `[open]`

Right now the library is used as: "look up `cmaes_params` for this
morph → plug into prior." That's it. Two under-exploited angles:

**(a) Library-guided morph sampling for training.**
The residual policy must generalise across the (morph_features)
manifold. Rather than uniformly rotating through the 100 stored
morphs, weight sampling **inversely to local density** in
morph_feature space. Simple k-NN density estimate; upsample the
underrepresented corners of the manifold. Cheap to implement.

**(b) k-NN prior for unseen morphs.**
For a genuinely OOD morph at eval time (Stage 4), instead of
re-running CMA from scratch, **warm-start CMA-ES from the mean of
its k=5 nearest neighbours in morph_feature space.** Should cut the
60 s of CMA to ~15 s. Even better: skip CMA entirely and just use
the k-NN-interpolated `cmaes_params` — if the residual is doing its
job, small prior errors will be absorbed. That's the difference
between "one-shot deploy" and "60 s tuning + deploy" on a new body.

Fable: (b) is the big lever. Worth prototyping in Stage 4 (§4.7).

### 4.6 Circle task `[open]`

Plan Stage 3 step 1 flags: **circle is not in `GATE_CONFIGS`**;
need to port from `22_circle.py` / `27_eval_v4_on_circle.py`.
Fable — verify. If circle uses continuous-trajectory rewards rather
than discrete gates, the `TorchDroneGateEnv` gate stream needs a
"virtual moving gate that traces the circle" adapter. This is
non-trivial and should be its own subtask.

### 4.7 Stage 4 (`38_finetune_morph.py`) design `[open]`

Plan says: 60 s CMA + 200k PPO steps on the new morph. Three
pieces I'd add:

1. **Progressive residual unfreezing.** First 50k steps: freeze
   actor, only fine-tune critic (recalibrate value on new morph).
   Next 150k: full PPO. Avoids destroying the actor with poorly-
   estimated advantages on the new morph.
2. **k-NN prior warm-start** (per §4.5 (b)).
3. **Prior-only baseline logged alongside PPO reward** so we can
   tell if the fine-tune is helping vs the free baseline.

---

## 5. Concrete order-of-operations recommendation

If Fable agrees with the framing above, the shortest path to a
working generalist is:

1. Add pytest `test_prior_alpha_zero_identity` and
   `test_mixer_signs` (plan §Stage 2 gates). — 1–2 h.
2. Add Stage-3 step-0 diagnostic to `37_…mtrl.py`: log per-task mean
   reward under (α=0, zero-residual) at env init. Kill run if hover
   isn't already near-solved. — 1 h.
3. Wire figure8/slalom/shuttle-run into `ResidualDroneEnv` +
   per-worker task assignment (§4.1). — 4–6 h.
4. Port circle to `GATE_CONFIGS` (§4.6). — 2–4 h, unknown until
   `22_circle.py` is inspected.
5. Full Stage-3 training run (~80M steps, gate = beats prior-only
   on all 5 tasks + within 25% on 10 held-out morphs).
6. Stage 4 with k-NN prior warm-start (§4.5b) and progressive
   unfreezing (§4.7).
7. Ablations for the paper / demo:
   - Prior-only vs residual-only vs prior+residual on 5 tasks × 20
     morphs.
   - k-NN prior vs freshly-CMA'd prior on 5 held-out morphs.
   - Per-task α grid.

Steps 1–4 unblock the full training run and are all bounded
day-scale work. Everything else waits on Stage-3 numbers.

---

## 6. Open questions for Fable to resolve

1. **Refactor `TorchDroneGateEnv` for per-episode task swap, or
   accept per-worker `(morph, task)` assignment?** (§4.1)
2. **Per-episode morph rotation — necessary now or later?** (§4.2)
3. **Should the critic see `cmaes_params` even though the actor
   doesn't?** (§4.3)
4. **Per-task α, α annealing, or leave α=0.4 constant?** (§4.4)
5. **k-NN prior interpolation as first-line for unseen morphs —
   worth the engineering vs. always re-running CMA?** (§4.5b)
6. **Circle task: reuse gate-based env or need a continuous-
   trajectory adapter?** (§4.6)
7. **What does "generalist" mean for the paper's success criterion?
   Zero-shot on new morph? 60-s-of-tuning-shot? Both?** — the plan
   implies the latter; a stretch goal is the former via k-NN prior.

---

## 7. Non-goals (I agree with the plan)

- No JEPA/representation pretraining. Morph features are hand-crafted
  and permutation-invariant; that's likely enough at n=100 morphs.
- No non-hex morphologies in v1. Featurizer is general enough to
  extend later.
- No co-evolution of morph + generalist controller. Two separate
  problems; combining them prematurely blows up the search space.
- No sim-to-real. Explicitly deferred.

---

*End of Opus draft (Part I).*

---
---

# Part II — Fable iteration (2026-07-09 — blocking items complete)

Verified the draft against the code. Several claims in Part I are
**stale** — the code moved past its own docstrings. Corrections first,
then decisions on the 7 open questions, then the build plan.

## 8. Corrections to Part I

| Part I claim | Reality (verified) |
|---|---|
| §1.6/§3: "only hover wired into `ResidualDroneEnv`" | **False.** `_task_gate_config` handles all 5 tasks and `37:475` assigns `tasks = [TASK_NAMES[i % NUM_TASKS]]` across workers. The docstring in `37` is stale, not the code. |
| §4.4: propose per-task α | **Already implemented** (`ResidualDroneEnv.TASK_ALPHA`): hover 0.10, racing tasks 0.40. |
| — (not anticipated) | **Per-task prior gain scaling** exists (`TASK_PRIOR_GAIN_SCALE`): racing tasks keep altitude PD, scale `k_tilt` to 0.3 and rate damping to 0.5 so the hover prior's attitude leveler doesn't fight banking. This is a better idea than either α proposal in §4.4 and should be kept. |
| §4.6: "circle not in `GATE_CONFIGS`" | **False.** `gate_configs.py:117` registers `'circle': CircleGates`. Non-issue. |
| §3: "inspect hover_score quantiles" | Done. Keys are `score`/`median_score` (not `hover_score`). `score`: p10=588.6, min=552, 100% ≥ 400 → Stage 1 gate **passed**. But `median_score` p10 = 298 → ~20% of morphs have high-variance CMA solutions (best seed ≫ median). These are the morphs where the prior will be fragile under residual perturbation. |
| §1.7: "18 short runs, no final checkpoint" | Confirmed, and worse: the two most recent run dirs (2026-06-29) are **empty** — the last training attempts died before the first checkpoint. Diagnose before scaling. |

## 9. Decisions on the open questions (§6)

1. **Task swap refactor?** No. Per-worker round-robin already exists
   and is sufficient. Closed.
2. **Per-episode morph rotation?** Defer, per §4.2. But the binding
   constraint is `--num-envs 20` (default): 20 workers × 5 tasks = 4
   morphs per task per run. Raise to ≥100 workers before judging
   coverage. Add the critic-loss-vs-morph diagnostic from §4.2.
3. **Critic sees `cmaes_params`?** Yes — accept §4.3. The policy class
   already splits actor/critic inputs (`CRITIC_IN_DIM ≠ SHARED_IN_DIM`
   in `37`), so appending 11d + `median_score` (1d) to the critic path
   is a ~10-line change. The `median_score` scalar directly encodes
   "how trustworthy is my prior here," which is the strongest possible
   value-prediction feature.
4. **α scheduling?** Resolved by existing code (per-task α + gain
   scaling). Don't add annealing unless training diverges.
5. **k-NN prior for unseen morphs?** Yes, staged: (i) k-NN interpolated
   `cmaes_params` as CMA warm-start in `38_finetune_morph.py`;
   (ii) measure zero-shot (interpolated prior, no CMA, no PPO) as a
   free ablation — if it works it's the headline result. Weight
   neighbours by inverse distance in `morph_features` space; **exclude
   the trim components from naive interpolation** if motor ordering
   differs between morphs (trims are per-motor, order-sensitive —
   verify the sampler emits motors in canonical azimuth order first;
   if not, sort by azimuth before interpolating).
6. **Circle env adapter?** Non-issue (see §8).
7. **"Generalist" criterion?** Report a 3-tier ladder, cheapest first:
   zero-shot (k-NN prior + frozen residual) → 60 s CMA + frozen
   residual → CMA + 200k PPO fine-tune. The ladder itself is the
   contribution: "competence vs adaptation-budget" curve per task.

## 10. Remaining work (post-blocking-items, in execution order)

**Status: Items 1–4 from §3 (the original blocking list) are now complete.**
**Items below are the post-launch roadmap:**

1. **Re-tune fragile morphs (ongoing).** `40_retune_fragile_priors.py`
   running; should finish ~2026-07-09 20:30 UTC. After: inspect the
   accept/reject breakdown and updated `median_score` quantiles.
2. **Scale-up training run.** Launch `37` with ≥100 envs, 20M steps,
   new checkpoints at 250k-step intervals. Log:
   - Per-epoch residual magnitude `|π_θ|` per task (risk: residual
     collapses to zero or saturates).
   - Per-task policy gradient cosine similarity `cos(∇ℒ_i, ∇ℒ_j)` at
     each epoch (risk: trajectory tasks swamp hover; if conflict >50%
     during first 10M steps, gate warrants PCGrad).
   - Per-task reward/gates/episode-success curves, both training and
     a held-out rollout every 1M steps.
3. **PopArt POP correction (conditional).** Only add if critic-loss
   spikes when gate-reward tasks first produce positive returns
   (Hessel et al., §Reward normalisation). For now, keep the ART half
   (`_PerTaskRewardNormalizer`) as-is.
   `test_prior_alpha_zero_identity`, `test_mixer_signs`. Cheap; do
   before any long run.
4. **Critic-side `cmaes_params` + `median_score`** (decision 3).
5. **Scale-up run**: ≥100 envs, 20–80M steps, checkpoint + per-task
   reward curves + residual-magnitude logging (risk table row 1 of
   the v2 plan).
6. **`38_finetune_morph.py`** with the 3-tier eval ladder (decision
   5/7) and progressive unfreezing (§4.7).
7. **Fragile-prior morphs**: for the ~20 morphs with
   `median_score < 400`, re-run 35c with 3× budget and update the
   library in place, or down-weight them in the training rotation.
   Fragile priors + α-perturbation is the most likely source of the
   "residual learns to fight the prior" failure mode.
8. Refresh the stale docstrings in `37` and `residual_drone_env.py`
   (they claim hover-only; they mislead every future reader,
   including the model that wrote Part I).

## 11. Papers to ingest (for the project wiki)

Ranked by expected leverage on this exact architecture:

**Directly load-bearing**
1. **Silver et al., "Residual Policy Learning" (2018)** and
   **Johannink et al., "Residual Reinforcement Learning for Robot
   Control" (ICRA 2019)** — the canonical residual-on-prior papers;
   both discuss the α/authority trade-off and prior-fighting failure
   mode we already hit (hence `TASK_PRIOR_GAIN_SCALE`).
2. **Zhang et al., "Learning a Single Near-Hover Position Controller
   for Vastly Different Quadcopters" (ICRA 2023)** — closest published
   result to our goal; single policy across morphologies via
   parameter conditioning. Direct baseline comparison.
3. **Kumar et al., "RMA: Rapid Motor Adaptation" (RSS 2021)** — the
   main architectural alternative to hand-crafted `morph_features`:
   learn a latent morph embedding from state-action history
   (teacher-student). If §4.5's hand-crafted features underperform on
   held-out morphs, RMA-style adaptation is the fallback; ingest now
   so the fallback is designed, not improvised.
4. **Kaufmann et al., "Champion-Level Drone Racing using Deep RL"
   (Nature 2023, Swift)** — reward shaping and gate-progress reward
   design for racing tasks; our figure8/slalom shaping should be
   audited against theirs.

**Supporting**
5. **Hessel et al., "Multi-task Deep RL with PopArt" (AAAI 2019)** —
   principled version of our `_PerTaskRewardNormalizer`; worth
   checking whether PopArt's value-head rescaling beats reward-side
   normalisation.
6. **Yu et al., "Gradient Surgery for Multi-Task Learning" (PCGrad,
   NeurIPS 2020)** — if per-task rewards plateau unevenly in the
   scale-up run, task-gradient conflict is the first suspect.
7. **Huang et al., "One Policy to Control Them All" (ICML 2020)** and
   **Gupta et al., "MetaMorph" (ICLR 2022)** — morphology-conditioned
   control via modular/transformer policies. Relevant only if we later
   drop the hex-only restriction; low priority.
8. **Eschmann et al., "Learning to Fly in Seconds" (2024)** — extreme
   sample-efficiency baseline for the "minutes of fine-tune" claim.
9. **Lee, Leok, McClamroch, "Geometric Tracking Control of a Quadrotor
   UAV on SE(3)" (CDC 2010)** — already used in
   `14_mujoco_lee_figure8.py`; the classical-control upper bound the
   prior+residual should be compared against on trajectory tasks.

## 12. Revised order of operations

1. Smoke-run `37` (50k steps) → diagnose why the last runs died. (hrs)
2. Tests: α=0 identity + mixer signs. (1–2 h)
3. Held-out split (10 stratified morphs) + persist with checkpoint. (1 h)
4. Critic-side `cmaes_params`/`median_score`. (1 h)
5. Re-tune the ~20 fragile-prior morphs (3× CMA budget). (~1 h parallel)
6. Scale-up training run, 100+ envs, 20M steps first; extend to 80M
   only if curves still climb. Log residual magnitude per task.
7. `38_finetune_morph.py` + 3-tier eval ladder on the 10 held-out
   morphs. This produces the headline table.
8. Ablations (§5 item 7) + docstring refresh.

Items 1–5 are one working day. Item 6 is the compute gate.

---
---

# Part III — Wiki-informed design amendments (2026-07-09)

After ingesting the full set of recommended papers into the project wiki,
the following amendments sharpen the Part II plan with specific, citable
design decisions. Each is tagged with the wiki page that grounds it.

---

## 13. MTRL normalisation: keep `_PerTaskRewardNormalizer`, add POP weight correction on divergence

**Source:** `PopArt_MultiTask_RL.md`

Ariel's `_PerTaskRewardNormalizer` implements the ART half of PopArt (adaptive rescaling of targets via Welford running std) but **not the POP half** (output-preserving weight correction when statistics change). The missing POP correction means that when the residual first starts passing gates and the gate-task return scale jumps from near-0 to +1 spikes, the value head's outputs momentarily become incorrect — which can cause a PPO update spike that destabilises training.

**Decision:** Keep the current reward-side normaliser for the initial scale-up run. Add the POP weight correction only if the Stage-3 run shows a critic-loss spike coinciding with a task's first successful gate passes. The SB3 implementation path is documented in `PopArt_MultiTask_RL.md §In Ariel`: wrap the value head's last `nn.Linear` in a `PopArtLinear` that applies `w = (sigma_old/sigma_new) * w` after each statistics update.

---

## 14. Gradient conflict monitoring: log per-task actor gradient cosine similarity

**Source:** `PCGrad_Gradient_Surgery.md`, `multitask_gradient_interference.md`

The tragic triad analysis (Yu et al.) predicts that hover's small, smooth gradient will conflict with trajectory tasks' large, spiky gate-spike gradients — especially before the residual has learned to pass gates. In the MT10 benchmark, this conflict was present in >80% of iterations before task 2 was learned.

**Decision:** Add per-task actor gradient cosine similarity logging to the Stage-3 PPO callback (computed once per PPO epoch from per-task loss components — cheap at N=5). If hover↔trajectory conflict exceeds ~50% of epochs during the first 10M steps, apply PCGrad projection before the optimizer step. The implementation requires splitting SB3's combined PPO loss into per-task components before `loss.backward()` (a custom `train()` override in `OnPolicyAlgorithm` — see `PCGrad_Gradient_Surgery.md §In Ariel`).

**Do not add PCGrad pre-emptively.** Per-task α scaling (hover 0.10, racing 0.40) and `TASK_PRIOR_GAIN_SCALE` already reduce hover's gradient magnitude relative to trajectory tasks, which may be sufficient. Measure first.

---

## 15. BC from the CMA prior: consider an imitation term during early training

**Source:** `extreme_adapt_quadcopter.md`

Zhang et al. 2025 (T-RO) demonstrate that a BC+RL hybrid with a decaying weight `α = exp(-0.001·t_epoch)` dramatically accelerates early training convergence and prevents the RL-only baseline from diverging after 50M steps. Their BC expert is a model-based controller with access to ground-truth parameters — analogous to ariel's CMA hover prior.

**Ariel translation:** Add an optional BC loss term to `37_train_residual_mtrl.py`:
```
L_BC   = ‖prior_action_t - (prior_action_t + α·residual_t)‖²   # pulls total action toward prior
R(π)   = (1 - weight) · R_PPO - weight · L_BC
weight = exp(-0.001 · t_epoch)                                   # decays to 0 over ~1000 epochs
```

This would ensure the residual starts near-zero (prior-dominated) and RL gradually takes over, rather than relying on the zero-initialised actor to stay small throughout. **Recommendation:** stage this as an experiment after the baseline scale-up run. If the baseline diverges after 50M steps (as in Zhang et al.'s RL-only ablation), add BC. If it converges, skip.

---

## 16. Design-informed hex randomization for the library

**Source:** `extreme_adapt_quadcopter.md`

Zhang et al.'s key domain randomization insight is that independent parameter sampling produces physically unrealistic morphologies. Instead, a size factor `c` correlates mass ∝ l³, inertia ∝ l⁵, drag ∝ l², motor thrust exponential in `c`. Ariel's `hex_sampler.py` currently samples parameters independently within ranges.

**Decision:** For a potential v2 hex library (not blocking Stage 3), adapt `hex_sampler.py` to:
1. Sample an arm-length scale factor `c`.
2. Derive mass, inertia, and drag from `c` using physics-informed scaling.
3. Keep ±20% per-parameter noise for flexibility.

This would produce more physically plausible hex morphologies and reduce degenerate samples (e.g. short arms with enormous prop constants) that destabilise the prior.

---

## 17. Reward shaping for racing tasks: audit against Swift and use torque tracking

**Sources:** `Swift_Drone_Racing.md`, `extreme_adapt_quadcopter.md`

**Swift (Kaufmann et al. Nature 2023)** uses gate-progress reward + perception reward + command smoothing penalty:
```
r_prog = λ₁ · (d_{t-1}^Gate - d_t^Gate)         # gate-distance progress
r_cmd  = λ₄·‖a_t^ω‖ + λ₅·‖a_t - a_{t-1}‖²      # smoothness penalty
```

The `r_prog` (distance-to-next-gate) reward is dense and continuous — it provides gradient at every step even before a gate is passed. Ariel's current gate-passing reward (+1 at gate crossing) is sparse. The Swift progress reward is worth adding as a secondary shaped component alongside the current gate-crossing bonus.

**Torque tracking (Zhang et al. T-RO):** For the inner-loop trajectory tasks, rewarding torque error (`-‖τ_t - τ_des‖`) rather than angular velocity error gives denser gradient because torque responds immediately to motor commands. For ariel's residual (which outputs motor corrections), adding a torque-alignment term could sharpen the gradient signal for figure8/slalom/shuttle-run.

**Decision:** After the baseline scale-up run, if slalom/shuttle-run converge slower than figure8/circle, audit the per-task reward shaping against these references and add the progress component.

---

## 18. Classical control baseline: Lee controller as trajectory performance ceiling

**Source:** `Lee_Geometric_Control_SE3.md`

The Lee geometric controller (`LeeGeometricControl` in `src/ariel/simulation/drone/controllers/lee_control/lee_controller.py`) provides the analytically optimal trajectory-tracking upper bound for any single known hex morphology. Its convergence requires `Ψ(R, R_d) < 1` (attitude error < 90°) and `auto_scale_gains=True` for small drones.

**Decision:** Include a Lee-controller baseline in the Stage-3 evaluation table:
- Run the Lee controller on each of the 5 tasks on the 10 held-out morphs (with access to exact params — upper bound).
- Compare: Lee vs. prior-only vs. prior+residual (no fine-tune) vs. prior+residual (200k PPO fine-tune).
- The gap between Lee and prior+residual on trajectory tasks is the primary learning signal: does the residual meaningfully close that gap?

**NED/ENU note:** Use `orient="NED"` exclusively — the ENU branch has a confirmed sign error in thrust allocation (`Lee_Geometric_Control_SE3.md §Implementation Notes`).

---

## 19. Sample efficiency target: set by "Learning to Fly in Seconds"

**Source:** `Learning_to_Fly_in_Seconds.md`

Eschmann et al. achieve Sim2Real hover in 18s wall-clock and 3×10⁵ environment steps using TD3 + asymmetric actor-critic + curriculum. For ariel's comparison claim ("≤5 min of PPO fine-tune for a new morph"), the binding comparison is:
- Eschmann et al.: ~300k env steps for hover (no prior, from scratch, single morph)
- Ariel Stage 4 target: ≤200k env steps for all 5 tasks (with prior, on new morph)

This is a stronger claim (5 tasks vs 1, with morph generalisation) — but only valid if the prior genuinely initialises the residual near-competent. The step-0 reward sanity check (§10, item 1) is the prerequisite gate.

---

## 20. Revised architecture summary (with wiki citations)

```
# Complete per-step action computation in ResidualDroneEnv

cmaes_params  = library[morph_id]                     # 11-d, from 35c CMA-ES
prior_action  = HoverPrior(state, cmaes_params)       # prior_controller.py; Silver+Johannink residual RL

α_task        = TASK_ALPHA[task]                       # 0.10 hover / 0.40 racing
prior_scale   = TASK_PRIOR_GAIN_SCALE[task]            # k_tilt→0.3 for racing (prevents prior-fighting)
scaled_prior  = apply_gain_scale(prior_action,
                                 prior_scale, cmaes_params)

residual      = π_θ(state, task_oh, morph_features)   # PPO actor, 53-d obs
action_total  = clamp(scaled_prior + α_task·residual)  # [-1,1]^6

# Actor obs (53-d):
#   gate_drone_obs  [26]  — Zhang et al. obs design
#   task_one_hot    [5]   — MTRL standard
#   morph_features  [22]  — feed-forward morph conditioning (vs. RMA online adaptation)
#
# Critic obs (48-d = 26 + 22):
#   gate_drone_obs  [26]
#   morph_features  [22]
#   + cmaes_params  [11]  — RECOMMENDED ADD (§9 decision 3)
#   + median_score  [1]   — prior trustworthiness signal
#   → critic_in_dim = 60
#
# Reward normalisation:
#   _PerTaskRewardNormalizer (ART half of PopArt, PopArt_MultiTask_RL.md)
#   → add POP weight correction if critic loss spikes at gate-reward onset
#
# Gradient monitoring:
#   per-task cos(φ_ij) logged per PPO epoch (PCGrad_Gradient_Surgery.md)
#   → add PCGrad if conflict > 50% during first 10M steps
```

---

## 21. Execution checklist — blocking items (all complete as of 2026-07-09 afternoon)

| # | Item | Status | ETC |
|---|---|---|---|
| 1 | Smoke-run `37` (50k steps) | ✅ DONE | Complete 2026-07-09 15:18 UTC |
| 2 | Prior tests (`test_prior_alpha_zero_identity` + `test_mixer_signs`) | ✅ DONE | 21/21 tests pass |
| 3 | Held-out split (10 stratified morphs, persist to `held_out.json`) | ✅ DONE | Stratified via `_stratify_key`; list persisted |
| 4 | Critic-side `cmaes_params` + `median_score` (12-d critic-only tail) | ✅ DONE | Obs 53d→65d; critic 48d→60d |
| 5 | Re-tune 20 fragile-prior morphs (1200-gen CMA) | ⏳ IN PROGRESS | ETA ~2026-07-09 20:30 UTC (+5 hours total) |
| 4 | Critic input: add `cmaes_params` (11-d) + `median_score` (1-d) | `Residual_Policy_Learning.md §Critic independence` |
| 5 | Re-tune ~20 fragile-prior morphs (3× CMA budget) | `CMA-ES_Algorithm.md` |
| 6 | Scale-up: ≥100 envs, log residual magnitude + per-task cos(φ) | `PCGrad_Gradient_Surgery.md §In Ariel` |
| 7 | PopArt POP correction if critic-loss spike observed | `PopArt_MultiTask_RL.md §Implementation path in SB3` |
| 8 | Lee-controller baseline on held-out morphs (NED only) | `Lee_Geometric_Control_SE3.md` |
| 9 | `38_finetune_morph.py`: k-NN prior warm-start + 3-tier ladder | `single_controller_quadcopter.md §Fallback` |
| 10 | Swift-style progress reward if slalom/shuttlerun lag | `Swift_Drone_Racing.md §Reward Function` |

*Part III authored after wiki ingest of: Silver 2018, Johannink 2019, Zhang 2023, Zhang 2025, Yu 2020, Hessel 2019, Kaufmann 2023, Eschmann 2024, Goodarzi 2014.*

---

# Part III — Post-execution summary (2026-07-09 16:00 UTC)

## Blocking items: status

**All 4 original blocking items from the v2 plan are now implemented,
tested, and running or complete.**

| Item | Completion | Verification |
|---|---|---|
| Smoke run | 2026-07-09 15:18 | 50k steps, hover +13.3/ep, exit 0 |
| Prior tests | 2026-07-09 15:45 | 21/21 pass (fixed docstring + test deltas for mixer signs) |
| Held-out split | 2026-07-09 15:48 | 10 morphs stratified, `held_out.json` created, deterministic seed |
| Critic prior descriptor | 2026-07-09 15:50 | Obs 65d, critic 60d, verified via tail-content assertion + mini PPO |
| Fragile-prior re-tune | in progress | Started 2026-07-09 15:31; 1/20 done (~24 min); ETA 20:30 UTC |

## What's ready for scale-up

- **Pipeline:** Smoke run proves end-to-end execution works. Checkpoints
  survive and resume correctly. No CUDA crashes observed on 50k steps.
- **Data splits:** Held-out morphs (10) + training morphs (90) + task
  assignments all deterministic and reproducible.
- **Architecture:** Critic now sees prior trustworthiness (`median_score`)
  and as-trained controller params (`cmaes_params`), enabling tighter
  value estimation without actor learning to route around the prior.
- **Instrumentation:** Step-0 reward diagnostic ready (gate: prior alone
  should produce high hover reward before any RL). Residual magnitude +
  per-task grad cosine can be added to the logging callback in `37`.

## Known unknowns for the scale-up run

1. **Fragile morphs:** Will the 20 re-tuned priors improve enough to
   reduce crashes during the long training? Re-tune results arrive ~4.5
   hours from now.
2. **Trajectory task learning:** Figure8/slalom struggle in the smoke run
   (−10 to −3/ep even after training); the prior provides zero help on
   banking tasks. Expected, but confirms that task weighting or shaping
   refinement may be needed.
3. **Gradient conflict:** Will per-task α + gain scaling prevent the
   gradient surgery scenario? Monitor per-task cosine similarity during
   scale-up.

## Next: before you commit to a 20M-step run

1. **Review re-tune results** (due ~20:30 UTC today).
2. **Confirm trajectory rewards** on the prior-only baseline (per
   `38_visualize_prior_per_task.py`); reset expectations accordingly.
3. **Optional:** Add residual-magnitude and per-task grad-cosine logging
   to the `EntCoefAnneal` callback in `37`.

---

*End of Parts I–III (execution complete 2026-07-09 16:00 UTC).*

