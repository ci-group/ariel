# Ariel Autoresearch Program

You are an autonomous RL experiment loop for the **ariel generalist drone controller**.
Propose one code change, run a short training experiment, keep the change if it
improves the ratchet metric, revert it if not, log the result, and continue
**indefinitely**. Do not stop. Do not ask for permission. Do not check in with the user.

---

## Project Context

A residual PPO policy (`π_θ`) outputs motor corrections on top of a CMA-ES-tuned
analytical hover prior. The prior handles "how to fly this body"; the residual handles
"how to track this task." 5 tasks: hover, figure8, slalom, shuttle-run, circle.
100 hex morphologies (90 training, 10 held-out, stratified).

**Architecture** (`37_train_residual_mtrl.py`):
- MTRL actor-critic: shared encoder (drone+morph → latent) + per-task encoders (gates →
  latent) + actor trunk → residual mean
- Per-task critics: each sees full 60d obs (base_obs + morph_features + cmaes_params +
  median_score)
- Observations: 65d total (gate+drone 26, task_oh 5, morph_features 22, cmaes_params 11,
  median_score 1). Last 12d are **critic-only** — actor path never sees them.
- Action: residual in [−1,1]^6; env applies `α_task × residual` on top of prior output.

**Known challenges:**
- Trajectory tasks (figure8, slalom, shuttle-run, circle) are sparse-reward. The prior
  contributes nothing useful on them. They need dense reward shaping to learn.
- Hover reward scale (~0.0125/step) ≪ gate-spike scale (+1). Per-task normalization
  (`_PerTaskRewardNormalizer`) addresses this but is only the ART half of PopArt.
- Risk: residual collapses to zero (policy learns to ignore it). Monitor `|res|` in eval.
- Risk: gradient conflict between hover (dense, early) and trajectory tasks (sparse,
  delayed). Per-task cos(φ) is logged by `GradientCosineCallback` in `37`.

**Wiki** (read before proposing changes to a relevant subsystem):
- `.claude/wiki/Residual_Policy_Learning.md` — α, prior-fighting failure mode
- `.claude/wiki/PCGrad_Gradient_Surgery.md` — gradient conflict mitigation
- `.claude/wiki/PopArt_MultiTask_RL.md` — reward normalisation (ART vs full PopArt)
- `.claude/wiki/Swift_Drone_Racing.md` — gate-progress reward shaping
- `.claude/wiki/ResidualDroneEnv.md` — env API, TASK_ALPHA, reward structure

**Cross-cutting empirical findings** (read before proposing reward or robustness changes):
- `examples/DRONE_RESEARCH_RECOMMENDATIONS.md` — 2026-07-27 morph-break study
  results and derived priorities (quadratic centering, ent_coef bump, vf clipping,
  axis-aware morph randomization). Trajectory tasks likely inherit the specialist's
  ~0.14 m steady-state drift; std collapse to 0.24 is the over-specialization mode.

---

## Files You May Modify

| File | Permitted changes |
|------|------------------|
| `examples/spear/library/37_train_residual_mtrl.py` | PPO hyperparams (lr, gamma, gae_lambda, clip_range, n_epochs, batch_size, n_steps), architecture dims (ENCODER_HIDDEN, ENCODER_LATENT, ACTOR_HIDDEN), entropy annealing (ent_start, ent_end, schedule shape), log_std_init, gradient clipping, callbacks, eval frequency |
| `examples/spear/library/envs/residual_drone_env.py` | Reward shaping weights, gate-progress multipliers, TASK_ALPHA values (per-task α), TASK_PRIOR_GAIN_SCALE values, episode length, crash penalty |

## Files You Must NEVER Modify

- `prior_controller.py` — hover prior, analytically correct
- `hex_sampler.py` — invalidates library if changed
- `test_prior_controller.py` — tests must remain valid
- `gate_configs.py` — gate geometry is fixed
- Any file under `__data__/` — library data and checkpoints
- Any file not listed in the "May Modify" table above

---

## Experiment Procedure

### 1. Read current state

Use `sqz_read_file` to read both modifiable files. Read `autoresearch_log.md` for
recent history. Do not repeat an idea that was tried and reverted in the last 5
experiments unless you have a new angle on it.

### 2. Check wiki if needed

If you are proposing a reward shaping or architecture change, read the relevant wiki
page first. If you are proposing gradient surgery (PCGrad), read
`PCGrad_Gradient_Surgery.md §In Ariel` for the exact SB3 integration path.

### 3. Propose ONE change

State your hypothesis explicitly:
> *"I believe X will improve the weighted metric because Y (supported by Z)."*

Rules:
- ONE focused change per experiment. Do not combine multiple ideas.
- No refactoring while experimenting. Surgical changes only.
- If the change requires a new constant, add it near the top of the file with a comment.
- Prefer changes that help **trajectory tasks** (figure8, slalom, shuttle-run, circle).
  Hover already converges quickly; the bottleneck is trajectory task performance.

**High-value experiment ideas (roughly priority order):**

*Top tier — derived from the morph-break study 2026-07-27 (see*
*`examples/DRONE_RESEARCH_RECOMMENDATIONS.md`)*. The specialist showed three
pathologies that likely apply to the multi-task residual policy: (a) telescoping
distance reward has ~0 gradient at target → ~0.14 m residual drift + biased
altitude, (b) policy std collapses to 0.24 by 20M steps (over-specialization),
(c) explained_variance spikes to −3..−5 coinciding with value-loss spikes.
The tuned intervention held std at 0.59 with all three fixes on. Apply one at a time:

1. **Quadratic centering reward `-c·‖pos−target‖²`, c≈0.1** — one-line env change;
   addresses the zero-gradient-at-target pathology. Applies to hover and every
   trajectory task waypoint (~expected +0.5–1.5 on hover, +0.1–0.5 per traj task).
2. **ent_coef 0.0 → 0.005** (additive to `ent_start`/`ent_end` schedule) — direct
   guard against std collapse. Cheap, one keyword.
3. **clip_range_vf 0.2** (currently None) — targets the explained-variance/value-loss
   spikes; no policy-side change, low risk.
4. **Morphology domain randomization** (long-shot generalist unlock): axis-aware
   jitter each epoch — σ_az≈1° (break study cliff at ±2.5°), σ_pitch≈5°
   (tolerance ±12.5–22.5°). If frozen-feature ablation shows no gap → morph
   conditioning is broken, needs auxiliary reconstruction loss.

*Standing candidates (unchanged):*

5. Gate-progress reward multiplier for trajectory tasks (Swift-style progress reward:
   `λ × (d_{t-1}^gate − d_t^gate)` — currently zero if not already in env)
6. Per-task α tuning: hover 0.10 may be too low/high; trajectory tasks may want 0.5-0.7
7. Entropy annealing shape: try cosine or stepped schedule instead of linear
8. Critic hidden dim increase (critics currently share ACTOR_HIDDEN; larger critic ↔ better value estimates)
9. BC-regularization term (pull total action toward prior during early training, per Zhang 2025)
10. Per-task worker reweighting (more workers on hard trajectory tasks in `tasks` list)
11. PopArt POP weight correction (add when critic-loss spikes are visible in log)
12. γ (gamma) adjustment — trajectory tasks with sparse gates may benefit from γ → 0.995
13. Reward clipping or shaping for crash events
14. Learning rate schedule (cosine warmup + decay instead of constant 3e-4)

### 4. Apply the change

Use `Edit` to make the minimal change. Keep it to <20 lines of diff if possible.

### 5. Launch experiment (~2h, background)

**Always use the explicit `cd` prefix — never use a relative path or rely on CWD.**

Each experiment is a single 20M-step run (seed 0). See Step 6 in the skill for the
launch procedure (tmux + sentinel). Reference run command:

```bash
cd /home/user/Desktop/EvoDevo/ariel/.claude/worktrees/autoresearch && \
  uv run examples/spear/library/37_train_residual_mtrl.py \
    --steps 20000000 --num-envs 16 --inner-batch 4 \
    --eval-steps 3000 --device cuda \
    --library /home/user/Desktop/EvoDevo/ariel/__data__/hex_library/v1/library.npz \
    --seed 0 --out-dir /tmp/autoresearch_runs/exp_NNN \
    > /tmp/autoresearch_s0.log 2>&1
```

After launching, write the sentinel and schedule a 3600s wakeup. Do NOT wait.

### 6. Parse the results (when run completes — Step 6b in skill)

Find the `[after training]` block in `/tmp/autoresearch_s0.log`:
```
[after training] trained-policy rollout (3000 steps):
         hover: reward/ep= +66.619  ...
       figure8: reward/ep= +16.595  ...
        slalom: reward/ep= +19.675  ...
   shuttle-run: reward/ep= +11.202  ...
        circle: reward/ep= +13.079  ...
```

Extract `reward/ep` for each task. If a task shows `nan`, treat it as −100.

**Weighted metric = (hover×1 + figure8×2 + slalom×2 + shuttle-run×2 + circle×2) / 9**

Round to 3 decimal places.

### 7. Establish or retrieve baseline

The baseline is the metric of the last **COMMITTED** or **BASELINE** row at **20M steps**
in `autoresearch_log.md`. Old 250k-step metrics are NOT comparable. If the most recent
BASELINE/COMMITTED entry predates the 20M-step switch, run a clean eval first (no code
change), log as `BASELINE`, then start proposing changes.

### 8. Ratchet decision

**New metric > baseline** → COMMIT:
```bash
git add examples/spear/library/37_train_residual_mtrl.py examples/spear/library/envs/residual_drone_env.py
git commit -m "autoresearch: <one-line summary> (metric: +X.XXX → Y.YYY)"
```

**New metric ≤ baseline** → REVERT:
```bash
git checkout -- examples/spear/library/37_train_residual_mtrl.py examples/spear/library/envs/residual_drone_env.py
```

**Crash (non-zero exit)** → REVERT (same checkout command), log as CRASHED.

### 9. Log the experiment

Append to `examples/spear/library/autoresearch_log.md` using this format:
```
| NNN | YYYY-MM-DD HH:MM | X.XXX | ±X.XXX | STATUS | Hypothesis: <...> / Change: <...> |
```

Status: `BASELINE`, `COMMITTED`, `REVERTED`, `CRASHED`

---

## NEVER STOP

Once the loop has begun, **do NOT pause to ask the user anything**. Do not say:
- "Should I continue?"
- "Is this a good stopping point?"
- "Do you want me to try X instead?"

The user may be asleep. Continue indefinitely. Each experiment runs ~2h in background.
Use ScheduleWakeup(delaySeconds=3600, prompt="/autoresearch") inside Step 6 (after
launching) to schedule the first status check. After evaluation completes, use
ScheduleWakeup(delaySeconds=60) to immediately start proposing the next experiment.

If an experiment crashes, log it as CRASHED, revert, try something different.
If you hit three crashes in a row, read both modifiable files carefully for syntax
errors before proposing the next experiment.
