# Implementation Plan — Crucial Papers for Drone MTRL

**Audience: you are an implementing model.** This document is your work order.
Read it fully before writing code. You will implement the mechanisms from six
papers, each as a self-contained module, inside a **new, separate folder**:

```
examples/spear/library/paper_impls/
├── README.md                     # one-paragraph index of the folder
├── common.py                     # shared helpers (baseline loading, metric)
├── p1_song2021_progress.py       # Song et al. 2021 — progress-projection reward
├── p2_kaufmann2023_swift.py      # Kaufmann et al. 2023 — track curriculum + eval
├── p3_molchanov2019_s2mr.py      # Molchanov et al. 2019 — morph/dynamics randomization
├── p4_johannink2019_residual.py  # Johannink et al. 2019 — residual-α ablation
├── p5_yu2020_pcgrad.py           # Yu et al. 2020 — PCGrad gradient surgery
└── p6_sodhani2021_care.py        # Sodhani et al. 2021 — context-conditioned encoders
```

Create that folder and those files. Do **not** modify the existing training
scripts (`27_*`, `37_*`, `38_*`, `40_*`, `42*`) or `envs/residual_drone_env.py`
in place — every paper implementation must be importable/runnable on its own and
must *wrap or subclass* the existing code. One paper = one file = one runnable
experiment.

---

## Repository context (read these before starting)

The project trains a **multi-task RL (MTRL) generalist controller** for evolved
drone morphologies:

- **Env**: `src/ariel/simulation/tasks/torch_drone_gate_env.py`
  (`TorchDroneGateEnv`) — vectorized torch dynamics, waypoint-gate tracks,
  telescoping progress reward `d_old − d_new` + gate bonus.
- **Residual MTRL**: `examples/spear/library/envs/residual_drone_env.py`
  (`ResidualDroneEnv`) — analytical CMA-ES hover prior + PPO residual,
  5 tasks (`hover, figure8, slalom, shuttle-run, circle`), 65-d obs with
  22-d morph features + critic-only prior descriptor.
- **Main trainer**: `examples/spear/library/37_train_residual_mtrl.py`
  (`MTRLActorCriticPolicy`, per-task reward normalization,
  `GradientCosineCallback` already logs inter-task gradient cosines).
- **Morph library**: `hex_sampler.py`, `__data__/hex_library/v1`.
- **Break-threshold study**: `40_morph_break_analysis.py` +
  `examples/DRONE_RESEARCH_RECOMMENDATIONS.md` — azimuth tolerance is a cliff
  at ±2.5°, pitch degrades gradually to ±20°; failure mode is *drift*, not
  crashing.
- **Drawn-trajectory pipeline**: `42a/42b/42c_*` — scratchpad tracks trained
  with waypoint gates + progress reward.

**Standard metric** (use it everywhere):
`metric = (hover + 2·(figure8 + slalom + shuttle + circle)) / 9`,
computed from per-task episode rewards as in `_eval_per_task` of `37_*`.

**Ground rules**
1. Run everything with `uv run <script>` (never bare `python`).
2. Verify any API attribute you have not seen in the code (use `dir()`/`help()`
   or read the source) before calling it.
3. Motor-state pitfall: `w = 0` in `TorchDroneGateEnv` state is **mid-throttle**,
   not zero RPM. Any custom reset must set the hover-equivalent value (see
   `BlueprintTrajEnv._reset_envs` in `42b_train_blueprint_traj.py`).
4. One experimental change per training run; always report the standard metric
   against the unmodified baseline trained with the same budget/seed.
5. CPU-first defaults; `--device` flag for CUDA. Short smoke budget
   (`--steps 100_000`) must run in minutes and be the default; full budgets
   behind explicit flags.
6. Outputs under `__data__/paper_impls/<paper>/<timestamp>/` with a
   `config.json` and a `results.json` (metric, per-task rewards, seed, steps).

---

## P1 — Song et al. 2021, *Autonomous Drone Racing with Deep Reinforcement Learning* (IROS 2021)

**Core idea to port.** Reward is **progress along the path projection**
(`s(t) − s(t−1)` where `s` is arc-length progress of the drone's projection
onto the gate-to-gate path segment), plus a safety/tracking penalty — instead
of the raw distance telescoping `d_old − d_new` currently in
`TorchDroneGateEnv`. The projection reward has non-zero gradient everywhere
along the track and does not flatten near the waypoint, directly attacking the
documented zero-gradient-at-target pathology (drift settles ~0.14 m off
target; see `DRONE_RESEARCH_RECOMMENDATIONS.md`, finding G4).

**Implement** in `p1_song2021_progress.py`:
- `ProgressRewardEnv(TorchDroneGateEnv)`: override the reward computation.
  Progress term: project position onto the segment (prev_gate → target_gate),
  reward `k_p · Δprojection`; add a lateral-deviation penalty
  `−k_d · ‖pos − closest_point_on_segment‖²` (this is also the paper's safety
  term and subsumes recommendation G4's quadratic centering).
  Keep gate bonus and existing shaping flags untouched.
- Training entry point mirroring `42b_train_blueprint_traj.py` (same canonical
  hex, same PPO hyperparameters) with `--reward {telescoping,progress}` so a
  paired comparison is one flag.

**Experiment / gate.** Same drawn or demo track (use
`42a --demo figure8`), 2M steps, 3 seeds per reward variant. Success: progress
variant ≥ telescoping variant on waypoints-passed AND mean lateral deviation
reduced ≥ 30%.

## P2 — Kaufmann et al. 2023, *Champion-level drone racing using deep reinforcement learning* (Nature)

**Core idea to port.** (a) **Random initialization along the track** for
uniform state coverage (already in `TorchDroneGateEnv`; keep). (b) **Track
curriculum**: train on a *distribution* of tracks rather than one, so the
policy learns gate-relative flying instead of memorizing a trajectory.
(c) Their eval protocol: fixed-start timed laps.

**Implement** in `p2_kaufmann2023_swift.py`:
- A track generator producing random smooth closed tracks (random control
  points + the resampling utilities from `42a_draw_trajectory.py` —
  import via `importlib` like `42c` does; do not duplicate the resampler).
- A `MultiTrackVecEnv` that assigns a different generated track per worker and
  regenerates the track on episode reset every N episodes.
- Eval: on K held-out generated tracks *and* on a user-drawn track from `42a`,
  report waypoints/sec (lap time proxy).

**Experiment / gate.** 5M steps single-track vs multi-track training, then eval
on 5 held-out tracks. Success: multi-track policy completes ≥ 80% of waypoints
on held-out tracks where the single-track policy generalizes worse (report
both).

## P3 — Molchanov et al. 2019, *Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors*

**Core idea to port.** A single policy generalizes across *physically
different* quadrotors when trained with **dynamics/morphology randomization**
(mass, inertia, motor lag, thrust coefficients) + noise. This is the paper
behind recommendation **G1 (morph domain randomization, axis-aware ranges)**.

**Implement** in `p3_molchanov2019_s2mr.py`:
- A morph-randomizing training loop over `ResidualDroneEnv` workers: each
  worker's morph is resampled from perturbed canonical-hex genomes with
  **axis-aware σ** (the break study's tolerance bands): σ_pitch ∈ {5°, 10°, 15°},
  σ_az ∈ {1°, 2°} — reuse `perturbed_genome`/`genome_to_morph` from
  `40_morph_break_analysis.py` via `importlib`.
- Additionally randomize `tau` (motor lag) and mass ±10% via the params dict
  passed to the env, following the paper's dynamics-randomization list.
- After training, re-run the break sweep (`40_morph_break_analysis.py` is
  reusable via `--suffix`/`--out-dir`) on the randomization-trained policy.

**Experiment / gate.** Success: single-arm azimuth tolerance band widens vs
the `hover_policy_tuned.zip` baseline (currently ±2.5°), pitch band widens
beyond ±22.5°, with canonical-morph success ≥ 95% retained.

## P4 — Johannink et al. 2019, *Residual Reinforcement Learning for Robot Control*

**Core idea to port.** The formal grounding of the repo's residual
architecture: policy output is added to a hand-designed controller, and the
**blend weight α and residual magnitude** determine the split between prior
stability and learned competence. The repo fixed α per task
(hover 0.10, trajectories 0.40) without a systematic study.

**Implement** in `p4_johannink2019_residual.py`:
- An α-ablation harness over `ResidualDroneEnv`: α ∈ {0.05, 0.1, 0.2, 0.4, 0.8,
  1.0-no-prior}, single morph (canonical hex), 2 tasks (hover, figure8),
  short fixed budget per cell.
- Also implement the paper's *residual warm-up*: train the first M steps with
  α annealed 0 → target (the paper initializes the residual to near-zero so
  the prior dominates early). Compare fixed-α vs annealed-α.

**Experiment / gate.** Produce an α-vs-metric curve per task
(`results.json` + a PNG). Success: reproduce the paper's qualitative result —
performance is non-monotonic in α with an interior optimum — and report
whether the current per-task defaults (0.10/0.40) sit near it.

## P5 — Yu et al. 2020, *Gradient Surgery for Multi-Task Learning* (PCGrad)

**Core idea to port.** When per-task policy gradients conflict (negative
cosine), project each task's gradient onto the normal plane of the conflicting
one before applying the update. This is the pre-registered escalation in
`37_train_residual_mtrl.py`'s `GradientCosineCallback` ("cosine < 0 in >50% of
logged updates after 20M steps") and the top open front **G6 (cross-task
interference — figure8 is repeatedly the casualty)**.

**Implement** in `p5_yu2020_pcgrad.py`:
- A `PCGradPPO(PPO)` subclass (or an optimizer wrapper) that, inside the PPO
  update, splits the minibatch by task one-hot (obs layout documented at the
  top of `37_*`), computes per-task gradients, applies pairwise PCGrad
  projection, then steps. Keep the value-function update unmodified (surgery
  on actor gradients only, per the paper's RL experiments).
- Reuse `MTRLActorCriticPolicy` and the worker setup from `37_*` via
  `importlib`; a `--pcgrad {on,off}` flag makes the paired run one switch.

**Experiment / gate.** 5M-step paired runs (same seeds), standard metric.
Success: metric not worse overall AND figure8 per-task reward improves;
also log the fraction of conflicting minibatches so the 20M escalation
trigger can be evaluated cheaply later.

## P6 — Sodhani et al. 2021, *Multi-Task Reinforcement Learning with Context-based Representations* (CARE)

**Core idea to port.** Condition the state encoding on task **context**
through a mixture of encoders with context-derived soft attention, instead of
a hard per-task encoder pick. The repo's `MTRLActorCriticPolicy` currently
hard-gates one task encoder by one-hot (`(all_task_lats · one_hot).sum`);
CARE replaces the one-hot with learned attention weights over the encoder
mixture — the "task-conditioned normalization / representation" candidate
named in G6, and a soft path for task transfer (figure8 could borrow from
circle's encoder).

**Implement** in `p6_sodhani2021_care.py`:
- `CAREActorCriticPolicy(MTRLActorCriticPolicy)`: replace one-hot gating with
  attention `softmax(f(task_context))` over the K task encoders (K need not
  equal the number of tasks; make it a flag, default K=5). Task context =
  the task one-hot passed through a small embedding (the repo has no language
  metadata; the embedding stands in for CARE's context encoder).
- Keep everything else in `37_*`'s setup identical; `--policy {onehot,care}`.

**Experiment / gate.** 5M-step paired runs, standard metric, 3 seeds.
Success: metric ≥ one-hot baseline and reduced variance across seeds; report
the learned attention maps (which encoders each task uses).

---

## Order of work and dependencies

1. **P1** (progress reward) — smallest, unblocks better tracking signal used by
   P2's curriculum. No dependency.
2. **P4** (residual α) — pure harness, no new training machinery; validates the
   residual defaults everything else builds on.
3. **P3** (morph randomization) — highest expected value per
   `DRONE_RESEARCH_RECOMMENDATIONS.md` priority #4 (G1); depends only on `40_*`
   utilities.
4. **P5** (PCGrad) and **P6** (CARE) — the two G6 candidates; run after P1/P3 so
   the MTRL baseline is current. They are alternatives — implement both, compare
   on the same seeds.
5. **P2** (track curriculum) — last; benefits from P1's reward and reuses the
   42-series pipeline.

Work through them in that order, marking progress in
`paper_impls/README.md` (one status line per paper: `pending / in progress /
done + one-line result`). After each paper's gate experiment, append the
result to the README before moving on.
