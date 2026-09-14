# Recommendations: Drone Evolution + Generalist Control

Date: 2026-07-27. Synthesis of (a) the codebase audit (`DRONE_CODEBASE_AUDIT.md`),
(b) ~76 autoresearch experiments on the residual multi-task controller
(`spear/library/autoresearch_log.md`), and (c) the morphology break-threshold study
(`spear/library/40_morph_break_analysis.py`, results in `spear/library/morph_break_out/`).

Goal: make **morphology evolution** and a **generalist (morphology + task conditioned)
controller** work together properly.

---

## 1. What the break-threshold study proved (the empirical core)

A fresh 20M-step PPO hover specialist on the canonical hexacopter (success = alive
600 steps AND final drift < 0.5 m, policy blind to morph change):

| Perturbation axis | Single-arm tolerance | All-arm tolerance | Failure mode |
|---|---|---|---|
| **Arm azimuth** (in-plane) | **±2.5°** (arm 3: ±5°) | σ=2.5° → 37% success; σ≥5° → 0% | cliff — mixing matrix rotates, control misdirected |
| **Arm pitch** (out-of-plane) | **±12.5° to ±22.5°** per arm | σ=7.5° → 88%; σ=15° → 63%; σ=30° → 0% | gradual — moment arms shrink, thrust axis intact |

Two structural facts:

1. **Azimuth is 5–10× more brittle than pitch.** Az perturbation rotates rows of the
   effective mixing matrix, so the policy's roll/pitch commands produce partially
   wrong torques immediately. Pitch perturbation only scales control authority.
2. **Failure is drift, not crashing.** Survival stayed ~100% almost everywhere; the
   policy loses *precision* (XY drift grows past 0.5 m) long before it loses
   *stability*. Break thresholds measured by crash rate would be wildly optimistic.

Corollary reward pathology: the telescoping distance reward has ~zero gradient at the
target — canonical drift settled at 0.138 m and altitude equilibrium at 1.57 m vs the
1.50 m target. Training curves showed std collapse (→0.24), explained-variance spikes
(→−3..−5), and approx_kl spikes (0.04).

**Tuned intervention result** (`hover_policy_tuned.zip`, three additive changes:
quadratic centering bonus c=0.1, ent_coef=0.005, clip_range_vf=0.2). Same 20M steps,
same morph-blind policy, same eval protocol:

| Metric                   | Baseline           | Tuned              | Change |
|--------------------------|--------------------|--------------------|--------|
| Canonical hover drift    | 0.138 m            | **0.047 m**        | ~3× tighter |
| Std at 20M               | 0.24 (collapsed)   | **0.59** (healthy) | held  |
| AZ single-arm tolerance  | ±2.5° (cliff)      | **±30°** (tested max, still 100%) | ≥10× |
| PITCH single-arm         | ±12.5–22.5°        | **±45°** (tested max, still 100%) | 2–3× |
| AZ all-arm σ (100% succ) | σ=2.5 (37% succ)   | **σ=12.5** (100%)  | ≥5× |
| PITCH all-arm σ=30°      | 0% succ            | **100% succ**      | full domain |

**Read this carefully:** the tuned policy was trained on a *single* canonical
morphology and never saw any perturbation. The apparent "morph brittleness" of the
baseline was almost entirely a symptom of **policy over-specialization on the shared
reward pathology**, not a true structural limit. When the policy retains higher
entropy and centers itself with a proper gradient near target, it also has
enough action-space slack to compensate for wrong mixing matrices and shortened
moment arms. The mixing-matrix / moment-arm interpretation of the failure modes is
still correct — but the *magnitude* of the tolerance band is set by the controller,
not the physics. Plot: `morph_break_out/baseline_vs_tuned.png`.

---

## 2. Recommendations for morphology **evolution**

**E1 — Make azimuth-symmetry a first-class fitness/viability signal.**
The az cliff at ±2.5° means a *fixed* controller effectively cannot evaluate az-varied
morphs: fitness collapses to controller-mismatch noise, not morph quality. Either
(a) evaluate each morph with a controller adapted to it (CMA-ES prior per morph — the
hex-library pipeline already does this), or (b) evolve primarily along pitch/length/
motor axes and keep az near-symmetric with a strong prior or repair operator.
`min_azimuth_gap_deg()` in `40_morph_break_analysis.py` is a ready viability filter.

**E2 — Use tolerance bands to set mutation step sizes per gene.**
Mutation σ should be axis-aware: az mutations of ±2.5° already cross the controller's
break threshold, while pitch mutations of ±10° stay inside it. A shared σ for all
angular genes either stalls az exploration or destroys pitch gradualism. Concretely:
σ_az ≈ 1–2°, σ_pitch ≈ 5–10° when evaluation reuses a nearby-morph controller.

**E3 — Evolvability requires the controller to move with the morph.**
The library pipeline (sample morph → CMA-ES hover prior → store) is the right
skeleton: it guarantees every evaluated morph has a matched baseline controller.
Keep it; never score morphs with a single frozen policy (the break study is the
demonstration of why).

**E4 — Fix the torch/numpy dynamics drift before trusting cross-backend evolution.**
Accelerations diverge up to ~0.62 abs between `_build_torch_dynamics` and
`drone_sim.dynamics_func` (xfail(strict) tests in
`tests/unit/test_simulation/test_tasks/test_torch_numpy_dynamics_equivalence.py`).
If CMA-ES priors are fit on numpy and RL fine-tunes on torch, part of the residual
budget is spent correcting *backend* error, not morph error. Reconcile the two (or
commit to torch-only for both stages) before scaling evolution.

**E5 — Success metric for evolution eval: survival AND drift, 600+ steps.**
Adopt the study's metric. Short episodes + crash-only checks systematically overrate
morphs because drift failure takes hundreds of steps to express.

---

## 3. Recommendations for the **generalist controller**

**G1 — Domain-randomize morphology during training, with axis-aware ranges.**
The single biggest untested lever. The specialist was morph-blind and still tolerated
±20° pitch — a policy *trained across* perturbations should widen both bands.
Randomize pitch broadly (σ up to ~15°) and az narrowly (σ ≈ 1–2°) at first; widen az
only after conditioning (G2) is verified to work.

**G2 — Verify the policy actually uses its 22-d morph features. VALIDATED 2026-07-27.**
Frozen-features ablation on the exp_075 residual MTRL checkpoint
(`41_conditioning_diagnostic.py`): freezing `morph_features` to the library mean
drops the weighted metric from 44.93 → 9.86 (−78%). Every task drops: hover −47%,
fig8 −86%, slalom −83%, shuttle −78%, circle −87%. The policy heavily conditions
on its 22-d morph slice, so the generalist path does NOT need an auxiliary
reconstruction loss; extending the training distribution (G1) should work through
the existing conditioning pathway.

**G3 — Keep the residual architecture; the prior earns its keep on the brittle axis.**
CMA-ES hover prior + RL residual (alpha-blended) means the mixing-matrix correction
for az-perturbed morphs lives in the *prior* (refit per morph, cheap: ~289 s/morph on
GPU) while RL only learns task behavior. This is the cleanest split of
morph-adaptation vs task-competence found so far. Autoresearch confirmed
hover-prior-only (prior on for hover, off for trajectories) as a committed win.

**G4 — Reward shaping: add a quadratic centering term to every task.**
The zero-gradient-at-target pathology is a property of the telescoping reward, not
of hover — figure8/slalom/circle tracking will tolerate the same lateral offset.
The `-c·‖pos−target‖²` term (c≈0.1) is one line in the env; apply per-waypoint.

**G5 — Guard against over-specialization with entropy + value clipping.**
std collapse to 0.24 by 20M steps means the policy stops exploring long before
convergence in a *single*-morph setting; under morph randomization premature
collapse is worse (it locks in the mean-morph solution). ent_coef≈0.005 and
clip_range_vf≈0.2 addressed the observed collapse and value-loss spikes; carry
these into `37_train_residual_mtrl.py` as an autoresearch candidate (one change
per experiment, per the loop rules).

**G6 — Cross-task interference remains the open front for the generalist.**
Autoresearch history: figure8 is fragile and repeatedly the casualty of otherwise-
good changes; per-task velocity coefficients helped, per-task actor heads did not.
Next candidates in order of expected value: task-conditioned normalization, PCGrad-
style gradient surgery, and per-task advantage normalization. Test on the standard
metric = (hover + 2·(figure8 + slalom + shuttle + circle))/9.

---

## 4. Priority order (research plan)

1. **VALIDATED** — tuned az/pitch sweeps done. Shaped reward + entropy floor + vf
   clipping widen tolerance bands by 5–10× on both axes with no morph exposure. G4/G5
   are the highest-ROI changes in the whole plan. **Next action: port these three
   changes into `37_train_residual_mtrl.py` as the next autoresearch experiment**
   (one at a time per loop rules — see updated `autoresearch_program.md` §top tier).
2. **VALIDATED** — G2 conditioning diagnostic done. exp_075 policy loses 78% of its
   metric when morph features are frozen. Conditioning works; no auxiliary loss
   needed. Next lever is genuinely G1 (morph domain randomization at training).
3. **Short** — E4 dynamics reconciliation. Everything downstream inherits this error.
4. **Medium** — G1 morph-randomized training of the residual MTRL policy with
   axis-aware ranges; re-run the break sweep on the result (script is reusable via
   `--suffix`/`--out-dir`) to quantify additional widening beyond #1.
5. **Medium** — E1–E3: wire the evolution loop so every candidate morph gets a
   refit CMA-ES prior; use az-gap viability filter; axis-aware mutation σ.
   *Reconsider σ_az constraint post-#1*: the effective az tolerance band under a
   properly tuned controller may allow σ_az up to ~5°, not the original ~1–2°.
6. **Long** — G6 cross-task interference; widen az randomization once conditioning
   works; drop per-morph refits only when a fully conditioned generalist matches
   refit-prior performance on held-out morphs.

## 5. Do not reuse / do not retry (from experiment history)

- Per-morph from-scratch RL training as the *evolution inner loop* (cost prohibitive;
  the CMA-ES prior + shared residual replaces it).
- Per-task actor heads (tried in autoresearch; hurt the metric).
- Single frozen policy for evaluating az-varied morphs (break study: invalid signal
  beyond ±2.5°).
- Crash-based break metrics (survival is ~100% while precision is already gone).
