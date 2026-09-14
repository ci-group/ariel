# Drone Codebase Audit — spear, d_drones, e_drones_ec

**Date:** 2026-07-27 · **Branch:** `drones` (with `autoresearch/experiments` worktree)
**Method:** source survey of all three folders, branch diffs, CPU test-suite run
(53 tests), and targeted dynamics experiments on the residual environment.

---

## 1. Executive summary

Three generations of drone work exist in `examples/`:

| Folder | Era / focus | Status |
|---|---|---|
| `d_drones` | Morphology evolution → Lee-controller sim → single-morph PPO → fabrication (STL/USD/video) | Complete pipeline, working; several small hardcoded-path bugs |
| `e_drones_ec` | Evolutionary-computation experiments: spherical/CPPN genomes, body+brain CMA-ES, evo+RL | Exploratory; per-morph training, superseded for the generalist goal |
| `spear` | GPU-vectorized torch env, task suite (hover/figure8/slalom/shuttle/circle), CMA-ES hover prior, morph library, **multi-task residual RL (current generalist effort)** | Active; 50/53 tests pass; one test failure root-caused as a fixture bug, not physics |

**Headline findings:**
- The physics core (`TorchDroneGateEnv`) is sound: dynamics are identical across
  branches, deterministic, and 10–50× faster than the numpy `DroneGateEnv`.
- The one failing test (`test_zero_residual_hovers` in the autoresearch worktree)
  is **not** an env regression — it is a library-path resolution bug in the test
  fixture (§6.1).
- The reusable assets for the generalist controller are: the CMA-ES hover prior +
  `prior_controller.py`, the hex morph library, `TorchDroneGateEnv`, the v4 task
  suite, and the hover-prior-only residual architecture validated by the
  autoresearch loop (metric 30.039 at 100M steps).

---

## 2. `d_drones` — evolution-to-fabrication pipeline

**What is built (17 scripts):**
1. Morphology evolution (EA and CPPN-NEAT variants) producing drone genomes.
2. Simulation with a **Lee geometric controller** including automatic gain
   derivation from morphology (mass/inertia).
3. 3-stage CMA-ES controller tuning per morph.
4. Single-morph PPO on figure8 (`6_train_rl_figure8.py`).
5. Export chain: `DroneBlueprint` JSON IR → STL meshes → MuJoCo model → USD →
   rendered videos (`7_make_video.py`).

**Pros:**
- End-to-end: genome → simulated → fabricable artifact. The `DroneBlueprint`
  JSON IR is a clean interchange format reused by later folders.
- Lee controller + auto-gains gives an analytical baseline controller for *any*
  morphology — this is the intellectual ancestor of the spear CMA-ES prior.
- GateChecker and morphology-repair utilities are morph-agnostic and reusable.

**Cons / bugs:**
- `7_make_video.py:95,111` — hardcoded `/top_view.mp4` output paths; breaks on
  any non-default output dir.
- `15_cppn_neat_circle_to_mujoco.py:270` — linear arm-mass fit (0.034 kg/m) is
  extrapolated outside its fitted range for long/short arms; mass model untested
  there.
- `6_train_rl_figure8.py:159` — references `env.motor_tau`, which may be
  undefined depending on env construction path (AttributeError risk).
- `_viz_best.py:97` — hardcoded NED↔ENU axis swap; silently wrong if the source
  env convention changes.

**Physics caveats:**
- Lee auto-gain derivation ignores inertia cross-coupling terms — fine for
  symmetric quads, degraded for the asymmetric evolved morphs.
- MuJoCo export can clamp core body mass at 1e-4 **silently**, producing
  sim-vs-export dynamics mismatch for very light genomes.

---

## 3. `e_drones_ec` — evolutionary computation experiments

**What is built:**
- Spherical-angular genome EA; CPPN genome EA.
- Body+brain joint CMA-ES (morphology and controller co-optimized).
- Evolution + per-morph PPO on figure8 (`5_drone_evo_rl_figure8.py`).
- Numpy-vs-torch environment benchmarks (`bench_torch_env.py`).

**Pros:**
- Demonstrated the search-space trade-offs (spherical vs CPPN encodings) that
  informed the hex library's stratified sampling.
- The benchmarks quantify the torch env speedup that justifies the current
  GPU-vectorized training stack.

**Cons / bugs:**
- Per-morph training throughout — the exact paradigm the generalist controller
  replaces. Body+brain CMA-ES is compute-hungry and should not be reused.
- `5_drone_evo_rl_figure8.py:89` — device hardcoded to `cpu`; ignores CUDA even
  when available.
- `4_*.py:306` — a **binary** spin gene is mutated with Gaussian noise σ=0.5;
  mutation operator mismatched to the gene's domain.
- `bench_torch_env.py:106-175` — manually reimplements the env dynamics instead
  of calling the env; benchmark can silently drift from the real physics.

---

## 4. `spear` — current stack

### 4.1 Task suite (scripts 18–22)
Hover, figure8, slalom (100 gates), shuttle-run (4 gates), circle (4 gates).
Shared conventions: altitude z = −1.0 (NED), 1200-step episodes, v4 reward
profile (`UPRIGHT_BONUS=0.01`, `VELOCITY_REWARD_COEF=0.005`,
`TILT_TERMINATE_COS=0.0` i.e. tilt-termination disabled, `GATES_AHEAD=2`).

Issues:
- `18c_hover.py` **diverges from the v4 standard**: yaw penalty −1.0 (vs −0.1),
  an extra spin penalty, 32 envs. Results from 18c are not comparable to 18–22.
- `18`/`18c` duplicate a bespoke `TorchDroneHoverEnv` (~121 lines) instead of
  reusing the gate env with a single hover gate — maintenance hazard.
- `19_figure8.py:75-88` — reward shaping is conditional on warmstart-vs-cold,
  so warm and cold runs optimize *different* rewards; comparisons across the
  two modes are unfair.

### 4.2 Multi-task RL (script 27, v4 architecture)
Obs 30d = shared 18 + task 8 + one-hot 4. Shared encoder 18→128→32, per-task
encoders 8→128→32, actor trunk →256→256→6, per-task gated critics. Wraps
`TorchDroneGateEnv` 4× with a **hover reward override at line 366** — the
override lives in the training script, not the env, so the env alone does not
reproduce training rewards (documented pitfall for anyone evaluating outside 27).

### 4.3 CMA-ES hover prior (35c/35d) and morph library (34, hex_sampler)
- 9-parameter analytical prior per morph: 6 motor trims + `k_alt_p`, `k_alt_d`,
  `k_tilt`. Tuned per morph via CMA-ES (GPU: 289 s/morph, 3.3× over CPU).
- Stored in `__data__/hex_library/v1/library.npz` (90 morphs), consumed by
  `prior_controller.py`. Library-based EA follows Rehberg et al. (RA-L 2026).
- **This is the load-bearing asset**: the entire residual-MTRL stack assumes
  tuned `cmaes_params` exist for each morph. With default (untuned) warm-start
  params, prior-alone hover *diverges within ~110 steps* (verified in §6.1) —
  the CMA-ES tuning is not optional polish, it is required for stability.

### 4.4 `spear/library` — residual MTRL (autoresearch subject)
`37_train_residual_mtrl.py` + `envs/residual_drone_env.py`. 100M-step PPO over
90 morphs × 5 tasks. Validated (committed) design decisions from ~76 logged
experiments:
- **Hover-prior-only**: prior active for hover (α small), trajectory tasks
  learn free (α=1.0, no prior). The prior is *critical* for multi-morph hover
  and *harmful* for trajectory tasks.
- **Cosine LR** 3e-4→3e-5 over 100M steps: the unlock for figure8 (+7.5).
- **Per-task velocity reward coefs** (fig8 0.40, shuttle 0.20, others 0.32);
  raising any *second* task to 0.40 crashes fig8 — cross-task gradient
  interference on the shared trunk is real and reproducible.
- **`random_init=False`** for trajectory tasks eliminated seed bimodality.
- Rejected at scale: per-task actor heads (−9.9), vf_coef<0.5, gae_lambda>0.97.

Current best metric: **30.039** (hover 64.1, fig8 18.5, slalom 22.6,
shuttle 27.8, circle 34.2; weighted (1,2,2,2,2)/9).

---

## 5. Environment audit

### 5.1 `TorchDroneGateEnv` (`src/ariel/simulation/tasks/torch_drone_gate_env.py`)
- GPU-vectorized torch dynamics, NED frame, Euler integration at fixed `dt`,
  optional IIR action filter. 10–50× faster than the numpy `DroneGateEnv`.
- **Byte-identical across `drones` and `autoresearch/experiments` branches**
  (verified by diff) — no physics drift from the autoresearch loop.
- **Physics footgun (by design, documented):** motor state `w=0` is
  *mid-throttle*, not zero RPM. In `_reset_envs` (lines 493–514) the
  fixed-init path (`initialize_at_random_gates=False`) sets `motors=zeros`,
  i.e. drones spawn at mid-throttle. For high-TWR morphs this is a large
  initial thrust transient. The random-init path uses uniform[−1,1] motors —
  same expectation, wider spread. Any new code touching motor state must init
  relative to hover-equivalent throttle, not 0.
- Minor: env lacks a `render_mode` attribute → SB3 emits a UserWarning on
  every construction (cosmetic).

### 5.2 `ResidualDroneEnv` (`examples/spear/library/envs/residual_drone_env.py`)
- Wraps the torch env; injects prior effort (`prior_effort` for hover,
  `trajectory_effort` with an outer position loop for gates) and scales the
  policy residual by per-task α.
- Branch diff is confined to reward coefficients, per-task α values,
  `random_init=False`, and the `use_prior` flag — **no dynamics changes**.
  Verified experimentally: with identical morph params, both branches produce
  identical trajectories.
- The hover reward override pattern (see §4.2) recurs here: reward semantics
  are split between env constants and `TASK_*` dicts; keep them in the env
  (as done) rather than in training scripts.

### 5.3 Numpy `DroneGateEnv`
Kept as reference implementation; do not use for training (slow). The
`bench_torch_env.py` dynamics-reimplementation bug (§3) means the numpy env is
the only trustworthy cross-check for the torch dynamics — worth an explicit
torch-vs-numpy trajectory-equivalence test (none exists today).

### 5.4 Test-suite results (CPU, worktree, exp_076 training untouched on GPU)
53 tests: **50 pass, 1 fail, 2 warnings.**
- Fail: `test_zero_residual_hovers` — root-caused below (§6.1), fixture bug.
- Warnings: missing `render_mode` (cosmetic); benign SB3 vec-env warning.

---

## 6. Bug register (all findings, file:line)

### 6.1 `test_residual_drone_env.py` library-path fixture bug — **root-caused this audit**
`test_residual_drone_env.py:23` resolves the morph library as
`Path(__file__).resolve().parents[3] / "__data__/hex_library/v1/library.npz"`.
Inside a **git worktree** (`.claude/worktrees/autoresearch/`), `parents[3]` is
the worktree root, where `__data__/` (untracked) does not exist. The fixture
then silently takes its fallback path (lines 46–51): a synthesized morph with
**default warm-start `cmaes_params` instead of CMA-ES-tuned ones**.

Consequence: `test_zero_residual_hovers` asserts a hover guarantee the untuned
prior cannot deliver. Verified experimentally:
- Worktree (dummy params): diverges at t≈107, deterministically, under **both**
  `initialize_at_random_gates=True` and `False` → init mode is not the cause.
- Main repo (tuned library params): survives 600 steps, final drift **0.000 m**.

**The env physics are correct on both branches.** Fixes (either):
1. Resolve `LIBRARY` via the git common dir / an env var rather than
   `parents[3]`, or
2. `pytest.skip` the hover-guarantee assertions when the fallback path is
   taken (dummy params cannot certify hover; the comment at line 50 already
   admits this).

### 6.2 Other bugs by folder
| File:line | Bug |
|---|---|
| `d_drones/7_make_video.py:95,111` | Hardcoded `/top_view.mp4` output paths |
| `d_drones/15_cppn_neat_circle_to_mujoco.py:270` | Arm-mass linear fit (0.034 kg/m) extrapolated outside fitted range |
| `d_drones/6_train_rl_figure8.py:159` | `env.motor_tau` possibly undefined → AttributeError |
| `d_drones/_viz_best.py:97` | Hardcoded NED/ENU axis swap |
| `e_drones_ec/5_drone_evo_rl_figure8.py:89` | Device hardcoded `cpu` |
| `e_drones_ec/4_*.py:306` | Binary spin gene mutated with Gaussian σ=0.5 |
| `e_drones_ec/bench_torch_env.py:106-175` | Reimplements env dynamics manually; can drift from real physics |
| `spear/18c_hover.py` | Reward profile diverges from v4 standard (yaw pen −1.0, extra spin pen, 32 envs) — results incomparable |
| `spear/18*, 18c*` | Duplicated ~121-line bespoke `TorchDroneHoverEnv` |
| `spear/19_figure8.py:75-88` | Warmstart-vs-cold runs optimize different rewards |
| `spear/27_*.py:366` | Hover reward override lives in training script, not env |
| `src/.../torch_drone_gate_env.py` | Missing `render_mode` attr → SB3 warning (cosmetic) |
| d_drones export chain | MuJoCo core mass silently clamped at 1e-4; Lee auto-gains ignore inertia cross-coupling for asymmetric morphs |

---

## 7. How this feeds the generalist multi-task controller

**Reuse directly (proven):**
1. **Hex morph library + CMA-ES prior** (`hex_sampler.py`, `prior_controller.py`,
   `library.npz`) — per-morph analytical stabilization is the foundation; do
   not attempt multi-morph hover without it (no-prior all-morph collapses,
   exp_064).
2. **`TorchDroneGateEnv`** — the only training-speed-viable env; physics
   verified stable across branches.
3. **Hover-prior-only residual split** — prior for hover, free learning for
   trajectory tasks. This is the validated architectural answer to the prior's
   dual role (stabilizer for hover, straitjacket for racing).
4. **Cosine LR + per-task reward coefs** — the two committed training-recipe
   wins at 100M scale.
5. **v4 task suite conventions** (18–22) as the frozen evaluation protocol —
   the autoresearch metric depends on these staying fixed.

**Do not reuse:**
- Per-morph PPO / body+brain CMA-ES (`e_drones_ec`) — contradicts the
  generalist goal and is compute-prohibitive across 90 morphs.
- Per-task actor heads / separate trunks — tested (exp_068, −9.9); the shared
  trunk with per-task encoders + gated critics is critical.
- `18c_hover.py` reward profile — non-standard.

**Known open problems for the generalist:**
- **fig8 fragility**: fig8 collapses whenever a second task shares the
  high-velocity regime (vel=0.40) on the shared trunk. Candidate directions
  already queued in the strategy memo: fig8-prior restoration, intermediate
  vel coefs (0.36), ent_start 0.018.
- **Cross-task gradient interference** is the central scientific obstacle —
  literature directions (PCGrad/CAGrad-style gradient surgery, task-conditioned
  LR) are natural next reads for the autoresearch literature step.
- **No torch-vs-numpy dynamics equivalence test** exists; adding one would
  guard the physics core the whole stack depends on.

**Recommended immediate fixes (cheap, high value) — ALL APPLIED 2026-07-27:**
1. ✅ `test_residual_drone_env.py`: `LIBRARY` now resolves by walking upward
   (respects `$ARIEL_HEX_LIBRARY`); hover assertion skips cleanly when only
   dummy prior params are available.
2. ✅ `TorchDroneGateEnv`: `self.render_mode = render_mode` is always set,
   and `get_attr("render_mode")` now returns it — SB3 no longer warns.
3. ✅ `27_train_rl_hex_mtrl_v4.py`: prepended a LEGACY header noting it is
   superseded by `37_train_residual_mtrl.py`.
4. ✅ Added `tests/unit/test_simulation/test_tasks/test_torch_numpy_dynamics_equivalence.py`.
   **NEW FINDING**: the test immediately caught real drift — torch and
   numpy dynamics produce different `state_dot` for the standard quad
   (max abs diff ~0.62 in accelerations, ~1.0 relative on some rate
   channels). Marked `xfail(strict=True)` with an explanation so CI
   surfaces the discrepancy without failing the build; flipping the
   marker will convert to a hard test once the upstream discrepancy is
   fixed. Suspected source: aero-drag or motor-lag term encoding.
   This is a real bug in the physics core that the audit hypothesised
   might exist and the new test now proves does.
