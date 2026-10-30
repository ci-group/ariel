## Current best metric
- Confirm baseline (100M, exp_080): **32.374** — hover=64.5 fig8=24.0 slalom=26.8 shuttle=28.1 circle=34.5
- Screen baseline (20M, exp_080): **14.260** — hover=58.7 fig8=18.5 slalom=5.4 shuttle=5.4 circle=5.5
- Config: hover+fig8 prior (α=0.45, k_tilt×0.3 for fig8); vel fig8=0.40 shuttle=0.20 others=0.32; cosine LR 3e-4→3e-5; ent_start=0.02; gamma=0.99; gae_lambda=0.97; n_steps=1024.

## What has been tried (grouped by outcome)
**Committed wins (cumulative +27.0 from original 5.33 baseline):**
random_init=False for slalom/shuttle/circle (+7.1, exp_044-046); hover-prior (+3.6, exp_065); vel coef 0.03→0.28 (+17.4, exp_006-066); cosine LR (+0.6, exp_069); per-task vel decoupling (+0.7, exp_071); fig8-prior (+2.3, exp_080).

**Screen-passed, confirm-failed (screen was wrong direction or underestimated magnitude):**
clip_range_vf=0.2 (screen −0.15, confirm −1.4); per-task advantage whitening / TOPPO (screen +0.27 with |res|=0.74-0.82, confirm −5.81 with |res|=0.999 saturation — slow-burn instability invisible at 20M).

**Screen-failed (correctly killed):**
quadratic gate penalty c=0.10 (exp_078, hover 57→-6.7); SiLU activation (exp_083, hover 57→42.5, |res|→0.85); n_steps=2048 (exp_084, near-miss −0.261); PopArt per-task normalisation (exp_085, broken GAE bootstrap); gate_reward ×2 (exp_086, |res| inflation); cosine entropy annealing (exp_088, near-miss −0.055).

**Catastrophic confirms (never re-run at 100M):**
circle-prior (exp_081, −5.844, circle 34.5→19.7); gamma=0.999 (exp_082, slalom 26.8→1.97); per-task actor heads (exp_068, −9.9).

## What to try next (prioritised)
1. **clip_range 0.2→0.15 (exp_076, aborted)** — killed at 10.9M with no data; prior 250k screens showed seed-1 slalom 1.893 (best-ever at that scale); tighter clipping may reduce multi-task gradient variance.
2. **n_steps=2048 at 100M directly** — exp_084 missed screen by 0.055 purely because 20M halves update count; at 100M the better-quality GAE per rollout should dominate. Run as direct 100M confirm without a screen.
3. **Per-task gamma: gamma_slalom=0.995 only** — gamma=0.999 collapsed slalom globally; a modest horizon extension isolated to slalom (leaving all others at 0.99) may avoid the GAE variance blowup seen in exp_082.
4. **ent_start=0.025 (exp_052, cancelled before completing)** — s0=19.181 looked fine before halt; the near-monotone improvement up to 0.02 suggests 0.025 could add marginal late exploration; cheap screen candidate.
5. **Slalom vel coef 0.32→0.36** — slalom and fig8 cannot both sit at 0.40 (exp_072 killed fig8); 0.36 is a cautious mid-step. Screen first.

## Do not retry
- **gamma ≥ 0.999** (exp_082): slalom 26.8→2.0 at 100M. Tried: 0.999.
- **gae_lambda 0.98, 0.99** (exp_027, 057): slalom collapses via same long-horizon variance mechanism.
- **Circle-prior** (exp_081): circle 34.5→19.7 at 100M; prior fights tight banking.
- **SiLU / any unbounded activation** (exp_083): |res| inflation overwhelms analytical prior.
- **Per-task advantage whitening** (exp_087): |res|→0.999 saturation over 100M; screen was false positive.
- **vf_coef** (tried 0.3, 0.5✓, 0.6, 0.75): 0.5 is optimal.
- **ent_start** (tried 0.015, 0.02✓, 0.04): 0.02 is the narrow sweet spot.
- **TASK_ALPHA > 0.40 for slalom/circle** (exp_039, 040, 056): destabilises shared trunk.
- **n_epochs > 10** (exp_031); **clip_range_vf=0.2** (exp_079); **per-task actor heads** (exp_068).

## Observations
- **|res| drift at screen signals saturation at confirm**: exp_087 |res|=0.74-0.82 at 20M → |res|=0.999 at 100M. Treat |res| > 0.80 at screen as a WARNING even if metric passes.
- **Slalom is the canary for long-horizon credit-assignment changes**: gamma=0.999, gae_lambda=0.99, n_steps=512 all collapsed slalom first (−10 to −25 points) while other tasks held.
- **Screen mispredicts tasks needing cosine LR runway**: fig8 and circle appear weaker at 20M; a slalom drop that looks like "20M underestimation" can be a true early warning (exp_082: screen slalom=−3.8 → confirm slalom=−24.8).
- **Cross-task vel interference via shared trunk**: raising any task's vel coef ≥ 0.40 tends to depress fig8 and/or slalom (exp_072, 074). Per-task decoupling helps but headroom is ~0.04 units per task.
- **Prior scaffold**: load-bearing for hover across 90 morphs; strictly beneficial for fig8 (stable arc scaffold); catastrophic for circle (fights tight banking). No other task benefits from prior.
