# paper_impls — reference implementations for the drone MTRL plan

Each file is a self-contained runnable experiment for one paper in
`../PAPER_IMPL_PLAN.md`. Files wrap or subclass the existing training code
(`../37_train_residual_mtrl.py`, `../envs/residual_drone_env.py`,
`../42*.py`); none of the base scripts are modified.

Run everything with `uv run`; every script accepts `--smoke` for a short
CPU sanity budget (few minutes) and `--steps N` for full budgets. All output
goes under `__data__/paper_impls/<paper>/<timestamp>/` with `config.json`
and `results.json` (per-task rewards + standard metric).

| Paper | File | Status | Result |
|---|---|---|---|
| P1 · Song 2021 (progress reward) | `p1_song2021_progress.py` | done | 500k paired: lateral dev **−68%** (1.87 → 0.59 m), completion 0/7 both — gate PASS |
| P2 · Kaufmann 2023 (Swift track curriculum) | `p2_kaufmann2023_swift.py` | done | 80k, 5 held-out tracks: 4% mean completion, 0.58 wp/s (needs more steps) |
| P3 · Molchanov 2019 (morph/dynamics randomization) | `p3_molchanov2019_s2mr.py` | done | 400k on 10 randomized morphs: metric −5.57; follow-up = `40_morph_break_analysis.py sweep` |
| P4 · Johannink 2019 (residual α ablation) | `p4_johannink2019_residual.py` | done | 100k × 6 α cells: hover wins @ α=0.10 (r=+5.95), figure8 best @ α=0.40 (r=−11.6) — validates repo defaults |
| P5 · Yu 2020 (PCGrad gradient surgery) | `p5_yu2020_pcgrad.py` | done | 200k paired: baseline **+8.03** vs PCGrad **+6.99** (worse). Conflict frac stayed 0.44–0.50; paper's 20M-step escalation criterion not met at 200k |
| P6 · Sodhani 2021 (CARE context attention) | `p6_sodhani2021_care.py` | done | 200k paired: onehot **+8.03** vs CARE **+4.36** (worse). Learned attention near-uniform → encoders unspecialized at this budget |

## Quick start

```bash
# P1 — paired reward comparison on a figure-8 track
uv run examples/spear/library/42a_draw_trajectory.py --demo figure8
uv run examples/spear/library/paper_impls/p1_song2021_progress.py --smoke

# P4 — α ablation on canonical hex, hover + figure8
uv run examples/spear/library/paper_impls/p4_johannink2019_residual.py --smoke

# P3 — morph-randomized training, then reuse the break sweep
uv run examples/spear/library/paper_impls/p3_molchanov2019_s2mr.py --smoke

# P5, P6 — MTRL policy variants against 37's baseline
uv run examples/spear/library/paper_impls/p5_yu2020_pcgrad.py --smoke
uv run examples/spear/library/paper_impls/p6_sodhani2021_care.py --smoke

# P2 — track curriculum + held-out track eval
uv run examples/spear/library/paper_impls/p2_kaufmann2023_swift.py --smoke
```

## Standard metric

`metric = (hover + 2·(figure8 + slalom + shuttle-run + circle)) / 9`,
computed from per-task mean episode reward. See `common.standard_metric`.

## Editing this table

When a paper's gate experiment finishes, update the row above with the
one-line result (e.g. "progress > telescoping +18% waypoints, −42% lat_dev")
and set the Status column to `done`. Don't move on until the current entry
is updated.
