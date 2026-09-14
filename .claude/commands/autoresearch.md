You are running one iteration of the **ariel autoresearch loop** — an autonomous,
scientifically honest hill-climb on the generalist drone controller. Execute exactly one
loop action, then schedule the next wakeup.

**Core principles (non-negotiable):**
1. **Fair comparisons only.** Same seed, same step count, same eval protocol, same library,
   same held-out split for every A/B. Never compare 20M numbers to 100M numbers.
2. **One variable at a time.** A run tests exactly one hypothesis.
3. **The eval protocol is frozen.** Never modify `_eval_per_task`, the metric formula, the
   eval-steps count, or the held-out split as part of an experiment. Changing any of these
   invalidates all previous numbers and requires re-baselining everything (only do this if
   the user explicitly asks).
4. **Report what happened, not what you hoped.** Log regressions with the same detail as
   improvements. Never round in your favor. Treat sub-threshold deltas as noise, not wins.

---

## Fixed paths (use these verbatim — never use relative paths for uv run)

```
MAIN_REPO   = /home/user/Desktop/EvoDevo/ariel
WORKTREE    = /home/user/Desktop/EvoDevo/ariel/.claude/worktrees/autoresearch
LIBRARY     = /home/user/Desktop/EvoDevo/ariel/__data__/hex_library/v1/library.npz
LOG         = /home/user/Desktop/EvoDevo/ariel/examples/spear/library/autoresearch_log.md
STRATEGY    = /home/user/Desktop/EvoDevo/ariel/examples/spear/library/autoresearch_strategy.md
PROGRAM     = /home/user/Desktop/EvoDevo/ariel/examples/spear/library/autoresearch_program.md
RESEARCH    = /home/user/Desktop/EvoDevo/ariel/examples/spear/library/autoresearch_research.md
FILE_TRAIN  = {WORKTREE}/examples/spear/library/37_train_residual_mtrl.py
FILE_ENV    = {WORKTREE}/examples/spear/library/envs/residual_drone_env.py
```

**CRITICAL — always prefix every `uv run` with the explicit worktree `cd`:**
```bash
cd /home/user/Desktop/EvoDevo/ariel/.claude/worktrees/autoresearch && uv run ...
```
Never use a relative path. Never rely on CWD persistence. Create tmux sessions with
`-c {WORKTREE}` so the shell starts in the right directory.

**Files you may modify:** `FILE_TRAIN`, `FILE_ENV`, and (for literature-derived ideas)
NEW files under `{WORKTREE}/examples/spear/library/` (e.g. `40_train_xxx.py`).
**Files you must NEVER modify:** `prior_controller.py`, `hex_sampler.py`,
`test_prior_controller.py`, `gate_configs.py`, anything under `__data__/`.

---

## Two-phase experiment lifecycle

Full 100M-step runs take ~6-7h. Most failed hypotheses are already visibly bad at 20M
(~80-90 min). So every hypothesis goes through two phases:

```
PROPOSE → SCREEN (20M steps, ~90 min)
            ├─ screen metric < screen_baseline − 0.3  → revert, log SCREEN-FAIL, next idea
            └─ otherwise                              → CONFIRM (100M steps, ~6-7h)
                                                          ├─ Δ ≥ +0.15 → COMMIT (ratchet)
                                                          ├─ |Δ| < 0.15 → NEUTRAL (revert)
                                                          └─ Δ ≤ −0.15 → REVERTED
```

- **Screen baselines and confirm baselines are separate numbers.** Screen decisions use
  the 20M screen baseline; commit decisions use the 100M confirm baseline. Never mix.
- When a change is COMMITTED, its own screen metric becomes the new screen baseline
  (free — no extra run), and its confirm metric becomes the new confirm baseline.
- If no screen baseline exists for the current committed code (first run after adopting
  this protocol, or after a commit whose screen was skipped), launch a **SCREEN-BASELINE**
  run: no code change, 20M steps, log as `SCREEN-BASELINE`.
- The 0.3 screen margin deliberately tolerates slow-starting changes (e.g. LR schedules
  that only pay off late). The 0.15 commit threshold is the noise floor: deltas smaller
  than this are not evidence. Log them as `NEUTRAL` and revert — a real effect can be
  re-proposed and confirmed later; ratcheting on noise corrupts the baseline.

---

## Step 0 — Resume pending experiment (check FIRST on every wakeup)

```bash
cat /tmp/ar_pending 2>/dev/null && echo "EXISTS" || echo "NONE"
```

**Sentinel format** (line 1 = exp number, line 2 = phase, line 3+ = hypothesis):
```
NNN
SCREEN | CONFIRM | SCREEN-BASELINE
one-line hypothesis
```
**Legacy compatibility:** if line 2 is not one of the three phase keywords, treat the run
as `CONFIRM` (100M) with the hypothesis starting at line 2.

**If the sentinel exists**, check tmux:
```bash
tmux has-session -t ar_pending 2>&1; echo "TMUX_EXIT:$?"
```

- **TMUX_EXIT:0 (still running):** Do a cheap health check before sleeping:
  ```bash
  tail -c 4000 /tmp/autoresearch_s0.log | grep -ci "nan"
  ```
  If the training loss lines show `nan`, the run is dead weight — kill the session
  (`tmux kill-session -t ar_pending`), `rm /tmp/ar_pending`, log as `CRASHED`
  (metric −100), and continue at Step 8. Otherwise schedule and STOP:
  - SCREEN phase: `ScheduleWakeup(delaySeconds=2700, ...)` (~45 min; screens are short)
  - CONFIRM phase: `ScheduleWakeup(delaySeconds=3600, ...)`
  **STOP. Do not proceed further.**

- **TMUX_EXIT:1 (session gone = run finished):** `rm /tmp/ar_pending`, then go to
  **Step 6b** to evaluate. Route by phase: SCREEN/SCREEN-BASELINE → Step 6c,
  CONFIRM → Step 7. Then Steps 8 → 8a → 9 → 9b → 10.

**If no sentinel:** fresh iteration. Proceed to Step 1.

---

## Step 1 — Worktree isolation

```bash
git -C /home/user/Desktop/EvoDevo/ariel worktree list | grep -q "autoresearch" \
  || git -C /home/user/Desktop/EvoDevo/ariel worktree add \
       .claude/worktrees/autoresearch -b autoresearch/experiments
```

---

## Step 2 — Load state

Read in parallel (sqz_read_file for the Python files):

1. `LOG` — last `COMMITTED`/`BASELINE` row = confirm baseline (100M); last
   `SCREEN-BASELINE`/committed-screen value = screen baseline (20M)
2. `STRATEGY` — current direction and "do not retry" list
3. `RESEARCH` — literature-derived proposals not yet tried (if the file exists)
4. `FILE_TRAIN`, `FILE_ENV` — current code in the worktree

If no screen baseline exists for the current committed code, launch a SCREEN-BASELINE
run now (Step 6 with no code change, phase `SCREEN-BASELINE`) and STOP.

---

## Step 3 — Propose ONE change

Sources, in priority order:
1. `STRATEGY` "what to try next" list
2. `RESEARCH` proposals (literature-grounded, not yet tried)
3. Your own analysis of the per-task numbers in `LOG`

State the hypothesis explicitly, including the *mechanism* and the *expected per-task
effect*:
> *"I believe X will improve [task(s)] because [mechanism]. Expected: fig8 +2, slalom
> unchanged, others unchanged."*

Rules:
- ONE change only; never combine ideas
- Nothing from the "Do not retry" list, and nothing *adjacent* to a repeated failure
  (e.g. if vel=0.40 crashed twice, don't try 0.42)
- Prefer changes that help trajectory tasks (figure8, slalom, shuttle-run, circle)
- Modifying `FILE_TRAIN`/`FILE_ENV` is the default. A NEW training script (e.g.
  `40_train_xxx.py`) is allowed **only** for literature-derived architectural ideas that
  cannot be expressed as an edit — it must import and reuse the same env, eval protocol
  (`_eval_per_task` + `_format_eval` semantics), library, held-out split, and CLI
  contract, so its numbers are directly comparable.

---

## Step 4 — Apply the change

Use `Edit`/`Write` on `{WORKTREE}/...` absolute paths only.

---

## Step 5 — Maker-checker verification

```bash
git -C /home/user/Desktop/EvoDevo/ariel/.claude/worktrees/autoresearch diff
git -C /home/user/Desktop/EvoDevo/ariel/.claude/worktrees/autoresearch status --short
```

```
Agent(
  description="Verify autoresearch change",
  subagent_type="claude",
  prompt="""
You are a code verifier for the ariel autoresearch loop. Review the proposed change.

Hypothesis: {hypothesis}

Git diff (and untracked new files, if any — read them):
{diff_output}

Check ALL of the following:
1. Only allowed files are touched: 37_train_residual_mtrl.py, envs/residual_drone_env.py,
   or a NEW numbered script under examples/spear/library/. NEVER prior_controller.py,
   hex_sampler.py, test_prior_controller.py, gate_configs.py, or anything in __data__/.
2. Python syntax is valid.
3. The change matches the stated hypothesis — nothing extra snuck in.
4. Exactly ONE conceptual change (one variable).
5. No hyperparameter set to a pathological value (lr=0, gamma>=1.5, negative coefs...).
6. FAIR-COMPARISON INVARIANTS: the eval protocol (_eval_per_task logic, eval-steps,
   metric inputs), the held-out split, the library path, and the seed handling are
   untouched. For a NEW script: it must reuse the same env + eval + CLI contract.

Respond with exactly: PASS  or  FAIL: <one-line reason>
"""
)
```

- **FAIL** → revert (`git checkout -- <files>`, delete new untracked files), log as
  `REJECTED`, skip to Step 8.
- **PASS** → continue.

---

## Step 6 — Launch run (SCREEN = 20M, CONFIRM = 100M)

Write the sentinel (three lines — number, phase, hypothesis):

```bash
printf "%s\n%s\n%s\n" "NNN" "SCREEN" "{one-line hypothesis}" > /tmp/ar_pending
```

Kill any stale session, then launch (note `-c` sets the CWD; `STEPS` is 20000000 for
SCREEN/SCREEN-BASELINE, 100000000 for CONFIRM; `TRAIN_SCRIPT` is FILE_TRAIN unless the
experiment introduced a new script):

```bash
tmux kill-session -t ar_pending 2>/dev/null || true
tmux new-session -d -s ar_pending -c /home/user/Desktop/EvoDevo/ariel/.claude/worktrees/autoresearch
tmux send-keys -t ar_pending \
  "uv run examples/spear/library/37_train_residual_mtrl.py \
     --steps STEPS --num-envs 16 --inner-batch 4 \
     --eval-steps 3000 --device cuda --hover-prior-only \
     --library /home/user/Desktop/EvoDevo/ariel/__data__/hex_library/v1/library.npz \
     --seed 0 --out-dir /tmp/autoresearch_runs/exp_NNN_PHASE \
     > /tmp/autoresearch_s0.log 2>&1" Enter
```

Schedule the first status check and **STOP** (do not wait, do not read logs):
- SCREEN: `ScheduleWakeup(delaySeconds=2700, prompt="/autoresearch", reason="autoresearch — exp NNN screen (20M) running")`
- CONFIRM: `ScheduleWakeup(delaySeconds=3600, prompt="/autoresearch", reason="autoresearch — exp NNN confirm (100M) running")`

---

## Step 6b — Evaluate completed run

```bash
grep -c "saved →" /tmp/autoresearch_s0.log
```

- **Count = 0 (crashed):** extract the error
  (`grep -E "(Error|Traceback|Exception)" /tmp/autoresearch_s0.log | tail -3`),
  metric = −100, log as `CRASHED`, revert any code change, go to Step 8.
- **Count ≥ 1:** parse the `[after training]` block
  (`grep -A 8 "\[after training\]" /tmp/autoresearch_s0.log | tail -10`).
  Extract `reward/ep` per task; `nan` counts as −100.

  **Metric = (hover×1 + figure8×2 + slalom×2 + shuttle-run×2 + circle×2) / 9**
  (round to 3 d.p.). Always record all five per-task values, not just the scalar.

Route: SCREEN/SCREEN-BASELINE → Step 6c. CONFIRM → Step 7.

---

## Step 6c — Screen decision (20M-scale numbers ONLY)

- **SCREEN-BASELINE run:** log the metric as `SCREEN-BASELINE` (Step 8), then schedule
  the next iteration (Step 10) — the loop proposes a change on the next wakeup.
- **SCREEN run:**
  - `screen_metric ≥ screen_baseline − 0.3` → **promote**: keep the code change, write a
    new sentinel with phase `CONFIRM`, relaunch with `--steps 100000000` (Step 6),
    log the screen result as `SCREEN-PASS` (include 20M per-task values), and STOP after
    scheduling.
  - Otherwise → **fail fast**: revert the code change, log as `SCREEN-FAIL` with the 20M
    numbers (clearly marked as 20M — they must never be compared to 100M rows), continue
    to Step 8a → 9 → 9b → 10.

---

## Step 7 — Confirm decision / Ratchet (100M-scale numbers ONLY)

Compare confirm metric to the confirm baseline (last `COMMITTED`/`BASELINE` at 100M).

- **Δ ≥ +0.15 — COMMITTED:**
  ```bash
  cd /home/user/Desktop/EvoDevo/ariel/.claude/worktrees/autoresearch && \
    git add examples/spear/library/37_train_residual_mtrl.py \
            examples/spear/library/envs/residual_drone_env.py && \
    git add <any new experiment script> && \
    git commit -m "autoresearch: {summary} (metric: {baseline} → {avg} / +{delta})"
  ```
  Its screen metric becomes the new screen baseline; its confirm metric the new confirm
  baseline. `PushNotification(title="autoresearch: improvement", body="+{delta:.3f} → {avg:.3f} | {summary}")`
- **|Δ| < 0.15 — NEUTRAL:** revert. The effect is within noise; do not ratchet on it.
- **Δ ≤ −0.15 — REVERTED:** revert (`git checkout -- <files>`; delete new untracked files).

Note in the log entry *which tasks* moved and whether the screen predicted the confirm
outcome — this calibrates the screen margin over time.

---

## Step 8 — Update log

Append to `LOG` (use `Edit`). Every row MUST include all five per-task values:

```
| NNN | YYYY-MM-DD HH:MM | X.XXX | ±X.XXX | STATUS | Hypothesis: ... / Change: ... / hover=X fig8=X slalom=X shuttle=X circle=X / scale=20M|100M |
```

Status: `BASELINE`, `SCREEN-BASELINE`, `SCREEN-PASS`, `SCREEN-FAIL`, `COMMITTED`,
`NEUTRAL`, `REVERTED`, `CRASHED`, `REJECTED`

---

## Step 8a — Stuck detection

If the last 5 *hypothesis-testing* entries (ignore SCREEN-PASS/BASELINE rows) are all
failures (`SCREEN-FAIL`/`NEUTRAL`/`REVERTED`/`CRASHED`/`REJECTED`):
```
PushNotification(title="autoresearch: stuck", body="Last 5 all failed — running literature research")
```
Then trigger Step 9b (literature research) NOW, and make the next proposal come from a
**different change category** than the last 5 (categories: reward shaping, PPO hparams,
architecture, training schedule, env dynamics, literature-derived).

---

## Step 9 — Strategy synthesis (every 5 completed hypotheses)

Count non-BASELINE, non-SCREEN-PASS rows in LOG. If `count % 5 == 0`:

```
Agent(
  description="Update autoresearch strategy memo",
  prompt="""
Read /home/user/Desktop/EvoDevo/ariel/examples/spear/library/autoresearch_log.md.
Then rewrite /home/user/Desktop/EvoDevo/ariel/examples/spear/library/autoresearch_strategy.md:

## Current best metric        (confirm baseline AND screen baseline, labeled by scale)
## What has been tried        (grouped by outcome)
## What to try next           (3-5 specific ideas, prioritised by expected impact)
## Do not retry               (tried multiple times, always reverted — include the values tried)
## Observations               (patterns across experiments; which screens mispredicted confirms)

Under 45 lines total. Keep per-task numbers where they explain a decision.
"""
)
```

---

## Step 9b — Literature research (every 10 hypotheses, or when stuck)

If `count % 10 == 0` OR stuck detection fired, spawn a research agent:

```
Agent(
  description="Literature scan for autoresearch",
  prompt="""
You are a research assistant for an autonomous RL experiment loop training a multi-task
(hover / figure8 / slalom / shuttle-run / circle) PPO residual policy over 90 drone
morphologies with a shared-trunk multi-head architecture (SB3 PPO, per-task encoders +
critics, analytical hover prior on the hover task only).

Read these two files for current state and failure history:
- /home/user/Desktop/EvoDevo/ariel/examples/spear/library/autoresearch_strategy.md
- /home/user/Desktop/EvoDevo/ariel/examples/spear/library/autoresearch_log.md (skim recent rows)

Then use WebSearch/WebFetch to find 2-4 methods from the multi-task RL / PPO literature
that plausibly address the CURRENT bottleneck identified in the strategy memo (e.g.
cross-task gradient interference, task reward-scale imbalance, plateaued task).
Candidates worth checking if relevant: PopArt reward normalisation, gradient conflict
methods beyond PCGrad (CAGrad, GradNorm), task-weighting curricula, LR/entropy schedules,
distral-style distillation. Prefer methods implementable in <100 lines on SB3 PPO.

APPEND to /home/user/Desktop/EvoDevo/ariel/examples/spear/library/autoresearch_research.md
(create it if missing) one section per method:

### {method} ({paper, year, arXiv id})
- Mechanism: 1-2 lines
- Why it fits our bottleneck: 1-2 lines
- Implementation sketch: which file, which functions, ~how many lines
- Risk: what could break

Do NOT implement anything. Under 60 lines of additions. Only cite papers you actually
verified exist via search — never invent citations.
"""
)
```

Proposals from `RESEARCH` are tried through the exact same screen→confirm pipeline as
any other change — literature provenance earns no shortcut on evidence.

---

## Step 10 — Schedule next iteration

```
ScheduleWakeup(delaySeconds=60, prompt="/autoresearch", reason="autoresearch — next iteration")
```

**Do NOT ask the user anything. Do NOT stop. The user may be away.**

$ARGUMENTS
