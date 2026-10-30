#!/usr/bin/env bash
# Sequential runner for the six paper implementations. Each writes to its
# own log and, on completion, prints a "[Pn] saved -> ..." line the
# Monitor uses to signal progress.
#
# Sizing: chosen to complete in a few hours total on CPU. Adjust --steps
# in individual sections for longer / more definitive runs.

set -uo pipefail
cd "$(dirname "$0")/../../../.."   # repo root

LOG_DIR="__data__/paper_impls/logs"
mkdir -p "$LOG_DIR"

log_step() {
    printf '\n===== %s starting (%s) =====\n' "$1" "$(date +%H:%M:%S)"
}

log_step "P1 song2021 progress"
uv run python examples/spear/library/paper_impls/p1_song2021_progress.py \
    --steps 500000 --num-envs 16 --n-steps 512 --max-steps 800 \
    --seeds 0 --modes telescoping progress \
    > "$LOG_DIR/p1.log" 2>&1
echo "[SEQ] P1 exit=$?"

log_step "P4 johannink2019 residual"
uv run python examples/spear/library/paper_impls/p4_johannink2019_residual.py \
    --steps 100000 --num-envs 8 --n-steps 512 --max-steps 800 \
    --alphas 0.05 0.1 0.2 0.4 0.8 nop --tasks hover figure8 --seeds 0 \
    > "$LOG_DIR/p4.log" 2>&1
echo "[SEQ] P4 exit=$?"

log_step "P3 molchanov2019 s2mr"
uv run python examples/spear/library/paper_impls/p3_molchanov2019_s2mr.py \
    --steps 400000 --num-envs 10 --n-steps 512 \
    --sigma-az-deg 2 --sigma-pitch-deg 10 \
    > "$LOG_DIR/p3.log" 2>&1
echo "[SEQ] P3 exit=$?"

log_step "P5 yu2020 pcgrad (paired)"
uv run python examples/spear/library/paper_impls/p5_yu2020_pcgrad.py \
    --pcgrad off --steps 200000 --num-envs 10 --n-steps 512 \
    > "$LOG_DIR/p5_off.log" 2>&1
echo "[SEQ] P5 off exit=$?"
uv run python examples/spear/library/paper_impls/p5_yu2020_pcgrad.py \
    --pcgrad on --steps 200000 --num-envs 10 --n-steps 512 \
    > "$LOG_DIR/p5_on.log" 2>&1
echo "[SEQ] P5 on exit=$?"

log_step "P6 sodhani2021 care (paired)"
uv run python examples/spear/library/paper_impls/p6_sodhani2021_care.py \
    --policy onehot --steps 200000 --num-envs 10 --n-steps 512 \
    > "$LOG_DIR/p6_onehot.log" 2>&1
echo "[SEQ] P6 onehot exit=$?"
uv run python examples/spear/library/paper_impls/p6_sodhani2021_care.py \
    --policy care --steps 200000 --num-envs 10 --n-steps 512 \
    > "$LOG_DIR/p6_care.log" 2>&1
echo "[SEQ] P6 care exit=$?"

log_step "P2 kaufmann2023 swift"
uv run python examples/spear/library/paper_impls/p2_kaufmann2023_swift.py \
    --mode multi --steps 500000 --num-envs 8 --n-steps 512 \
    --max-steps 800 --n-eval-tracks 5 \
    > "$LOG_DIR/p2.log" 2>&1
echo "[SEQ] P2 exit=$?"

echo "[SEQ] ALL DONE $(date +%H:%M:%S)"
