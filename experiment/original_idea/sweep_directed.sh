#!/usr/bin/env bash
# Policy-capacity × morphology sweep for DIRECTED locomotion PPO.
#
# Directed-task counterpart of sweep_arch.sh. Question: as insect_n grows
# (more segments → more legs/actuators), how much policy width/depth is
# needed to learn target-seeking, and how does capacity change sample
# efficiency?
#
# Design choices (differences from sweep_arch.sh):
#   - Only the POLICY varies; the critic is fixed (CRITIC, default 256×3) so
#     small policies don't also get a failing value function.
#   - Full vision (FOV=360): an MLP has no memory, so a limited FOV adds a
#     failure mode no width/depth can fix. Heading gate stays on.
#   - Linear-policy baseline (depth 0) runs once per n (width is irrelevant
#     at depth 0), at the first width in WIDTHS.
#   - Resumable: runs whose output dir already has a non-empty
#     training_history.json are skipped. A failed run is logged and the
#     sweep continues.
#
# Default grid: n {1,2,4} × [widths {16,64,256} × depths {1,3} + linear]
#               × seeds {0,1,2} = 63 runs at 100M steps (~8-9 h, RTX 5070 Ti).
#
# Runs land in $OUTDIR/insect_n_n{N}_w{W}_d{D}_seed{S}-<timestamp>/ — the
# naming plot_sweep.py expects:
#   experiment/mujoco_playground/.venv/bin/python experiment/original_idea/plot_sweep.py \
#       --data-dir experiment/__data__/directed_capacity_sweep \
#       --out-dir  experiment/original_idea/__plots_directed__
#
# Usage:
#   bash experiment/original_idea/sweep_directed.sh
#   N_LIST="4" WIDTHS="16" DEPTHS="1" SEEDS="0" LINEAR_BASELINE=0 \
#       bash experiment/original_idea/sweep_directed.sh      # single pilot run

set -uo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXPERIMENT_DIR"

PY="$EXPERIMENT_DIR/mujoco_playground/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "error: expected mujoco_playground venv interpreter at $PY" >&2
    exit 1
fi
TRAIN_SCRIPT="rl/directed_locomotion_ppo_jax.py"

WIDTHS="${WIDTHS:-16 64 256}"
DEPTHS="${DEPTHS:-1 3}"
N_LIST="${N_LIST:-1 2 4}"
SEEDS="${SEEDS:-0 1 2}"
LINEAR_BASELINE="${LINEAR_BASELINE:-1}"
NUM_ENVS="${NUM_ENVS:-2048}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-100000000}"
NUM_EVALS="${NUM_EVALS:-20}"
CRITIC="${CRITIC:-256,256,256}"
FOV="${FOV:-360}"
OUTDIR="${OUTDIR:-$EXPERIMENT_DIR/__data__/directed_capacity_sweep}"
mkdir -p "$OUTDIR"
LOG_DIR="$OUTDIR/_logs"
mkdir -p "$LOG_DIR"

# Build the run list: "n w d s" tuples. Order: n (outer) → arch → seed, so a
# whole morphology's results become usable early.
first_width="${WIDTHS%% *}"
runs=()
for n in $N_LIST; do
    if [[ "$LINEAR_BASELINE" == "1" ]]; then
        for s in $SEEDS; do runs+=("$n $first_width 0 $s"); done
    fi
    for w in $WIDTHS; do
        for d in $DEPTHS; do
            for s in $SEEDS; do runs+=("$n $w $d $s"); done
        done
    done
done
total=${#runs[@]}

echo "sweep: directed capacity  n=[$N_LIST]  width=[$WIDTHS]  depth=[$DEPTHS] (+linear=$LINEAR_BASELINE)  seeds=[$SEEDS]"
echo "       critic=$CRITIC  fov=$FOV  num_envs=$NUM_ENVS  num_timesteps=$NUM_TIMESTEPS  num_evals=$NUM_EVALS"
echo "       out=$OUTDIR  ($total run(s))"
echo

done_run() {  # $1 = run-dir prefix; true if a finished run exists
    local d
    for d in "$OUTDIR"/"$1"-*/; do
        [[ -s "$d/training_history.json" ]] && return 0
    done
    return 1
}

i=0; failed=0; skipped=0
for r in "${runs[@]}"; do
    read -r n w d s <<< "$r"
    i=$((i + 1))
    tag="insect_n_n${n}_w${w}_d${d}_seed${s}"
    if done_run "$tag"; then
        echo "----- [$i/$total] $tag already done, skipping"
        skipped=$((skipped + 1))
        continue
    fi
    echo "===== [$i/$total] $tag  ($(date '+%H:%M:%S'))"
    "$PY" "$TRAIN_SCRIPT" \
        --morphology insect_n \
        --n-pairs "$n" \
        --hidden-width "$w" \
        --n-hidden-layers "$d" \
        --value-hidden-layer-sizes "$CRITIC" \
        --fov-deg "$FOV" \
        --seed "$s" \
        --num-timesteps "$NUM_TIMESTEPS" \
        --num-evals "$NUM_EVALS" \
        --num-envs "$NUM_ENVS" \
        --outdir "$OUTDIR" \
        > "$LOG_DIR/$tag.log" 2>&1
    rc=$?
    grep -E "^\[ *[0-9]+\]" "$LOG_DIR/$tag.log" | tail -1
    if [[ $rc -ne 0 ]]; then
        echo "!!!!! $tag FAILED (exit $rc) — see $LOG_DIR/$tag.log"
        failed=$((failed + 1))
    fi
done

echo
echo "sweep complete: $total runs, $skipped skipped, $failed failed. data: $OUTDIR"
