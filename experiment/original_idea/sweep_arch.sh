#!/usr/bin/env bash
# Architecture × morphology sweep for undirected locomotion PPO.
#
# Varies hidden-layer width (16/32/64), number of hidden layers (0/1/2/3), and
# the insect_n scaling parameter (1..4). Policy and value heads share the same
# (width, depth). 0 hidden layers ⇒ linear policy / value head.
#
# Default grid:  3 widths × 4 depths × 4 n_pairs × 3 seeds = 144 runs.
# At --num-timesteps 200M, each run ≈ 16-30 min on an RTX 5070 Ti (16 GiB),
# so the full sweep is ~2-3 days of wall-clock. Shrink SEEDS or N_LIST to
# iterate faster during development.
#
# Runs land in __data__/undirected_locomotion_ppo_jax/insect_n_n{N}_w{W}_d{D}_seed{S}-<timestamp>/
# and are consumed by experiment/original_idea/plot_sweep.py.
#
# Usage (from anywhere):
#   bash experiment/original_idea/sweep_arch.sh
#   WIDTHS="16 32"     bash experiment/original_idea/sweep_arch.sh
#   DEPTHS="0 2"       bash experiment/original_idea/sweep_arch.sh
#   N_LIST="2 3"       bash experiment/original_idea/sweep_arch.sh
#   SEEDS="0"          bash experiment/original_idea/sweep_arch.sh
#   NUM_TIMESTEPS=50000000 bash experiment/original_idea/sweep_arch.sh

set -euo pipefail

# Resolve script dir so relative paths work from any CWD. The training script
# lives one level up at experiment/rl/.
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXPERIMENT_DIR"

# Explicit interpreter from the nested mujoco_playground venv.
PY="$EXPERIMENT_DIR/mujoco_playground/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "error: expected mujoco_playground venv interpreter at $PY" >&2
    exit 1
fi

TRAIN_SCRIPT="rl/undirected_locomotion_ppo_jax.py"

# Overridable sweep parameters.
WIDTHS="${WIDTHS:-16 32 64}"
DEPTHS="${DEPTHS:-0 1 2 3}"
N_LIST="${N_LIST:-1 2 3 4}"
SEEDS="${SEEDS:-0 1 2}"
NUM_ENVS="${NUM_ENVS:-3072}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-200000000}"

# Count total runs for the progress banner.
total=0
for w in $WIDTHS; do
    for d in $DEPTHS; do
        for n in $N_LIST; do
            for s in $SEEDS; do
                total=$((total + 1))
            done
        done
    done
done

echo "sweep: original_idea  width=[$WIDTHS]  depth=[$DEPTHS]  n_pairs=[$N_LIST]  seeds=[$SEEDS]"
echo "       --num-envs=$NUM_ENVS  --num-timesteps=$NUM_TIMESTEPS  (${total} run(s))"
echo

# Loop order: width (outer) → depth → n → seed. Smaller nets run first so
# config problems surface quickly; within a (width, depth), all n and seeds
# finish together.
i=0
for w in $WIDTHS; do
    for d in $DEPTHS; do
        for n in $N_LIST; do
            for s in $SEEDS; do
                i=$((i + 1))
                echo "===== [$i/$total] width=$w depth=$d insect_n=$n seed=$s ====="
                "$PY" "$TRAIN_SCRIPT" \
                    --morphology insect_n \
                    --n-pairs "$n" \
                    --hidden-width "$w" \
                    --n-hidden-layers "$d" \
                    --seed "$s" \
                    --num-timesteps "$NUM_TIMESTEPS" \
                    --num-envs "$NUM_ENVS"
                echo
            done
        done
    done
done

echo "sweep complete. runs in: __data__/undirected_locomotion_ppo_jax/insect_n_n*_w*_d*_seed*"
