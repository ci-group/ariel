#!/usr/bin/env bash
set -euo pipefail

RUNS=10
POP=50
GEN=100
SCRIPT="src/ariel/body_phenotypes/lynx_mjspec/unified_pipeline/evolve.py"
BASE_OUT="__data__/lynx_mjspec/unified_batch_10x"

mkdir -p "${BASE_OUT}"

echo "Starting ${RUNS} runs (population=${POP}, generations=${GEN})"

for i in $(seq 1 "${RUNS}"); do
  run_id=$(printf "%02d" "${i}")
  ts=$(date +"%Y%m%d_%H%M%S")
  out_dir="${BASE_OUT}/run_${run_id}_${ts}"
  mkdir -p "${out_dir}"

  log_file="${out_dir}/train.log"

  echo "========================================"
  echo "Run ${run_id}/${RUNS} -> ${out_dir}"
  echo "========================================"

  uv run "${SCRIPT}" \
    --population "${POP}" \
    --generations "${GEN}" \
    --out-dir "${out_dir}" \
    | tee "${log_file}"

  # Optional: easy-to-find pointers to latest run outputs
  ln -sfn "$(basename "${out_dir}")" "${BASE_OUT}/latest_run"
done

echo "All runs complete. Artifacts under ${BASE_OUT}"