#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Inference for the winning seed of VARIANT on one CSV. Reads
# <REPL_OUTPUT_DIR>/winners.json to find the winning finetune seed dir, resolves
# the Lightning checkpoint inside it, then runs the EXISTING `python -m src.inference`
# entry point (no model/experiment code is modified).
#
# Required env:
#   REPO_ROOT
#   REPL_OUTPUT_DIR
#   VARIANT
#   INPUT_CSV          path to the CSV to predict on
#   OUTPUT_FILENAME    name for the predictions CSV (e.g. test_predictions.csv)
#   MAX_LENGTH         max token length for this window
#   CONFIG_PATH        Caduceus model_config.json (local)
# Optional env:
#   BATCH_SIZE (32), THRESHOLD (0.5), D_OUTPUT (2), CONJOIN_TEST (false),
#   WINNER_CKPT_RELPATH (checkpoints/val/accuracy.ckpt), CONDA_ENV (caduceus_env)


echo "=== inference ${VARIANT}  input=${INPUT_CSV}  output=${OUTPUT_FILENAME} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load conda
module load cuda/12.8
source activate "${CONDA_ENV:-caduceus_env}"
echo "  conda env: ${CONDA_DEFAULT_ENV:-<none>}   python: $(command -v python || echo none)"
export PYTHONNOUSERSITE=1

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

BATCH_SIZE=${BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-2048}
THRESHOLD=${THRESHOLD:-0.5}
D_OUTPUT=${D_OUTPUT:-2}
CONJOIN_TEST=${CONJOIN_TEST:-false}
WINNER_CKPT_RELPATH=${WINNER_CKPT_RELPATH:-checkpoints/val/accuracy.ckpt}

if [ -z "${CONFIG_PATH:-}" ]; then echo "ERROR: CONFIG_PATH is not set"; exit 1; fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WINNERS_JSON="${REPL_OUTPUT_DIR}/winners.json"
if [ ! -f "${WINNERS_JSON}" ]; then
    echo "ERROR: ${WINNERS_JSON} not found (select_best_model must run first)"; exit 1
fi

# print_winner_exports.py emits shlex-quoted exports (WINNER_PATH=winning seed
# dir, WINNER_SEED) read from winners.json[VARIANT].
eval "$(python "${SCRIPT_DIR}/print_winner_exports.py" "${WINNERS_JSON}" "${VARIANT}")"

# Resolve the actual Lightning checkpoint inside the winning seed dir. Prefer the
# best-val checkpoint that the test phase used (cross_validation=true ->
# checkpoints/val/accuracy.ckpt); fall back to the always-saved last.ckpt.
CKPT="${WINNER_PATH}/${WINNER_CKPT_RELPATH}"
if [ ! -f "${CKPT}" ]; then
    echo "  NOTE: ${CKPT} not found — falling back to checkpoints/last.ckpt"
    CKPT="${WINNER_PATH}/checkpoints/last.ckpt"
fi
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: no checkpoint found under ${WINNER_PATH}/checkpoints/"; exit 1
fi

echo "  winner seed:   ${WINNER_SEED}"
echo "  winner dir:    ${WINNER_PATH}"
echo "  checkpoint:    ${CKPT}"
echo "  config:        ${CONFIG_PATH}"

OUTPUT_DIR="${REPL_OUTPUT_DIR}/inference/${VARIANT}"
mkdir -p "${OUTPUT_DIR}"

# Build conjoin flag.
CONJOIN_FLAG=""
if [ "${CONJOIN_TEST}" == "true" ]; then
    CONJOIN_FLAG="--conjoin_test"
fi

# src.inference writes <output_csv stem>_metrics.json next to the predictions CSV
# when --save_metrics and labels are present (e.g. test_predictions.csv ->
# test_predictions_metrics.json).
python -m src.inference \
    --input_csv="${INPUT_CSV}" \
    --checkpoint_path="${CKPT}" \
    --config_path="${CONFIG_PATH}" \
    --output_csv="${OUTPUT_DIR}/${OUTPUT_FILENAME}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --d_output=${D_OUTPUT} \
    --threshold=${THRESHOLD} \
    ${CONJOIN_FLAG} \
    --save_metrics

echo "Done: $(date)"
