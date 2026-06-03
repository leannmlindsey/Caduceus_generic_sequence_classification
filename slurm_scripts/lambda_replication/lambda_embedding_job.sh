#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Pretrained-embedding analysis (Surface D) for one (length, variant). Runs the
# EXISTING `python -m src.embedding_analysis` entry point on the PRETRAINED base
# checkpoint (not a finetuned checkpoint) — no model/experiment code is modified.
#
# Required env:
#   REPO_ROOT
#   REPL_OUTPUT_DIR    per-length replication output dir (outputs/<LEN>)
#   LAMBDA_DIR         train/val/test CSV directory (staged dev.csv alias)
#   VARIANT            caduceus
#   LEN                window label (2k/4k/8k)
#   MAX_LENGTH         max token length for this window
#   CONFIG_PATH        Caduceus model_config.json (local)
#   PRETRAINED_PATH    pretrained Caduceus checkpoint (local .ckpt)
# Optional env:
#   POOLING, EMB_SEED, NN_EPOCHS, NN_HIDDEN_DIM, NN_LR, EMB_BATCH_SIZE,
#   INCLUDE_RANDOM_BASELINE, CONDA_ENV (caduceus_env)


echo "=== embedding ${VARIANT} len=${LEN:-?} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load CUDA/12.8
source /data/lindseylm/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-caduceus_env}"
if [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV:-caduceus_env}" ]; then
    echo "ERROR: could not activate conda env '${CONDA_ENV:-caduceus_env}' (active: '${CONDA_DEFAULT_ENV:-none}'). Aborting." >&2
    exit 1
fi
echo "  conda env: ${CONDA_DEFAULT_ENV}   python: $(command -v python)"
export PYTHONNOUSERSITE=1

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

POOLING=${POOLING:-mean}
EMB_SEED=${EMB_SEED:-42}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}
EMB_BATCH_SIZE=${EMB_BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-2048}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}

if [ -z "${CONFIG_PATH:-}" ]; then echo "ERROR: CONFIG_PATH is not set"; exit 1; fi
if [ -z "${PRETRAINED_PATH:-}" ]; then echo "ERROR: PRETRAINED_PATH is not set"; exit 1; fi

# src.embedding_analysis reads {train,dev,test}.csv from --csv_dir. Stage a dir
# with a dev.csv alias for LAMBDA_v1's val.csv (no model code changed).
STAGE_DIR="${REPL_OUTPUT_DIR}/embedding/${VARIANT}/_data"
mkdir -p "${STAGE_DIR}"
ln -sf "${LAMBDA_DIR}/train.csv" "${STAGE_DIR}/train.csv"
ln -sf "${LAMBDA_DIR}/test.csv"  "${STAGE_DIR}/test.csv"
if [ -f "${LAMBDA_DIR}/dev.csv" ]; then
    ln -sf "${LAMBDA_DIR}/dev.csv" "${STAGE_DIR}/dev.csv"
elif [ -f "${LAMBDA_DIR}/val.csv" ]; then
    ln -sf "${LAMBDA_DIR}/val.csv" "${STAGE_DIR}/dev.csv"
else
    echo "ERROR: neither dev.csv nor val.csv in ${LAMBDA_DIR}"; exit 1
fi
CSV_DIR="${STAGE_DIR}"

OUTPUT_DIR="${REPL_OUTPUT_DIR}/embedding/${VARIANT}"
mkdir -p "${OUTPUT_DIR}"

echo "  pretrained:   ${PRETRAINED_PATH}"
echo "  config:       ${CONFIG_PATH}"
echo "  csv dir:      ${CSV_DIR}  (from ${LAMBDA_DIR})"
echo "  output:       ${OUTPUT_DIR}"
echo "  pooling=${POOLING}  max_length=${MAX_LENGTH}  nn_epochs=${NN_EPOCHS}  random_baseline=${INCLUDE_RANDOM_BASELINE}"

RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE}" == "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

python -m src.embedding_analysis \
    --csv_dir="${CSV_DIR}" \
    --checkpoint_path="${PRETRAINED_PATH}" \
    --config_path="${CONFIG_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --batch_size=${EMB_BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --pooling="${POOLING}" \
    --seed=${EMB_SEED} \
    --nn_epochs=${NN_EPOCHS} \
    --nn_hidden_dim=${NN_HIDDEN_DIM} \
    --nn_lr=${NN_LR} \
    ${RANDOM_BASELINE_FLAG}

echo "Done: $(date)"
