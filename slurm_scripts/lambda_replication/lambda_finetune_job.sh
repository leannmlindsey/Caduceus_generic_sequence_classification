#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 1 of Caduceus LAMBDA replication: finetune ONE (variant, seed).
# Submitted by run_lambda_training.sh. All paths/resources come via --export.
# This is the orchestration job body — it calls the EXISTING `python -m train`
# (Hydra) entry point with the same overrides as run_csv_binary.sh, only
# redirecting hydra.run.dir into the per-seed replication output dir. It does
# NOT modify any model/experiment code.
#
# Required env:
#   REPO_ROOT          repo root (holds train.py / configs/)
#   REPL_OUTPUT_DIR    per-length replication output dir (outputs/<LEN>)
#   LAMBDA_DIR         train/val/test CSV directory (LAMBDA_v1 train_val_test/<LEN>)
#   VARIANT            caduceus
#   SEED               integer
#   LEN                window label (2k/4k/8k)
#   MAX_LENGTH         max token length for this window (= window bp; char tokenizer)
#   BATCH_SIZE         per-device batch size for this window
#   CONFIG_PATH        Caduceus model_config.json (local)
#   PRETRAINED_PATH    pretrained Caduceus checkpoint (local .ckpt)
# Optional env (with defaults):
#   MODEL, MODEL_NAME, LR, MAX_EPOCHS, D_OUTPUT, RC_AUG, CONJOIN_TEST,
#   CONJOIN_TRAIN_DECODER, CONDA_ENV (caduceus_env)


echo "=== finetune ${VARIANT} seed=${SEED} len=${LEN:-?} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

# Activate conda (bare style — no set -e; conda activate under set -e silently
# kills SLURM jobs). Mirror setup_env.sh: PYTHONPATH=repo root, caduceus_env.
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

MODEL=${MODEL:-caduceus}
MODEL_NAME=${MODEL_NAME:-dna_embedding_caduceus}
LR=${LR:-1e-4}
MAX_EPOCHS=${MAX_EPOCHS:-10}
D_OUTPUT=${D_OUTPUT:-2}
RC_AUG=${RC_AUG:-false}
MAX_LENGTH=${MAX_LENGTH:-2048}
BATCH_SIZE=${BATCH_SIZE:-32}
CONJOIN_TEST=${CONJOIN_TEST:-false}
CONJOIN_TRAIN_DECODER=${CONJOIN_TRAIN_DECODER:-true}

if [ -z "${CONFIG_PATH:-}" ]; then echo "ERROR: CONFIG_PATH is not set"; exit 1; fi
if [ -z "${PRETRAINED_PATH:-}" ]; then echo "ERROR: PRETRAINED_PATH is not set"; exit 1; fi

# The csv_dataset loader reads {train,dev,test}.csv but LAMBDA_v1 ships val.csv.
# Stage a per-seed input dir that symlinks the CSVs and provides a dev.csv alias
# for val.csv. This avoids modifying the dataloader.
STAGE_DIR="${REPL_OUTPUT_DIR}/finetune/${VARIANT}/seed-${SEED}/_data"
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
DATASET_DIR="${STAGE_DIR}"

HYDRA_RUN_DIR="${REPL_OUTPUT_DIR}/finetune/${VARIANT}/seed-${SEED}"
mkdir -p "${HYDRA_RUN_DIR}"

echo "  pretrained:   ${PRETRAINED_PATH}"
echo "  config:       ${CONFIG_PATH}"
echo "  dataset dir:  ${DATASET_DIR}  (from ${LAMBDA_DIR})"
echo "  output:       ${HYDRA_RUN_DIR}"
echo "  lr=${LR}  batch=${BATCH_SIZE}  max_length=${MAX_LENGTH}  epochs=${MAX_EPOCHS}  d_output=${D_OUTPUT}"
echo "  conjoin_test=${CONJOIN_TEST}  conjoin_train_decoder=${CONJOIN_TRAIN_DECODER}  rc_aug=${RC_AUG}"

# `python -m train` runs trainer.fit then trainer.test (csv_binary sets
# train.test=true, cross_validation=true). The TestResultsCallback writes
# test_results.json (with key eval_mcc, computed via sklearn on the full test
# set) directly into hydra.run.dir, which select_best_model.py reads. No
# surfacing step is needed. Overrides below are verbatim from run_csv_binary.sh.
python -m train \
  experiment=csv_binary \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
  dataset.data_dir="${DATASET_DIR}" \
  dataset.max_length=${MAX_LENGTH} \
  dataset.d_output=${D_OUTPUT} \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.rc_aug="${RC_AUG}" \
  +dataset.conjoin_train=false \
  +dataset.conjoin_test="${CONJOIN_TEST}" \
  model="${MODEL}" \
  model._name_="${MODEL_NAME}" \
  +model.config_path="${CONFIG_PATH}" \
  +model.conjoin_test="${CONJOIN_TEST}" \
  +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
  +decoder.conjoin_test="${CONJOIN_TEST}" \
  optimizer.lr="${LR}" \
  trainer.max_epochs=${MAX_EPOCHS} \
  train.pretrained_model_path="${PRETRAINED_PATH}" \
  train.seed=${SEED} \
  wandb=null \
  hydra.run.dir="${HYDRA_RUN_DIR}"

if [ -f "${HYDRA_RUN_DIR}/test_results.json" ]; then
    echo "  wrote ${HYDRA_RUN_DIR}/test_results.json"
else
    echo "  WARNING: ${HYDRA_RUN_DIR}/test_results.json not found — training/test may have failed"
fi

echo "Done: $(date)"
