#!/bin/bash
#SBATCH --job-name=caduceus_csv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=caduceus_csv_%j.out
#SBATCH --error=caduceus_csv_%j.err

# Biowulf batch script for Caduceus CSV binary classification
# Usage: sbatch run_csv_binary.sh
#
# Required environment variables (set via --export or edit below):
#   DATA_DIR: Path to directory containing train.csv, dev.csv, test.csv
#   PRETRAINED_PATH: Path to pretrained model checkpoint
#   CONFIG_PATH: Path to model config JSON
#
# Optional environment variables:
#   DATASET_NAME: Name for output directory (default: csv_dataset)
#   MODEL: Model type - hyena, mamba, caduceus (default: caduceus)
#   MODEL_NAME: Registry name (default: dna_embedding_caduceus)
#   LR: Learning rate (default: 6e-4)
#   BATCH_SIZE: Batch size (default: 32)
#   MAX_LENGTH: Max sequence length (default: 1024)
#   MAX_EPOCHS: Training epochs (default: 100)
#   D_OUTPUT: Number of classes (default: 2)
#   RC_AUG: Reverse complement augmentation (default: false)
#   SEED: Random seed (default: 2222)

echo "============================================================"
echo "Caduceus CSV Binary Classification"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda
module load cuda/12.8

# Activate conda environment
source activate caduceus_env

# Ignore user site-packages to avoid conflicts with ~/.local packages
export PYTHONNOUSERSITE=1

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""
echo "Python environment:"
which python
python --version
echo ""

# Set defaults for optional parameters
DATASET_NAME=${DATASET_NAME:-csv_dataset}
MODEL=${MODEL:-caduceus}
MODEL_NAME=${MODEL_NAME:-dna_embedding_caduceus}
LR=${LR:-6e-4}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-1024}
MAX_EPOCHS=${MAX_EPOCHS:-100}
D_OUTPUT=${D_OUTPUT:-2}
RC_AUG=${RC_AUG:-false}
SEED=${SEED:-2222}
CONJOIN_TRAIN_DECODER=${CONJOIN_TRAIN_DECODER:-false}
CONJOIN_TEST=${CONJOIN_TEST:-true}

# Validate required parameters
if [ -z "${DATA_DIR}" ]; then
    echo "ERROR: DATA_DIR is not set"
    exit 1
fi

if [ -z "${PRETRAINED_PATH}" ]; then
    echo "ERROR: PRETRAINED_PATH is not set"
    exit 1
fi

if [ -z "${CONFIG_PATH}" ]; then
    echo "ERROR: CONFIG_PATH is not set"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

# Add to PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Set output directory
DISPLAY_NAME="${MODEL}_lr-${LR}_batch_size-${BATCH_SIZE}_rc_aug-${RC_AUG}"
HYDRA_RUN_DIR="./outputs/downstream/csv_binary/${DATASET_NAME}/${DISPLAY_NAME}/seed-${SEED}"
mkdir -p "${HYDRA_RUN_DIR}"

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Model: ${MODEL}"
echo "  Model name: ${MODEL_NAME}"
echo "  Config path: ${CONFIG_PATH}"
echo "  Pretrained path: ${PRETRAINED_PATH}"
echo "  Dataset name: ${DATASET_NAME}"
echo "  Data dir: ${DATA_DIR}"
echo "  Learning rate: ${LR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "  Output classes: ${D_OUTPUT}"
echo "  RC augmentation: ${RC_AUG}"
echo "  Conjoin test: ${CONJOIN_TEST}"
echo "  Seed: ${SEED}"
echo "  Output dir: ${HYDRA_RUN_DIR}"
echo "============================================================"
echo ""

# Run training
python -m train \
  experiment=csv_binary \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
  dataset.data_dir="${DATA_DIR}" \
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

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Results saved to: ${HYDRA_RUN_DIR}"
echo "============================================================"
