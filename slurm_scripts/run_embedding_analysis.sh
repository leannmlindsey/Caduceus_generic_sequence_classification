#!/bin/bash
#SBATCH --job-name=caduceus_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=caduceus_emb_%j.out
#SBATCH --error=caduceus_emb_%j.err

# Biowulf batch script for Caduceus embedding analysis
# Usage: sbatch run_embedding_analysis.sh
#
# Required environment variables:
#   CSV_DIR: Path to directory containing train.csv, dev.csv, test.csv
#   CHECKPOINT_PATH: Path to Caduceus checkpoint (.ckpt file)
#   CONFIG_PATH: Path to Caduceus config JSON file

echo "============================================================"
echo "Caduceus Embedding Analysis"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda
module load cuda/12.8

# Activate conda environment
source activate caduceus_env

# Ignore user site-packages
export PYTHONNOUSERSITE=1

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Set defaults
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-1024}
POOLING=${POOLING:-mean}
SEED=${SEED:-42}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}

# Validate required parameters
if [ -z "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR is not set"
    exit 1
fi

if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: CHECKPOINT_PATH is not set"
    exit 1
fi

if [ -z "${CONFIG_PATH}" ]; then
    echo "ERROR: CONFIG_PATH is not set"
    exit 1
fi

# Navigate to repo root
# Use REPO_ROOT if provided (when called via sbatch), otherwise compute from BASH_SOURCE
if [ -n "${REPO_ROOT}" ]; then
    cd "${REPO_ROOT}" || exit
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "${SCRIPT_DIR}/.." || exit
fi
echo "Working directory: $(pwd)"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Set output directory
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/embedding_analysis/$(basename ${CSV_DIR})}
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Config: ${CONFIG_PATH}"
echo "  CSV dir: ${CSV_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Pooling: ${POOLING}"
echo "  Seed: ${SEED}"
echo "  NN epochs: ${NN_EPOCHS}"
echo "  NN hidden dim: ${NN_HIDDEN_DIM}"
echo "  NN learning rate: ${NN_LR}"
echo "  Include random baseline: ${INCLUDE_RANDOM_BASELINE}"
echo "============================================================"
echo ""

# Build random baseline flag
RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE}" == "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

# Run embedding analysis
python -m src.embedding_analysis \
    --csv_dir="${CSV_DIR}" \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --config_path="${CONFIG_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --pooling="${POOLING}" \
    --seed=${SEED} \
    --nn_epochs=${NN_EPOCHS} \
    --nn_hidden_dim=${NN_HIDDEN_DIM} \
    --nn_lr=${NN_LR} \
    ${RANDOM_BASELINE_FLAG}

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
