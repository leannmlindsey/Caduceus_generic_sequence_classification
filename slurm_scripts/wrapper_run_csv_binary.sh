#!/bin/bash

# Wrapper script for running CSV binary classification with Caduceus on Biowulf
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_csv_binary.sh
#
# Or submit directly with environment variables:
#   sbatch --export=ALL,DATA_DIR=/path/to/data,PRETRAINED_PATH=/path/to/ckpt,CONFIG_PATH=/path/to/config run_csv_binary.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Dataset Configuration ===
# Path to directory containing train.csv, dev.csv, test.csv
export DATA_DIR="/path/to/your/csv/data"
# Name for this dataset (used in output directory structure)
export DATASET_NAME="my_dataset"

# === REQUIRED: Model Configuration ===
# Path to model config JSON
export CONFIG_PATH="/path/to/model_config.json"
# Path to pretrained model checkpoint
export PRETRAINED_PATH="/path/to/checkpoint.ckpt"

# === Model Type ===
# Options: hyena, mamba, caduceus
export MODEL="caduceus"
# Options: dna_embedding, dna_embedding_mamba, dna_embedding_caduceus
export MODEL_NAME="dna_embedding_caduceus"

# === Post-hoc RC (for Caduceus) ===
# Set to "true" for Caduceus post-hoc, "false" for others
export CONJOIN_TEST="true"
export CONJOIN_TRAIN_DECODER="false"

# === OPTIONAL: Hyperparameters ===
export LR="6e-4"
export BATCH_SIZE="32"
export MAX_LENGTH="1024"
export MAX_EPOCHS="100"
export D_OUTPUT="2"
export RC_AUG="false"
export SEED="2222"

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ "${DATA_DIR}" == "/path/to/your/csv/data" ]; then
    echo "ERROR: Please set DATA_DIR to your actual data directory"
    exit 1
fi

if [ "${CONFIG_PATH}" == "/path/to/model_config.json" ]; then
    echo "ERROR: Please set CONFIG_PATH to your model config file"
    exit 1
fi

if [ "${PRETRAINED_PATH}" == "/path/to/checkpoint.ckpt" ]; then
    echo "ERROR: Please set PRETRAINED_PATH to your checkpoint file"
    exit 1
fi

# Verify files exist
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: DATA_DIR does not exist: ${DATA_DIR}"
    exit 1
fi

if [ ! -f "${DATA_DIR}/train.csv" ]; then
    echo "ERROR: train.csv not found in ${DATA_DIR}"
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_PATH}"
    exit 1
fi

if [ ! -f "${PRETRAINED_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${PRETRAINED_PATH}"
    exit 1
fi

echo "=========================================="
echo "Submitting Caduceus CSV Binary Job"
echo "=========================================="
echo "Dataset: ${DATASET_NAME}"
echo "Data dir: ${DATA_DIR}"
echo "Model: ${MODEL}"
echo "Checkpoint: ${PRETRAINED_PATH}"
echo "LR: ${LR}, Batch: ${BATCH_SIZE}, Epochs: ${MAX_EPOCHS}"
echo "=========================================="

# Submit the job
sbatch --export=ALL run_csv_binary.sh

echo ""
echo "Job submitted. Monitor with: squeue -u \$USER"
