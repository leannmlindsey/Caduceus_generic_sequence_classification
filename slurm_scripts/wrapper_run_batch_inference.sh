#!/bin/bash

# Wrapper script for running batch inference with Caduceus on Biowulf
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_batch_inference.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input Files ===
# Path to text file containing one input CSV path per line
# Example contents of input_files.txt:
#   /path/to/dataset1.csv
#   /path/to/dataset2.csv
#   /path/to/dataset3.csv
INPUT_LIST="/path/to/input_files.txt"

# === REQUIRED: Output Directory ===
# All predictions and SLURM logs will be saved here
OUTPUT_DIR="/path/to/output_directory"

# === REQUIRED: Model Configuration ===
# Path to model config JSON
CONFIG_PATH="/path/to/model_config.json"
# Path to fine-tuned model checkpoint
CHECKPOINT_PATH="/path/to/checkpoint.ckpt"

# === OPTIONAL: Inference Parameters ===
BATCH_SIZE="32"
MAX_LENGTH="1024"
D_OUTPUT="2"
THRESHOLD="0.5"

# === OPTIONAL: Reverse Complement ===
# Set to "true" for Caduceus-Ph post-hoc conjoining, "false" otherwise
CONJOIN_TEST="true"

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ "${INPUT_LIST}" == "/path/to/input_files.txt" ]; then
    echo "ERROR: Please set INPUT_LIST to your input files list"
    exit 1
fi

if [ "${OUTPUT_DIR}" == "/path/to/output_directory" ]; then
    echo "ERROR: Please set OUTPUT_DIR to your output directory"
    exit 1
fi

if [ "${CONFIG_PATH}" == "/path/to/model_config.json" ]; then
    echo "ERROR: Please set CONFIG_PATH to your model config file"
    exit 1
fi

if [ "${CHECKPOINT_PATH}" == "/path/to/checkpoint.ckpt" ]; then
    echo "ERROR: Please set CHECKPOINT_PATH to your checkpoint file"
    exit 1
fi

# Verify files exist
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_PATH}"
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build conjoin flag
CONJOIN_FLAG=""
if [ "${CONJOIN_TEST}" != "true" ]; then
    CONJOIN_FLAG="--no_conjoin"
fi

echo "=========================================="
echo "Submitting Caduceus Batch Inference Jobs"
echo "=========================================="
echo "Input list: ${INPUT_LIST}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Config: ${CONFIG_PATH}"
echo ""
echo "Parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  D output: ${D_OUTPUT}"
echo "  Threshold: ${THRESHOLD}"
echo "  Conjoin test: ${CONJOIN_TEST}"
echo "=========================================="

# Call the batch submission script
"${SCRIPT_DIR}/submit_batch_inference.sh" \
    --input_list "${INPUT_LIST}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --config "${CONFIG_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --d_output "${D_OUTPUT}" \
    --threshold "${THRESHOLD}" \
    ${CONJOIN_FLAG}
