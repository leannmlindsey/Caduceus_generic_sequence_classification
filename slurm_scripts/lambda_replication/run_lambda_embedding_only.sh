#!/bin/bash
# run_lambda_embedding_only.sh — re-submit ONLY the Caduceus embedding (Surface D:
# LP + NN + random baseline) jobs, WITHOUT redoing the already-complete
# diagnostics / PHROG / genome-wide. Mirrors the embedding-submission block of
# run_lambda_inference.sh exactly (same conf, same exports, same job script).
#
# Use after the ReduceLROnPlateau(verbose=) torch-2.x fix in src/embedding_analysis.py.
# Run from the (caduceus_env) LOGIN prompt so the jobs inherit the env:
#   conda activate /work/hdd/bfzj/llindsey1/conda/envs/caduceus_env
#   bash slurm_scripts/lambda_replication/run_lambda_embedding_only.sh

set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"
[ -f "${CONFIG}" ] || { echo "ERROR: missing ${CONFIG}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG}"

if [[ "${LAMBDA_BASE}" == /path/to/* ]] || [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — LAMBDA_BASE/OUTPUT_DIR still placeholder"; exit 1
fi

LOGDIR="${OUTPUT_DIR}/logs"; mkdir -p "${LOGDIR}"
EMB_FLAGS=(--account=bfzj-dtai-gh --partition=ghx4 --gpus-per-node=1 \
           --mem="${EMB_MEM:-64g}" --time="${EMB_TIME:-04:00:00}" --cpus-per-task=8)
EMB_ENV_BASE="REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},CONFIG_PATH=${CONFIG_PATH},PRETRAINED_PATH=${PRETRAINED_PATH},POOLING=${POOLING},EMB_SEED=${EMB_SEED},NN_EPOCHS=${NN_EPOCHS},NN_HIDDEN_DIM=${NN_HIDDEN_DIM},NN_LR=${NN_LR},EMB_BATCH_SIZE=${EMB_BATCH_SIZE},INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}"

echo "Embedding-only re-run  OUTPUT_DIR=${OUTPUT_DIR}  windows='${SEGMENT_LENGTHS}'  variants='${VARIANTS}'  random=${INCLUDE_RANDOM_BASELINE:-false}"
cd "${REPO_ROOT}"
N=0
for LEN in ${SEGMENT_LENGTHS}; do
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    LAMBDA_DIR="${LAMBDA_BASE}/train_val_test/${LEN}"
    ml_var="MAX_LENGTH_${LEN}"; MAX_LENGTH="${!ml_var:-2048}"
    for VARIANT in ${VARIANTS}; do
        EMB_JOB="emb_${LEN}_${VARIANT}"
        echo "  submitting ${EMB_JOB}..."
        sbatch \
            --job-name="${EMB_JOB}" \
            --output="${LOGDIR}/${EMB_JOB}_%j.out" \
            --error="${LOGDIR}/${EMB_JOB}_%j.err" \
            "${EMB_FLAGS[@]}" \
            --export="ALL,REPL_OUTPUT_DIR=${REPL_LEN_DIR},LAMBDA_DIR=${LAMBDA_DIR},${EMB_ENV_BASE},VARIANT=${VARIANT},LEN=${LEN},MAX_LENGTH=${MAX_LENGTH}" \
            "${SCRIPT_DIR}/lambda_embedding_job.sh"
        N=$((N+1))
    done
done
echo "Submitted ${N} embedding job(s). Watch: squeue -u \$USER | grep emb_"
echo "Verify env+random: grep -E 'conda env|random_baseline' ${LOGDIR}/emb_*_*.out | tail"
