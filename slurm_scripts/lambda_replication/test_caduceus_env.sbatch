#!/bin/bash
# Tiny GPU job to run test_caduceus_env.py on a GH200 node WITHOUT needing an
# interactive allocation. Short walltime so it backfills ahead of a full queue.
#
# Submit once per candidate env (TEST_ENV = full env path):
#   sbatch --export=ALL,TEST_ENV=/u/llindsey1/miniconda3/envs/caduceus_env test_caduceus_env.sbatch
#   sbatch --export=ALL,TEST_ENV=/u/llindsey1/miniconda3/envs/CADUCEUS_3   test_caduceus_env.sbatch
# Then read the result:  cat cadtest_*_<jobid>.out
#
#SBATCH --job-name=cadtest
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=cadtest_%j.out
#SBATCH --error=cadtest_%j.out

set -uo pipefail

: "${TEST_ENV:?set TEST_ENV to the full conda env path via --export}"
TEST_PY="${TEST_PY:-/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/test_caduceus_env.py}"
CKPT_DIR="${CKPT_DIR:-/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/Caduceus_generic_sequence_classification/checkpoints/caduceus-ps_seqlen-8k_d_model-256_n_layer-4_lr-8e-3}"

echo "node: $(hostname)   env: ${TEST_ENV}"
source /u/llindsey1/miniconda3/etc/profile.d/conda.sh
conda activate "${TEST_ENV}"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
python "${TEST_PY}" --ckpt-dir "${CKPT_DIR}"
