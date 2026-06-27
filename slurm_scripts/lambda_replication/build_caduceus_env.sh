#!/bin/bash
# build_caduceus_env.sh — build the Caduceus conda env on Delta-AI (GH200/aarch64)
# as ONE logged batch job, then validate it. Compiles the Mamba CUDA kernels for
# sm_90. See requirements-delta.txt for the rationale + the three risk points.
#
# IMPORTANT (Delta): `conda activate` does NOT work in a non-interactive SLURM
# job — that is why the run jobs use the inherit model. We cannot inherit an env
# that does not exist yet, so this build NEVER calls `conda activate`. Instead it
# uses the base `conda` binary by full path for create/install (`conda install
# -p <prefix>`) and the new env's own `bin/pip` / `bin/python` for everything
# else. So it does not matter which env is active when you submit.
#
# Submit:
#   cd /work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION
#   sbatch /path/to/build_caduceus_env.sh
#   tail -f buildcad_*.out
#
#SBATCH --job-name=buildcad
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=buildcad_%j.out
#SBATCH --error=buildcad_%j.out

set -uo pipefail

CONDA_BIN="${CONDA_BIN:-/u/llindsey1/miniconda3/bin/conda}"
ENV_PREFIX="${ENV_PREFIX:-/work/hdd/bfzj/llindsey1/conda/envs/caduceus_env}"
TEST_PY="${TEST_PY:-/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/test_caduceus_env.py}"
CKPT_DIR="${CKPT_DIR:-/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/Caduceus_generic_sequence_classification/checkpoints/caduceus-ps_seqlen-8k_d_model-256_n_layer-4_lr-8e-3}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/work/hdd/bfzj/llindsey1/.pip_cache}"

PY="${ENV_PREFIX}/bin/python"
PIP="${ENV_PREFIX}/bin/pip"

step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }
die()  { echo "ERROR: $*" >&2; exit 1; }

echo "node: $(hostname)   start: $(date)"
echo "conda: ${CONDA_BIN}"
echo "env:   ${ENV_PREFIX}   (NO conda activate is used — full-path binaries only)"
[ -x "${CONDA_BIN}" ] || die "conda binary not found/executable: ${CONDA_BIN}"

step "1/8  create env (python 3.11)"
"${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.11 || die "conda create failed"
[ -x "${PY}" ] || die "env python missing after create: ${PY}"
echo "  $("${PY}" --version)"

# fail-fast for the install steps from here on
set -e

step "2/8  put env bin on PATH + set CUDA_HOME (no activate)"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export CUDA_HOME="${ENV_PREFIX}"
echo "  PATH head: ${ENV_PREFIX}/bin"
echo "  CUDA_HOME: ${CUDA_HOME}"

step "3/8  install torch (aarch64 CUDA wheel)"
"${PIP}" install --no-cache-dir torch
TORCH_CUDA="$("${PY}" -c 'import torch;print(torch.version.cuda or "")')"
"${PY}" -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"
[ -n "${TORCH_CUDA}" ] || die "torch has no CUDA — wrong wheel (aarch64 CPU build?)"

step "4/8  nvcc into the env, matching torch CUDA ${TORCH_CUDA} (conda install -p, no activate)"
"${CONDA_BIN}" install -y -p "${ENV_PREFIX}" -c nvidia "cuda-nvcc=${TORCH_CUDA}" "cuda-cudart-dev=${TORCH_CUDA}" \
  || "${CONDA_BIN}" install -y -p "${ENV_PREFIX}" -c nvidia cuda-nvcc cuda-cudart-dev \
  || die "could not install nvcc"
command -v nvcc && nvcc --version | tail -1 || die "nvcc not on PATH after install"

step "5/8  framework + utility deps"
"${PIP}" install --no-cache-dir \
  hydra-core==1.3.2 omegaconf==2.3.0 einops==0.7.0 pytorch-lightning==1.8.6 \
  transformers datasets scikit-learn numpy pandas matplotlib seaborn h5py \
  pyfaidx pysam biopython huggingface-hub timm rich tqdm wandb

step "6/8  build causal-conv1d (sm_90, from source)"
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS="${MAX_JOBS:-8}" \
  "${PIP}" install --no-cache-dir --no-build-isolation "causal-conv1d>=1.4.0"

step "7/8  build mamba-ssm (sm_90, from source)"
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS="${MAX_JOBS:-8}" \
  "${PIP}" install --no-cache-dir --no-build-isolation "mamba-ssm>=2.2.2"

set +e
step "8/8  validate"
"${PY}" -c "import torch, mamba_ssm, causal_conv1d, pytorch_lightning; print('  imports ok')" \
  || echo "  WARNING: an import failed — see above"
if [ -f "${TEST_PY}" ]; then
    "${PY}" "${TEST_PY}" --ckpt-dir "${CKPT_DIR}"
    rc=$?
    echo
    [ "${rc}" -eq 0 ] && echo "RESULT: env build + validation PASSED" \
                      || echo "RESULT: build done but ${rc} validation check(s) FAILED — see above"
else
    echo "  (test script not found at ${TEST_PY}; scp it over and run test_caduceus_env.sbatch)"
fi
echo "done: $(date)"
