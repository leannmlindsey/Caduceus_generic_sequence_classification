#!/bin/bash
# build_flashattn_delta.sbatch — build flash-attn into caduceus_env on Delta-AI
# (GH200 / aarch64 / sm_90), from source. flash-attn is REQUIRED by the Caduceus
# repo: src/models/sequence/long_conv_lm.py imports flash_attn.modules.{block,mha,
# mlp,embedding} and flash_attn.utils.generation.GenerationMixin unconditionally,
# so the model won't even import without it. (It was in the Biowulf caduceus_env.yml
# as flash-attn==2.5.6; that version won't build on torch 2.12, so we build current.)
#
# Heavy build: limit MAX_JOBS to bound RAM. No GPU needed to compile, but we grab
# one to validate the import chain after.
#
# Submit:  cd /work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION && sbatch /path/to/build_flashattn_delta.sbatch
#          tail -f buildflash_*.out
#
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=buildflash
#SBATCH --output=buildflash_%j.out
#SBATCH --error=buildflash_%j.out

set -uo pipefail

ENV=/work/hdd/bfzj/llindsey1/conda/envs/caduceus_env
PIP="${ENV}/bin/pip"
PY="${ENV}/bin/python"
REPO=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/Caduceus_generic_sequence_classification
# flash-attn version: empty = latest (best chance of supporting torch 2.12). Override
# with FA_VERSION=2.7.0.post2 etc. if latest's flash_attn.modules API has moved.
FA_SPEC="flash-attn${FA_VERSION:+==$FA_VERSION}"

export PIP_CACHE_DIR=/work/hdd/bfzj/llindsey1/.pip_cache
export CUDA_HOME="${ENV}"
export PATH="${ENV}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE   # don't grab a mismatched prebuilt wheel

step(){ echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }
die(){ echo "ERROR: $*" >&2; exit 1; }

echo "node: $(hostname)   start: $(date)"
[ -x "$PY" ] || die "caduceus_env python missing: $PY"

step "0  torch + nvcc sanity"
"$PY" -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda)" || die "torch import failed"
command -v nvcc && nvcc --version | tail -1 || die "nvcc not on PATH (need cuda-nvcc in the env)"

step "1  build $FA_SPEC (sm_90, from source, MAX_JOBS=$MAX_JOBS)"
"$PIP" install --no-cache-dir --no-build-isolation "$FA_SPEC" || die "flash-attn build FAILED — see compiler output above"

step "2  validate the flash_attn API the Caduceus code imports"
"$PY" - <<'PY' || die "flash_attn installed but the required API is missing (version mismatch)"
import flash_attn
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp
from flash_attn.modules.embedding import GPT2Embeddings
from flash_attn.utils.generation import GenerationMixin
print("  flash_attn", flash_attn.__version__, "— modules/mha/mlp/embedding/GenerationMixin OK")
PY

step "3  validate the FULL Caduceus model import chain (dna_embedding -> long_conv_lm)"
cd "$REPO"
PYTHONPATH="$PWD" "$PY" -c "import src.models.sequence.dna_embedding; print('  caduceus model import chain OK')" \
  || die "model import still fails — paste this log"

echo
echo "RESULT: flash-attn built + Caduceus model imports. Re-run run_lambda_training.sh."
echo "done: $(date)"
