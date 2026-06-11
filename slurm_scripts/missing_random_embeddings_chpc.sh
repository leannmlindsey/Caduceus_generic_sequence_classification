#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --mem=60G
#SBATCH --job-name=caduceus_random
#SBATCH --output=caduceus_random_%j.out
#SBATCH --error=caduceus_random_%j.err
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#
# Fills the missing Caduceus v1 RANDOM embedding baseline for 4k and 8k (2k was
# already done). Runs the existing embedding analysis with --include_random_baseline.
# University of Utah CHPC (Notchpeak, soc-gpu-np), x86 + A100.
#
# Usage (env is inherited via --export=ALL, so activate it FIRST):
#   conda activate CADUCEUS_3
#   sbatch slurm_scripts/missing_random_embeddings_chpc.sh

REPO_DIR=/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/LAMBDA_REPLICATION/Caduceus_generic_sequence_classification
LAMBDA_V1=/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/LAMBDA_REPLICATION/LAMBDA_v1
CKPT_DIR=/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/LAMBDA_REPLICATION/Caduceus_generic_sequence_classification/checkpoints/caduceus-ps_seqlen-8k_d_model-256_n_layer-4_lr-8e-3
OUT=/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/LAMBDA_REPLICATION

export PYTHONNOUSERSITE=1

# fail fast if the transfer is incomplete or the path is wrong
[ -f "$CKPT_DIR/model_config.json" ]     || { echo "MISSING: $CKPT_DIR/model_config.json"; exit 1; }
[ -f "$CKPT_DIR/checkpoints/last.ckpt" ] || { echo "MISSING: $CKPT_DIR/checkpoints/last.ckpt"; exit 1; }

cd "$REPO_DIR"

echo "=== 4k ==="
python -m src.embedding_analysis --csv_dir="$LAMBDA_V1/train_val_test/4k" --checkpoint_path="$CKPT_DIR/checkpoints/last.ckpt" --config_path="$CKPT_DIR/model_config.json" --output_dir="$OUT/caduceus_random_4k" --batch_size=32 --max_length=4096 --pooling=mean --seed=42 --nn_epochs=100 --nn_hidden_dim=256 --nn_lr=0.001 --include_random_baseline

echo "=== 8k ==="
python -m src.embedding_analysis --csv_dir="$LAMBDA_V1/train_val_test/8k" --checkpoint_path="$CKPT_DIR/checkpoints/last.ckpt" --config_path="$CKPT_DIR/model_config.json" --output_dir="$OUT/caduceus_random_8k" --batch_size=32 --max_length=8192 --pooling=mean --seed=42 --nn_epochs=100 --nn_hidden_dim=256 --nn_lr=0.001 --include_random_baseline
