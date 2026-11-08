#!/bin/bash
#============================================================================
# SD1.5 Dense UNet Training for InstructPix2Pix (Baseline)
# ASU Sol Cluster - 2x A100 GPUs
#============================================================================

#SBATCH --job-name=sd15-dense-pix2pix
#SBATCH --output=sd15-dense-%j.out
#SBATCH --error=sd15-dense-%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# ============================================================================
# Configuration - Matching Official InstructPix2Pix
# ============================================================================
NUM_GPUS=2
BATCH_SIZE=4              # Per GPU (effective batch = 8)
RESOLUTION=256
EPOCHS=10
LR=5e-5                   # Diffusers default (or 1e-4 for original paper)
CONDITIONING_DROPOUT=0.05  # Critical for dual CFG training

DATASET_PATH=/scratch/khasti/datasets/instructpix2pix
OUTPUT_DIR=/scratch/khasti/sd15-dense-pix2pix/run-$(date +%Y%m%d-%H%M%S)

# ============================================================================
# Environment
# ============================================================================
source ~/miniconda3/bin/activate sd-moe

export HF_HOME=/scratch/khasti/.cache/huggingface
export TORCH_HOME=/scratch/khasti/.cache/torch
export TRANSFORMERS_CACHE=/scratch/khasti/.cache/huggingface
mkdir -p $HF_HOME $TORCH_HOME $OUTPUT_DIR

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

cd ~/moe-minimal

# ============================================================================
# Print Info
# ============================================================================
echo "============================================"
echo "SD1.5 Dense InstructPix2Pix Training (Baseline)"
echo "============================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $NUM_GPUS x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "Config:"
echo "  Dataset: $DATASET_PATH (FULL)"
echo "  Output: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE x $NUM_GPUS GPUs"
echo "  Resolution: $RESOLUTION"
echo "  Epochs: $EPOCHS"
echo "  Model: Dense (no MoE) - BASELINE"
echo "============================================"

# ============================================================================
# Training - FULL DATASET (Matching Official InstructPix2Pix)
# ============================================================================
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    train_sd15_dense_pix2pix.py \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --resolution $RESOLUTION \
    --learning_rate $LR \
    --conditioning_dropout_prob $CONDITIONING_DROPOUT \
    --mixed_precision \
    --log_every 100 \
    --save_every 1000 \
    --num_workers 4

echo ""
echo "============================================"
echo "Training completed at $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================"

