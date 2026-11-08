#!/bin/bash
#============================================================================
# Resume SD1.5 + MoE Training (Matching Official InstructPix2Pix)
# ASU Sol Cluster - 2x A100 GPUs
#============================================================================

# Usage: sbatch resume_sd15_moe.slurm <checkpoint_path>
# Example: sbatch resume_sd15_moe.slurm /scratch/khasti/sd15-moe-pix2pix/run-xxx/latest_checkpoint.pt

#SBATCH --job-name=sd15-moe-resume
#SBATCH --output=sd15-moe-resume-%j.out
#SBATCH --error=sd15-moe-resume-%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# ============================================================================
# Configuration - Matching Official InstructPix2Pix
# ============================================================================
NUM_GPUS=2
BATCH_SIZE=32
RESOLUTION=256
EPOCHS=40              # Total epochs (will continue from checkpoint)
LR=5e-5                # Diffusers default (matching official)
NUM_EXPERTS=4
MOE_AUX_WEIGHT=0.01
CONDITIONING_DROPOUT=0.05  # Critical for dual CFG training

DATASET_PATH=/scratch/khasti/datasets/instructpix2pix

# Get checkpoint from argument
CHECKPOINT=${1:-""}
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: Please provide checkpoint path as argument"
    echo "Usage: sbatch resume_sd15_moe.slurm <checkpoint_path>"
    exit 1
fi

# Output to same directory as checkpoint
OUTPUT_DIR=$(dirname "$CHECKPOINT")

# ============================================================================
# Environment
# ============================================================================
source ~/miniconda3/bin/activate sd-moe

export HF_HOME=/scratch/khasti/.cache/huggingface
export TORCH_HOME=/scratch/khasti/.cache/torch
mkdir -p $HF_HOME $TORCH_HOME

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

cd ~/moe-minimal

# ============================================================================
# Print Info
# ============================================================================
echo "============================================"
echo "Resuming SD1.5 + MoE Training"
echo "============================================"
echo "Date: $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Total epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Conditioning dropout: $CONDITIONING_DROPOUT"
echo "Experts: $NUM_EXPERTS (top-2)"
echo "============================================"

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# ============================================================================
# Training - Resume (Matching Official InstructPix2Pix)
# ============================================================================
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_sd15_moe_pix2pix.py \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --resume_from $CHECKPOINT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --resolution $RESOLUTION \
    --learning_rate $LR \
    --conditioning_dropout_prob $CONDITIONING_DROPOUT \
    --num_experts $NUM_EXPERTS \
    --moe_aux_weight $MOE_AUX_WEIGHT \
    --mixed_precision \
    --log_every 100 \
    --save_every 1000 \
    --num_workers 4

echo ""
echo "============================================"
echo "Training completed at $(date)"
echo "============================================"

