#!/bin/bash
#============================================================================
# Full Test - SD1.5 MoE Training with InstructPix2Pix Dataset
# Tests mini training loop with real data (10 samples, 5 steps)
#============================================================================

#SBATCH --job-name=sd15-test-full
#SBATCH --output=sd15-test-full-%j.out
#SBATCH --error=sd15-test-full-%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source ~/miniconda3/bin/activate sd-moe

export HF_HOME=/scratch/khasti/.cache/huggingface
export TORCH_HOME=/scratch/khasti/.cache/torch
mkdir -p $HF_HOME $TORCH_HOME

cd ~/moe-minimal

DATASET_PATH=/scratch/khasti/datasets/instructpix2pix

echo "============================================"
echo "SD1.5 MoE Full Test"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Dataset: $DATASET_PATH"
echo "============================================"

# Check dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

python validate_training.py --test_mode full --dataset_path $DATASET_PATH

echo ""
echo "============================================"
echo "Test completed at $(date)"
echo "============================================"

