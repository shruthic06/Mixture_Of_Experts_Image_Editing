#!/bin/bash
#============================================================================
# Quick Test - SD1.5 MoE/Dense Training Components
# Tests model loading, forward pass, training step (no real data)
#============================================================================

#SBATCH --job-name=sd15-test-quick
#SBATCH --output=sd15-test-quick-%j.out
#SBATCH --error=sd15-test-quick-%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source ~/miniconda3/bin/activate sd-moe

export HF_HOME=/scratch/khasti/.cache/huggingface
export TORCH_HOME=/scratch/khasti/.cache/torch
mkdir -p $HF_HOME $TORCH_HOME

cd ~/moe-minimal

echo "============================================"
echo "SD1.5 MoE Quick Test"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

python validate_training.py --test_mode quick

echo ""
echo "============================================"
echo "Test completed at $(date)"
echo "============================================"

