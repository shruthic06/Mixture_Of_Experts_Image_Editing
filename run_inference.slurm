#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=inference-%j.out
#SBATCH --error=inference-%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --partition=public

# Use EXPLICIT Python path (bypasses system Python)
PYTHON=~/miniconda3/envs/sd-moe/bin/python

echo "Using: $PYTHON"
$PYTHON --version
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

cd ~/moe-minimal

$PYTHON inference_sd15.py \
    --checkpoint /scratch/khasti/sd15-moe-pix2pix/run-20251201-023626/checkpoint_step_252000.pt \
    --input_image test_input.jpg \
    --prompt "add a tree" \
    --output test_output.jpg

echo "Done!"
