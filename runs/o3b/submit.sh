#!/bin/bash

#SBATCH --job-name=niamh-ref
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --chdir=/work/nagarajan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --time=2-00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nagarajan@uni-potsdam.de

# cu126 torch bundles its own CUDA 12.6 runtime; no module load needed.
# Expose all pip-installed NVIDIA shared libs so cuequivariance can find libnvrtc.so.12 etc.
PROJECT_DIR="$HOME/research/niamh"
PYTHON="$PROJECT_DIR/.conda/bin/python"
NVIDIA_SITE="$PROJECT_DIR/.conda/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NVIDIA_SITE" -maxdepth 2 -name 'lib' -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"

# Redirect all model/compile caches to /work so nothing large lands in /home.
export MACE_CACHE_DIR=/work/nagarajan/cache/mace
export TORCH_HOME=/work/nagarajan/cache/torch
export TRITON_CACHE_DIR=/work/nagarajan/cache/triton
export HF_HOME=/work/nagarajan/cache/huggingface
mkdir -p "$MACE_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$HF_HOME"

cd "$PROJECT_DIR" || exit 1

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

"$PYTHON" -u runs/reference_materials/run_reference_analysis.py "$@"

echo "Finished: $(date)"
