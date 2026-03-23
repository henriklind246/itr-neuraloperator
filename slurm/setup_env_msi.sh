#!/bin/bash
# One-time MSI environment setup for no-tps-ihcp
# Usage: bash slurm/setup_env_msi.sh

set -euo pipefail

echo "=== Setting up MSI environment for no-tps-ihcp ==="

# Load conda
module load miniforge

# Create conda env with CUDA PyTorch
conda create -n ihcp python=3.11 -y
conda activate ihcp

# PyTorch with CUDA (adjust cuda version to match MSI's GPU driver)
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Remaining Python deps (not torch — already installed via conda)
pip install numpy==2.2.6 scipy==1.15.3 PyYAML==6.0.2
pip install hydra-core==1.3.2 hydra-optuna-sweeper==1.2.0 optuna==2.10.0

# Install project in editable mode (makes src/ and data/ importable)
cd "$HOME/no-tps-ihcp"
pip install -e .

# Create runs output directory
mkdir -p "$HOME/fno_runs"

echo "=== Environment ready. Activate with: conda activate ihcp ==="
echo "=== Verify with: python -c 'import torch; print(torch.cuda.is_available())' ==="
