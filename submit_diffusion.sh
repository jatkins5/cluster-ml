#!/bin/bash
#SBATCH --job-name=cluster-ml-diff
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/diff_%j.out
#SBATCH --error=logs/diff_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# build the diffusion tensor from RAW radio NPZ (tuned, invertible arcsinh)
python build_diffusion_data.py --img-size 64 --extent-r500 2.0

# unconditional DDPM + memorization/fidelity readout
python train_diffusion.py \
    --data diffusion_radio_64.h5 \
    --epochs 400 \
    --batch-size 64
