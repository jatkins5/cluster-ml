#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -J smoothsweep
#SBATCH -o logs/smooth_sweep_%j.out

cd /oscar/data/idellant/cluster-ml
for S in 0.0 1.0 1.5 2.0; do
    echo "================ sim_smooth_px = $S ================"
    ./venv/bin/python -u forward_model_lotss.py \
        --mask-compact --mask-max-beams 10 --invert-arcsinh \
        --sim-smooth-px "$S" \
        --out-prefix "forward_lotss_sinh_s${S}"
done
