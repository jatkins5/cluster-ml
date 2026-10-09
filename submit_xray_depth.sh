#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=32G -t 02:00:00
#SBATCH -J xdepth
#SBATCH --array=0-14
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/xdepth_%A_%a.out
# X-ray-only model vs exposure: 5 depths x 3 seeds, snapshot 99, mocks
# placed at the radio mocks' redshifts with every correction.
cd /oscar/data/idellant/cluster-ml
DEPTHS=(5 20 50 150 500)
I=$SLURM_ARRAY_TASK_ID
KS=${DEPTHS[$((I / 3))]}
S=$((42 + I % 3))
./venv/bin/python -u train_cnn_mock.py --dataset xray_depth_${KS}ks.h5 \
    --seed $S --out-prefix cnn_preds/xdepth_${KS}ks_s$S
