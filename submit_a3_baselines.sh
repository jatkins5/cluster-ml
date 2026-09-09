#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=32G -t 01:00:00
#SBATCH -J a3base
#SBATCH -a 0-9%2
#SBATCH -o logs/a3_base_%A_%a.out
# A3: re-run the headline pooled-CNN baseline under honest checkpoint
# selection. 5 seeds x {inner-split selection, final epoch}, so the run also
# measures what the two protocols cost relative to each other. The number
# these replace is 0.567 +- 0.018 (nh4, best-epoch-on-the-test-fold).
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
SEED=$((42 + I % 5))
if [ $I -lt 5 ]; then SEL=inner; else SEL=final; fi
echo "seed=$SEED select=$SEL"
./venv/bin/python -u train_cnn_pooled.py \
    --dataset dataset_nh4_128.h5 --pseudo-tsc --seed "$SEED" \
    --select "$SEL" --save-preds "cnn_preds/pooled_nh4_${SEL}_s${SEED}.npz"
