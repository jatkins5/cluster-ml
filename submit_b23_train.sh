#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 02:00:00
#SBATCH -J b23train
#SBATCH -a 0-8%2
#SBATCH -o logs/b23_train_%A_%a.out
# B2 option 3: same three conditions as the anchored run, on mocks whose
# total flux is the simulation's own prediction rather than the observed
# mass-power relation.
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
SEED=$((42 + I % 3))
case $((I / 3)) in
  0) NAME=sfimg;      FLAGS="" ;;
  1) NAME=sfimgmassn; FLAGS="--mass --mass-noise" ;;
  2) NAME=sfmassonly; FLAGS="--mass --no-image" ;;
esac
echo "condition=$NAME seed=$SEED flags='$FLAGS'"
./venv/bin/python -u train_cnn_mock.py \
    --dataset injected_nh4_simflux.h5 --seed "$SEED" $FLAGS \
    --out-prefix "cnn_preds/b2_${NAME}_s${SEED}"
