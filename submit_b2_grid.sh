#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 02:00:00
#SBATCH -J b2grid
#SBATCH -a 0-7%2
#SBATCH -o logs/b2_grid_%A_%a.out
# B2: does the image add anything once the model is handed halo mass?
# Four conditions x 2 seeds, matched so the comparison is paired:
#   0-1 image only        (the status quo: brightness is a mass proxy)
#   2-3 image + mass      (image must supply what mass does not explain)
#   4-5 image + blurred mass (mass carrying the real sample's 25% error)
#   6-7 mass only         (the baseline the increment is measured against)
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
SEED=$((42 + I % 2))
case $((I / 2)) in
  0) NAME=img;      FLAGS="" ;;
  1) NAME=imgmass;  FLAGS="--mass" ;;
  2) NAME=imgmassn; FLAGS="--mass --mass-noise" ;;
  3) NAME=massonly; FLAGS="--mass --no-image" ;;
esac
echo "condition=$NAME seed=$SEED flags='$FLAGS'"
./venv/bin/python -u train_cnn_mock.py \
    --dataset injected_nh4.h5 --seed "$SEED" $FLAGS \
    --out-prefix "cnn_preds/b2_${NAME}_s${SEED}"
