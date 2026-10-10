#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J round4
#SBATCH --array=0-9
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/round4_%A_%a.out
# Review round 3: a MATCHED test of what the radio image adds beyond mass.
# Both arms: same halos/folds (3 Mpc physical set, snapshots 99+91), same
# training seeds 42-46, and the SAME blurred mass measurement for every
# simulated cluster (--mass-seed 0, fixed across seeds and arms).
#   0-4  mass only (images zeroed)   5-9  image + mass
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 5))
D=injected_fix_phys_b3000_s9991.h5
if [ $I -lt 5 ]; then
  ./venv/bin/python -u train_cnn_mock.py --dataset $D --seed $S --mass --mass-noise \
      --mass-seed 0 --no-image --out-prefix cnn_preds/r4_massonly_s$S
else
  ./venv/bin/python -u train_cnn_mock.py --dataset $D --seed $S --mass --mass-noise \
      --mass-seed 0 --out-prefix cnn_preds/r4_imgmass_b3000_s$S
fi
