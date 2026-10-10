#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 05:00:00
#SBATCH -J fixtrain
#SBATCH --array=0-59
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/fix_train_%A_%a.out
# Review fixes F1/F5 (+F3 single-view scores), snapshots 99+91, seeds 42-46.
#   0-24  radio: physical 1/2/3 Mpc (0-14), arcsinh arm 1/3 Mpc (15-24)
#  25-39  radio + noisy mass: physical 1/2/3 Mpc
#  40-44  X-ray, new count model
#  45-59  joint radio + X-ray: physical 1/2/3 Mpc with the new X-ray
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 5))
PY="./venv/bin/python -u"
X=xray_fix_s9991.h5
if [ $I -lt 25 ]; then
  SETS=(phys_b1000 phys_b2000 phys_b3000 asinh_b1000 asinh_b3000); T=${SETS[$((I / 5))]}
  $PY train_cnn_mock.py --dataset injected_fix_${T}_s9991.h5 --seed $S \
      --out-prefix cnn_preds/fix_${T}_sfimg_s$S
elif [ $I -lt 40 ]; then
  SETS=(phys_b1000 phys_b2000 phys_b3000); T=${SETS[$(((I - 25) / 5))]}
  $PY train_cnn_mock.py --dataset injected_fix_${T}_s9991.h5 --seed $S --mass --mass-noise \
      --out-prefix cnn_preds/fix_${T}_sfimgmassn_s$S
elif [ $I -lt 45 ]; then
  $PY train_cnn_mock.py --dataset $X --seed $S --out-prefix cnn_preds/fix_x_s$S
else
  SETS=(phys_b1000 phys_b2000 phys_b3000); T=${SETS[$(((I - 45) / 5))]}
  $PY train_cnn_mock.py --dataset injected_fix_${T}_s9991.h5 --xray-dataset $X \
      --seed $S --out-prefix cnn_preds/fix_${T}_joint_s$S
fi
