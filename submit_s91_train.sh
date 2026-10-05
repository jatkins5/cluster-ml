#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J s91train
#SBATCH --array=0-22
#SBATCH -o logs/s91_train_%A_%a.out
# Snapshots 99 + 91 against snapshot 99 alone, everything at 144 MHz.
# Folds group the two epochs of each cluster. Snapshot-99-only baselines
# already run: pooled_nh4_144_inner_s42-46, nu144_sfimg / nu144_sfimgmassn
# s42-44 (submit_nu144_train.sh); the X-ray and joint ones are rebuilt here
# because the X-ray set was re-placed on the 144 MHz radio redshifts.
#    0-4   clean pooled CNN, 99+91              seeds 42-46
#    5-7   transfer radio image, 99+91          seeds 42-44
#    8-10  transfer radio image + noisy mass    seeds 42-44
#   11-13  X-ray, 99 only                       seeds 42-44
#   14-16  X-ray, 99+91
#   17-19  joint radio + X-ray, 99 only
#   20-22  joint radio + X-ray, 99+91
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
PY="./venv/bin/python -u"
R99=injected_nh4_144_simflux.h5
RB=injected_nh4_144_s9991_simflux.h5
if [ $I -lt 5 ]; then
  S=$((42 + I))
  $PY train_cnn_pooled.py --dataset dataset_nh4_144_s9991_128.h5 --pseudo-tsc \
      --seed $S --select inner --save-preds cnn_preds/pooled_s9991_inner_s$S.npz
  exit
fi
J=$((I - 5)); S=$((42 + J % 3))
case $((J / 3)) in
  0) $PY train_cnn_mock.py --dataset $RB --seed $S --out-prefix cnn_preds/s9991_sfimg_s$S ;;
  1) $PY train_cnn_mock.py --dataset $RB --seed $S --mass --mass-noise \
         --out-prefix cnn_preds/s9991_sfimgmassn_s$S ;;
  2) $PY train_cnn_mock.py --dataset xray_placed_144.h5 --seed $S \
         --out-prefix cnn_preds/x144_s$S ;;
  3) $PY train_cnn_mock.py --dataset xray_placed_144_s9991.h5 --seed $S \
         --out-prefix cnn_preds/s9991_x_s$S ;;
  4) $PY train_cnn_mock.py --dataset $R99 --xray-dataset xray_placed_144.h5 \
         --seed $S --out-prefix cnn_preds/joint144_s$S ;;
  5) $PY train_cnn_mock.py --dataset $RB --xray-dataset xray_placed_144_s9991.h5 \
         --seed $S --out-prefix cnn_preds/s9991_joint_s$S ;;
esac
