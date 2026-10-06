#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J r144seed
#SBATCH --array=0-23
#SBATCH -o logs/r144_seed_%A_%a.out
# 144 MHz follow-ups:
#    0-4   shape-only pooled CNN (flux-normalised maps), seeds 42-46, paired
#          with pooled_nh4_144_inner_s42-46
#    5-20  seeds 45-46 for every realistic row of the snapshot table, so
#          each has 5 paired seeds (42-44 already run):
#            5-6 radio 99, 7-8 radio+mass 99, 9-10 X-ray 99, 11-12 joint 99,
#            13-14 radio 99+91, 15-16 radio+mass 99+91, 17-18 X-ray 99+91,
#            19-20 joint 99+91
#   21-23  mass-only model on 99+91 (seeds 42-44): reference for whether a
#          real cluster's predicted youth is just its mass
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
PY="./venv/bin/python -u"
R99=injected_nh4_144_simflux.h5
RB=injected_nh4_144_s9991_simflux.h5
X99=xray_placed_144.h5
XB=xray_placed_144_s9991.h5
if [ $I -lt 5 ]; then
  S=$((42 + I))
  $PY train_cnn_pooled.py --dataset dataset_nh4_144_128_shape.h5 --pseudo-tsc \
      --seed $S --select inner --save-preds cnn_preds/pooled_shape144_inner_s$S.npz
  exit
fi
if [ $I -ge 21 ]; then
  S=$((42 + I - 21))
  $PY train_cnn_mock.py --dataset $RB --seed $S --mass --no-image \
      --out-prefix cnn_preds/s9991_massonly_s$S
  exit
fi
J=$((I - 5)); S=$((45 + J % 2))
case $((J / 2)) in
  0) $PY train_cnn_mock.py --dataset $R99 --seed $S --out-prefix cnn_preds/nu144_sfimg_s$S ;;
  1) $PY train_cnn_mock.py --dataset $R99 --seed $S --mass --mass-noise \
         --out-prefix cnn_preds/nu144_sfimgmassn_s$S ;;
  2) $PY train_cnn_mock.py --dataset $X99 --seed $S --out-prefix cnn_preds/x144_s$S ;;
  3) $PY train_cnn_mock.py --dataset $R99 --xray-dataset $X99 --seed $S \
         --out-prefix cnn_preds/joint144_s$S ;;
  4) $PY train_cnn_mock.py --dataset $RB --seed $S --out-prefix cnn_preds/s9991_sfimg_s$S ;;
  5) $PY train_cnn_mock.py --dataset $RB --seed $S --mass --mass-noise \
         --out-prefix cnn_preds/s9991_sfimgmassn_s$S ;;
  6) $PY train_cnn_mock.py --dataset $XB --seed $S --out-prefix cnn_preds/s9991_x_s$S ;;
  7) $PY train_cnn_mock.py --dataset $RB --xray-dataset $XB --seed $S \
         --out-prefix cnn_preds/s9991_joint_s$S ;;
esac
