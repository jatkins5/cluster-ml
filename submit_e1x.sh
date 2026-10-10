#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J e1x
#SBATCH --array=0-19%2
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/e1x_%A_%a.out
# E1 X-ray + optical structure (IDEALISED combined-input experiment; review
# round 8 contract). Four arms on the same cohort, manifest (folds_e0.npz),
# fixed noisy mass (--mass-seed 0), placement redshift, E0 protocol (pooled
# training, single-view selection), seeds 42-46; scored on single
# observations with 5-model ensembles (compare_e0.py-style driver):
#    0-4  controls                       images zeroed; mass + redshift
#    5-9  controls + structure           images zeroed; + per-view optical
#   10-14 X-ray + controls
#   15-19 X-ray + controls + structure   PRIMARY vs 10-14
# Optical features: FoF members, true r500, simulated hot-gas centre -- not
# an observational result. Prepared, NOT submitted.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 5))
ARM=$((I / 5))
COMMON="--dataset xray_fix_s9991.h5 --fold-manifest folds_e0.npz --train-view pooled
        --select-view single --mass --mass-noise --mass-seed 0 --redshift-scalar --seed $S"
VIEW="--view-scalars optical_features_idealised.csv --view-cols gap12,off_bcg_x,off_bcg_gal,asym"
case $ARM in
  0) ./venv/bin/python -u train_cnn_mock.py $COMMON --no-image --out-prefix cnn_preds/e1x_ctrl_s$S ;;
  1) ./venv/bin/python -u train_cnn_mock.py $COMMON --no-image $VIEW --out-prefix cnn_preds/e1x_ctrlopt_s$S ;;
  2) ./venv/bin/python -u train_cnn_mock.py $COMMON --out-prefix cnn_preds/e1x_xctrl_s$S ;;
  3) ./venv/bin/python -u train_cnn_mock.py $COMMON $VIEW --out-prefix cnn_preds/e1x_xctrlopt_s$S ;;
esac
