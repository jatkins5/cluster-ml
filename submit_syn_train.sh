#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 05:00:00
#SBATCH -J syntrain
#SBATCH --array=0-74%6
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/syn_train_%A_%a.out
# Retrain every radio-dependent model in the paper on Chuiyang's regenerated
# radio model (dataset_syn_* / injected_syn_*), seeds 42-46, same protocols
# as the runs they replace (old prefix in brackets). X-ray-only models (fix_x,
# E0/E1X/E5, depth study) do not use the radio maps and are not rerun; the
# joint models reuse the X-ray mocks only because their placement redshifts
# were verified identical (submit_syn_inject.sh).
#    0-4   clean pooled CNN, snapshot 99            [pooled_nh4_144_inner]
#    5-9   clean pooled CNN, 99+91                  [pooled_s9991_inner]
#   10-14  clean shape-only (flux-normalised)       [pooled_shape144_inner]
#   15-19  clean, no density cut                    [pooled_nocut144_inner]
#   20-24  clean joint radio + X-ray                [joint144_clean_inner]
#   25-49  realistic radio: phys 1/2/3 Mpc, arcsinh 1/3 Mpc  [fix_*_sfimg]
#   50-54  mass only, fixed measurement (3 Mpc set) [r4_massonly]
#   55-59  radio 3 Mpc + mass, fixed measurement    [r4_imgmass_b3000]
#   60-74  realistic joint radio phys 1/2/3 Mpc + X-ray      [fix_*_joint]
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 5))
PY="./venv/bin/python -u"
out () { if [ -e "cnn_preds/$1" ]; then echo "refusing to overwrite $1"; exit 1; fi; }
if [ $I -lt 25 ]; then
  case $((I / 5)) in
    0) D=dataset_syn_nh4_144_128.h5; O=pooled_syn_inner ;;
    1) D=dataset_syn_nh4_144_s9991_128.h5; O=pooled_syn_s9991_inner ;;
    2) D=dataset_syn_nh4_144_128_shape.h5; O=pooled_shape_syn_inner ;;
    3) D=dataset_syn_nocut_144_128.h5; O=pooled_nocut_syn_inner ;;
    4) D=dataset_syn_nh4_144_128.h5; O=joint_syn_clean_inner ;;
  esac
  out ${O}_s$S.npz
  X=""; [ $((I / 5)) -eq 4 ] && X="--xray-dataset dataset_xray_128_scaled.h5"
  $PY train_cnn_pooled.py --dataset $D $X --pseudo-tsc --seed $S --select inner \
      --save-preds cnn_preds/${O}_s$S.npz
elif [ $I -lt 50 ]; then
  SETS=(phys_b1000 phys_b2000 phys_b3000 asinh_b1000 asinh_b3000); T=${SETS[$(((I - 25) / 5))]}
  out syn_${T}_sfimg_s${S}_preds.npz
  $PY train_cnn_mock.py --dataset injected_syn_${T}_s9991.h5 --seed $S \
      --out-prefix cnn_preds/syn_${T}_sfimg_s$S
elif [ $I -lt 60 ]; then
  D=injected_syn_phys_b3000_s9991.h5
  if [ $I -lt 55 ]; then O=syn_massonly; NI="--no-image"; else O=syn_imgmass_b3000; NI=""; fi
  out ${O}_s${S}_preds.npz
  $PY train_cnn_mock.py --dataset $D --seed $S --mass --mass-noise --mass-seed 0 $NI \
      --out-prefix cnn_preds/${O}_s$S
else
  SETS=(phys_b1000 phys_b2000 phys_b3000); T=${SETS[$(((I - 60) / 5))]}
  case $T in phys_b1000) X=xray_fix_s9991.h5 ;; phys_b2000) X=xray_fix_b2000_s9991.h5 ;;
             phys_b3000) X=xray_fix_b3000_s9991.h5 ;; esac
  out syn_${T}_joint_s${S}_preds.npz
  $PY train_cnn_mock.py --dataset injected_syn_${T}_s9991.h5 --xray-dataset $X --seed $S \
      --out-prefix cnn_preds/syn_${T}_joint_s$S
fi
