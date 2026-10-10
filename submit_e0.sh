#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J e0
#SBATCH --array=0-9
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/e0_%A_%a.out
# Review experiment E0 (CODEX_CLAUDE_REVIEW.md rounds 4-5): does training on
# single projections improve single-observation prediction?
#   X-ray (real count model), snapshots 99+91, shared persisted manifest
#   folds_e0.npz (sha256 recorded in each output), seeds 42-46,
#   checkpoint selection on individual single-view inner observations in
#   BOTH arms, same epochs/optimiser/updates per epoch:
#     0-4  pooled-view training     5-9  single-view training
# Extensions prepared, NOT part of this array (submit separately if wanted):
#   radio 3 Mpc: --dataset injected_fix_phys_b3000_s9991.h5
#   joint 1 Mpc: --dataset injected_fix_phys_b1000_s9991.h5 --xray-dataset xray_fix_s9991.h5
# Estimate: ~6-8 min per run on one GPU, ~1 h wall at the 2-GPU limit,
# <50 MB of prediction files (no checkpoints are saved).
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 5))
if [ $I -lt 5 ]; then V=pooled; else V=single; fi
./venv/bin/python -u train_cnn_mock.py --dataset xray_fix_s9991.h5 \
    --fold-manifest folds_e0.npz --train-view $V --select-view single \
    --seed $S --out-prefix cnn_preds/e0_x_${V}_s$S
