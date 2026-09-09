#!/bin/bash
#SBATCH --job-name=injtrans
#SBATCH --partition=gpu
#SBATCH -N 1 -n 1 -c 4 --mem=24G --gres=gpu:1 -t 04:00:00
#SBATCH --array=0-1
#SBATCH -o logs/inject_transfer_%A_%a.out
#SBATCH -e logs/inject_transfer_%A_%a.err
mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate
SEEDS=(42 43)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "=========== field-injected transfer, seed=$S ==========="
python train_cnn_mock.py --dataset injected_nh4.h5 --seed "$S" \
    --folds 5 --epochs 60 --out-prefix "injtrans_s${S}"
