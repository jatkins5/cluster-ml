#!/bin/bash
#SBATCH --job-name=regentrans
#SBATCH --partition=gpu
#SBATCH -N 1 -n 1 -c 4 --mem=24G --gres=gpu:1 -t 04:00:00
#SBATCH --array=0-5
#SBATCH -o logs/regen_transfer_%A_%a.out
#SBATCH -e logs/regen_transfer_%A_%a.err
mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate
TAGS=(gp nh3 nh4)
SEEDS=(42 43)
I=$SLURM_ARRAY_TASK_ID
T=${TAGS[$(( I % 3 ))]}
S=${SEEDS[$(( I / 3 ))]}
echo "=========== transfer $T seed=$S ==========="
python train_cnn_mock.py --dataset "mockcn_${T}.h5" --seed "$S" \
    --folds 5 --epochs 60 --out-prefix "regentrans_${T}_s${S}"
