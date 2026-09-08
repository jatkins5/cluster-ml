#!/bin/bash
#SBATCH --job-name=mocktrans
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/mock_transfer_%A_%a.out
#SBATCH --error=logs/mock_transfer_%A_%a.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

TAGS=(nh4 nh3 nocut)
T=${TAGS[$SLURM_ARRAY_TASK_ID]}
echo "=========== mock transfer, cut=$T ==========="
python train_cnn_mock.py --dataset "mock_dataset_${T}.h5" \
    --folds 5 --epochs 60 --out-prefix "mocktransfer_${T}"
