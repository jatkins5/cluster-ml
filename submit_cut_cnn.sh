#!/bin/bash
#SBATCH --job-name=cutcnn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/cut_cnn_%A_%a.out
#SBATCH --error=logs/cut_cnn_%A_%a.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Does the density cut help the task the project actually cares about --
# predicting TSC from the images -- or only the LoTSS forward model? Pooled
# CNN, the best model, over 5 seeds per dataset; seeds are matched across
# datasets so the comparison can be paired.
DATASETS=(dataset_gp_128.h5 dataset_nh3_128.h5 dataset_nh4_128.h5)
SEEDS=(42 43 44 45 46)

I=$SLURM_ARRAY_TASK_ID
D=${DATASETS[$(( I % 3 ))]}
S=${SEEDS[$(( I / 3 ))]}

echo "=========== pooled  $D  seed=$S ==========="
python train_cnn_pooled.py --folds 5 --epochs 60 --batch-size 32 \
    --seed "$S" --pseudo-tsc --dataset "$D"
