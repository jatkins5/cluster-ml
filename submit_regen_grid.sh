#!/bin/bash
#SBATCH --job-name=regengrid
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=24G -t 04:00:00
#SBATCH --array=0-29
#SBATCH -o logs/regen_grid_%A_%a.out

cd /oscar/data/idellant/cluster-ml
TAGS=(gp nh3 nh4)          # gp = no phase cut
MODES=(arc lin)
SEEDS=(0 1 2 3 4)
I=$SLURM_ARRAY_TASK_ID
T=${TAGS[$(( I % 3 ))]}
M=${MODES[$(( (I / 3) % 2 ))]}
S=${SEEDS[$(( I / 6 ))]}
FLAG=""; [ "$M" = "lin" ] && FLAG="--invert-arcsinh"
echo "=========== $T $M seed=$S (correlated noise) ==========="
./venv/bin/python -u forward_model_lotss.py \
    --dataset "dataset_${T}_512.h5" --seed "$S" \
    --mask-compact --mask-max-beams 10 $FLAG \
    --out-prefix "regen_${T}_${M}_s${S}"
