#!/bin/bash
#SBATCH --job-name=fwdseeds
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=24G -t 03:00:00
#SBATCH --array=0-19
#SBATCH -o logs/fwd_seeds_%A_%a.out

cd /oscar/data/idellant/cluster-ml

# The nh3-vs-nh4 forward-model gap (+0.390 vs +0.365) came from single runs.
# The mock realization is random -- sim/obs pairing, P150 flux scatter, noise
# -- so that gap needs an error bar before it can justify a choice.
TAGS=(nh3 nh4)
MODES=(arc lin)
SEEDS=(0 1 2 3 4)

I=$SLURM_ARRAY_TASK_ID
T=${TAGS[$(( I % 2 ))]}
M=${MODES[$(( (I / 2) % 2 ))]}
S=${SEEDS[$(( I / 4 ))]}

FLAG=""
[ "$M" = "lin" ] && FLAG="--invert-arcsinh"

echo "=========== $T $M seed=$S ==========="
./venv/bin/python -u forward_model_lotss.py \
    --dataset "dataset_${T}_512.h5" --seed "$S" \
    --mask-compact --mask-max-beams 10 $FLAG \
    --out-prefix "fwdseed_${T}_${M}_s${S}"
