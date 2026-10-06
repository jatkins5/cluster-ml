#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=48G -t 02:00:00
#SBATCH -J r144anl
#SBATCH -o logs/r144_analyze_%j.out
# After submit_144_rerun.sh and submit_144_seeds.sh: the paper's clean-
# simulation numbers at 144 MHz, the 5-seed snapshot table, real-cluster
# agreement, the A399/A401 check, and every figure.
cd /oscar/data/idellant/cluster-ml
PY="./venv/bin/python -u"
echo "################ scalars, terciles, CNN increment (144 MHz)"
$PY analyze_mass_control.py --dataset dataset_nh4_144_128.h5 \
    --preds "cnn_preds/pooled_nh4_144_inner_s*.npz"
echo "################ shape only vs full flux (144 MHz)"
$PY compare_shape.py --dataset dataset_nh4_144_128.h5 \
    --full "cnn_preds/pooled_nh4_144_inner_s*.npz" \
    --shape "cnn_preds/pooled_shape144_inner_s*.npz"
echo "################ snapshot table, 5 seeds"
$PY summarize_s91.py
echo "################ real radio vs X-ray, 5-seed ensembles"
$PY real_agreement_s91.py
echo "################ A399 / A401"
$PY check_a399_a401.py
echo "################ figures"
$PY paper/scripts/make_figures.py
