#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -J arcsinh_fix
#SBATCH -o logs/arcsinh_fix_%j.out

cd /oscar/data/idellant/cluster-ml
PY=./venv/bin/python

# Kept for reproducing the arcsinh-inversion experiment. The flag is now
# explicit: inversion was the default when this was written, and is not
# any more, so without --invert-arcsinh the labels below would be wrong.
# The conclusion was that inverting is a regression -- see make_mock.

echo "### brightness diagnostic, arcsinh inverted ###"
$PY -u check_mock_brightness.py --invert-arcsinh
echo

echo "### forward model, arcsinh inverted, no masking ###"
$PY -u forward_model_lotss.py --invert-arcsinh \
    --out-prefix forward_lotss_lin
echo

echo "### forward model, arcsinh inverted, obs-only mask max_beams=10 ###"
$PY -u forward_model_lotss.py --invert-arcsinh \
    --mask-compact --mask-max-beams 10 \
    --out-prefix forward_lotss_lin_mask10
