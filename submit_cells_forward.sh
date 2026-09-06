#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=24G -t 04:00:00
#SBATCH -J cellfwd
#SBATCH -o logs/cells_forward_%j.out

cd /oscar/data/idellant/cluster-ml

# With cells spread over their own volume the linear map should no longer be a
# delta function, so --invert-arcsinh becomes the physically correct choice
# rather than a regression. Test that claim against both controls.
echo "############ cells 512, linear (inverted) ############"
./venv/bin/python -u forward_model_lotss.py --dataset dataset_cells_512.h5 \
    --mask-compact --mask-max-beams 10 --invert-arcsinh \
    --out-prefix forward_cells_lin

echo "############ cells 512, compressed ############"
./venv/bin/python -u forward_model_lotss.py --dataset dataset_cells_512.h5 \
    --mask-compact --mask-max-beams 10 \
    --out-prefix forward_cells_arc

echo "############ point deposit 512, linear (control) ############"
./venv/bin/python -u forward_model_lotss.py --dataset dataset_gp_512.h5 \
    --mask-compact --mask-max-beams 10 --invert-arcsinh \
    --out-prefix forward_point_lin

echo "############ point deposit 512, compressed (control) ############"
./venv/bin/python -u forward_model_lotss.py --dataset dataset_gp_512.h5 \
    --mask-compact --mask-max-beams 10 \
    --out-prefix forward_point_arc
