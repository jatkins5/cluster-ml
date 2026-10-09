#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=8G -t 12:00:00
#SBATCH -J lotss80
#SBATCH -o logs/lotss80_%j.out
# Fix 1: the 1 Mpc radio box holds a median 4% of a cluster's shock
# emission (check_box_fraction.py); 3 Mpc holds 65%. At the lowest target
# redshift (0.035) 3 Mpc subtends ~72', so targets and background fields are
# re-downloaded at 80'. Same seed: the same sky positions as lotss_fields_40.
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u download_lotss_image.py --all --size 80 --force \
    --output-dir lotss_images_80
./venv/bin/python -u download_lotss_fields.py \
    --out-dir lotss_fields_80 --size-arcmin 80 --seed 0
ls lotss_images_80/*.fits | wc -l
ls lotss_fields_80/*.fits | wc -l
