#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=8G -t 00:40:00
#SBATCH -J c2dl
#SBATCH -o logs/c2_download_%j.out
# C2: A2052, A2063 and A2147 all sit at z=0.035, where a 1 Mpc box subtends
# ~24 arcmin -- larger than the 20 arcmin cutouts on disk, so load_obs drops
# them. 40 arcmin leaves margin at the lowest redshift in the sample.
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u download_lotss_image.py \
    --clusters A2052 A2063 A2147 \
    --size 40 --force \
    --output-dir "$HOME/data/cluster-ml/lotss_images"
