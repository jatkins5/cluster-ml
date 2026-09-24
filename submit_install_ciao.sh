#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=16G -t 04:00:00
#SBATCH -J ciao
#SBATCH -o logs/install_ciao_%j.out
# CIAO in its own conda env (not the project venv): needed for Chandra
# exposure maps (fluximage), which the vignetting and contamination
# correction of the real data depend on.
module load miniforge3/25.3.0-3-a6hh
# In the project directory, not ~/data (that is the shared group area).
PREFIX=/oscar/data/idellant/cluster-ml/ciao_env
conda create -y -p "$PREFIX" \
    -c https://cxc.cfa.harvard.edu/conda/ciao -c conda-forge \
    ciao ciao-contrib caldb_main 2>&1 | tail -5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PREFIX"
echo "CALDB=$CALDB"
ciaover
check_ciao_caldb 2>&1 | tail -3
