#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=16G -t 01:00:00
#SBATCH -J ciaofix
#SBATCH -o logs/fix_ciao_%j.out
module load miniforge3/25.3.0-3-a6hh
PREFIX=/oscar/data/idellant/cluster-ml/ciao_env
conda install -y -p "$PREFIX" -c https://cxc.cfa.harvard.edu/conda/ciao \
    -c conda-forge "numpy>=2.0,<2.4" 2>&1 | tail -25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PREFIX"
python -c "import numpy, pycrates; print('numpy', numpy.__version__, 'pycrates ok')"
ciaover | tail -3
check_ciao_caldb 2>&1 | tail -2
which fluximage
