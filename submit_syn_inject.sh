#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 16:00:00
#SBATCH -J syninject
#SBATCH -o logs/syn_inject_%j.out
# After submit_syn_cells.sh: the review-fix (F1) injected LoTSS sets rebuilt
# on the new radio model -- same recipes, boxes, 80' fields, seeds and flux
# convention as submit_fix_build.sh -- plus the merged clean set, a check that
# the placement redshifts are unchanged (so the X-ray mocks placed on them can
# be reused), the radio realism check against LoTSS, and the per-cell model
# comparison against the old nh4_144 cells.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
PY="./venv/bin/python -u"
LG="lotss_images_80/*.fits"
COMMON="--field-dir lotss_fields_80 --flux-mode sim"
$PY merge_snapshots.py dataset_syn_nh4_144_128.h5 dataset_syn_nh4_144_s91_128.h5 \
    --output dataset_syn_nh4_144_s9991_128.h5
build () {  # tag recipe box [extra]
  $PY build_injected_dataset.py --dataset dataset_syn_nh4_144_512.h5 $COMMON --lotss-glob "$LG" \
      --recipe $2 --box-kpc $3 ${4:-} --output injected_syn_${1}.h5
  if [ -z "${4:-}" ]; then
    OFF=$($PY -c "import h5py; print(h5py.File('injected_syn_${1}.h5').attrs['sim_flux_offset_dex'])")
    $PY build_injected_dataset.py --dataset dataset_syn_nh4_144_s91_512.h5 $COMMON --lotss-glob "$LG" \
        --recipe $2 --box-kpc $3 --sim-offset "$OFF" --seed 91 \
        --output injected_syn_${1}_s91.h5
    $PY merge_snapshots.py injected_syn_${1}.h5 injected_syn_${1}_s91.h5 \
        --output injected_syn_${1}_s9991.h5
  fi
}
build phys_b1000 physical 1000
build phys_b2000 physical 2000
build phys_b3000 physical 3000
build asinh_b1000 arcsinh 1000
build asinh_b3000 arcsinh 3000
build ctrl_b1000 physical 1000 --control
build ctrl_b3000 physical 3000 --control
echo "################ placement redshifts vs the old radio sets (X-ray reuse)"
$PY - <<'PYEOF'
import h5py, numpy as np
ok = True
for t in ("phys_b1000", "phys_b2000", "phys_b3000", "asinh_b1000", "asinh_b3000"):
    for suf in ("", "_s91", "_s9991"):
        a = h5py.File(f"injected_fix_{t}{suf}.h5")["mock/z"][:]
        b = h5py.File(f"injected_syn_{t}{suf}.h5")["mock/z"][:]
        same = a.shape == b.shape and np.array_equal(a, b)
        ok &= same
        print(f"  {t}{suf}: identical redshift draws {same}")
print("ALL PLACEMENTS IDENTICAL:", ok)
PYEOF
echo "################ radio realism vs LoTSS"
$PY check_radio_realism.py phys_1Mpc_old=injected_fix_phys_b1000.h5 phys_1Mpc=injected_syn_phys_b1000.h5 \
    phys_3Mpc_old=injected_fix_phys_b3000.h5 phys_3Mpc=injected_syn_phys_b3000.h5 \
    arcsinh_3Mpc=injected_syn_asinh_b3000.h5 control_1Mpc=injected_syn_ctrl_b1000.h5 \
    control_3Mpc=injected_syn_ctrl_b3000.h5
