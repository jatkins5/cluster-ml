#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=96G -t 05:00:00
#SBATCH -J xdepth2b
#SBATCH -o logs/xdepth_fix2_build_%j.out
# Depth study, second correction (review R7): the hard-band particle error
# scales with livetime, and target/background/error draws are keyed per
# (cluster, realization) so every depth shares the same assignments. Five
# fixed Cycle-22-equivalent depths plus each mock at its target's real
# exposure, all built by this same code; snapshot 99.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
PY=./venv/bin/python
A=$($PY -c "import h5py; print(h5py.File('xray_fix.h5').attrs['stretch_scale'])")
COMMON="build --count-model real --z-from injected_fix_phys_b1000.h5 --lx-scale 0.358 --nh-csv lofar_nh.csv --stretch-scale $A"
for KS in 5 20 50 150 500; do
  $PY -u build_xray_realistic.py $COMMON --fixed-ks $KS --output xray_depthfix2_${KS}ks.h5
done
$PY -u build_xray_realistic.py $COMMON --output xray_depthfix2_real.h5
$PY - <<'PYEOF'
import h5py, numpy as np
fs = [f"xray_depthfix2_{k}.h5" for k in ("5ks", "20ks", "50ks", "150ks", "500ks", "real")]
ref = h5py.File(fs[0])["mock"]
for f in fs[1:]:
    g = h5py.File(f)["mock"]
    same_t = np.array_equal(g["target"][:], ref["target"][:])
    same_b = np.array_equal(g["bkg_index"][:], ref["bkg_index"][:])
    # error z-scores identical; magnitudes scale with sqrt(T_live,target/T_live)
    print(f"{f}: same targets {same_t}, same backgrounds {same_b}")
    assert same_t and same_b
print("assignments identical across all depth arms")
PYEOF
