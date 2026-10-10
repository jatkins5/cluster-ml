#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=128G -t 16:00:00
#SBATCH -J fixbuild
#SBATCH -o logs/fix_build_%j.out
# Review fixes F1 + F5 (+M1), CODEX_CLAUDE_REVIEW.md.
#  Radio (F1): injected mocks from linear emission with a flux-conserving
#    rebin, power from the linear sum, one flux constant fixed on total
#    power; boxes 1/2/3 Mpc; the arcsinh-morphology sensitivity arm at 1 and
#    3 Mpc; background-only controls; both snapshots; 80' fields throughout.
#  X-ray (F5): the real processed-rate model (T_eq for focused photons,
#    livetime for particles, particle subtraction with its noise), L_X scale
#    0.358, target identity stored; both snapshots, on the radio redshifts.
#  Then realism of every radio set against LoTSS, X-ray mass-matched
#  brightness, and merges for training.
cd /oscar/data/idellant/cluster-ml
set -e
PY="./venv/bin/python -u"
LG="lotss_images_80/*.fits"
COMMON="--field-dir lotss_fields_80 --flux-mode sim"

build () {  # tag recipe box [extra]
  $PY build_injected_dataset.py --dataset dataset_nh4_144_512.h5 $COMMON --lotss-glob "$LG" \
      --recipe $2 --box-kpc $3 $4 --output injected_fix_${1}.h5
  if [ -z "$4" ]; then
    OFF=$($PY -c "import h5py; print(h5py.File('injected_fix_${1}.h5').attrs['sim_flux_offset_dex'])")
    $PY build_injected_dataset.py --dataset dataset_nh4_144_s91_512.h5 $COMMON --lotss-glob "$LG" \
        --recipe $2 --box-kpc $3 --sim-offset "$OFF" --seed 91 \
        --output injected_fix_${1}_s91.h5
    $PY merge_snapshots.py injected_fix_${1}.h5 injected_fix_${1}_s91.h5 \
        --output injected_fix_${1}_s9991.h5
  fi
}
build phys_b1000 physical 1000
build phys_b2000 physical 2000
build phys_b3000 physical 3000
build asinh_b1000 arcsinh 1000
build asinh_b3000 arcsinh 3000
build ctrl_b1000 physical 1000 --control
build ctrl_b3000 physical 3000 --control

$PY -c "
import h5py, numpy as np
z = [h5py.File(f'injected_fix_{t}_s9991.h5')['mock/z'][:] for t in
     ('phys_b1000', 'phys_b2000', 'phys_b3000', 'asinh_b1000', 'asinh_b3000')]
print('identical redshift draws across all radio sets:', all(np.array_equal(z[0], q) for q in z[1:]))"

echo "################ radio realism (F1)"
$PY check_radio_realism.py legacy_1Mpc_40arcmin=injected_nh4_144_simflux.h5 \
    phys_1Mpc=injected_fix_phys_b1000.h5 phys_2Mpc=injected_fix_phys_b2000.h5 \
    phys_3Mpc=injected_fix_phys_b3000.h5 arcsinh_1Mpc=injected_fix_asinh_b1000.h5 \
    arcsinh_3Mpc=injected_fix_asinh_b3000.h5 control_1Mpc=injected_fix_ctrl_b1000.h5 \
    control_3Mpc=injected_fix_ctrl_b3000.h5

echo "################ X-ray (F5)"
$PY build_xray_realistic.py build --count-model real --z-from injected_fix_phys_b1000.h5 \
    --lx-scale 0.358 --nh-csv lofar_nh.csv --output xray_fix.h5
A=$($PY -c "import h5py; print(h5py.File('xray_fix.h5').attrs['stretch_scale'])")
$PY build_xray_realistic.py build --count-model real --counts-h5 xray_counts_2Ms_128_s91.h5 \
    --z-from injected_fix_phys_b1000_s91.h5 --lx-scale 0.358 --nh-csv lofar_nh.csv \
    --stretch-scale "$A" --seed 91 --output xray_fix_s91.h5
$PY merge_snapshots.py xray_fix.h5 xray_fix_s91.h5 --output xray_fix_s9991.h5
$PY process_chandra.py --train-h5 xray_fix.h5 --output xray_obs_fix.h5
for X in xray_fix.h5 xray_fix_s9991.h5; do
  $PY attach_xray_obs.py --obs-h5 xray_obs_fix.h5 --train-h5 $X
done
$PY check_f5_fix.py
