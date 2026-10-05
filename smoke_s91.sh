#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=48G -t 01:30:00
#SBATCH -J s91smoke
#SBATCH -o logs/s91_smoke_%j.out
# Two-halo run through every new snapshot-91 code path, into smoke_s91/.
cd /oscar/data/idellant/cluster-ml
set -e
PY="./venv/bin/python -u"
C=/oscar/data/idellant/Chuiyang
D=smoke_s91
rm -rf $D && mkdir -p $D
$PY -m py_compile build_dataset.py build_injected_dataset.py build_xray_realistic.py \
    xray_counts_regrid.py merge_snapshots.py train_cnn_pooled.py train_cnn_mock.py check_s91.py

$PY build_radio_cells.py --max-nh 1e-4 --nu-ghz 0.144 --w-scale 12.5 \
    --z 0.0994018026302219 --radio-dir $C/Radio_Snap91 \
    --cutout-dir $C/TNGCluster_Cutout/snap91 --out-dir $D/cells --limit 2
$PY build_dataset.py --snap 91 --groupcat $C/groupcat_classification/groupcat_091 \
    --halo-map $C/TNGCluster_Cutout/fof_halo_to_sub91.csv --cells-dir $D/cells \
    --img-size 128 --output $D/ds91.h5
$PY - <<'EOF'
import h5py, numpy as np
with h5py.File("smoke_s91/ds91.h5") as f:
    print(dict(f.attrs)["snapshot"], dict(f.attrs)["redshift"])
    for k in ("halo_id", "halo_id_snap", "r500c_kpc", "log_m500", "mass_ratio"):
        print(" ", k, f["meta/" + k][:])
    print("  pseudo_tsc", f["labels/pseudo_tsc"][:], "img max", f["images"][:].max())
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    cat = dict(zip(f["haloID"][:], zip(f["r500c"][:] * 1e3, f["mhalo_500c"][:])))
with h5py.File("smoke_s91/ds91.h5") as f:
    for h in f["meta/halo_id"][:]:
        print(f"  z=0 catalogue for {h}: r500 {cat[h][0]:.0f} kpc, logM {cat[h][1]:.3f}")
EOF
$PY merge_snapshots.py dataset_nh4_144_128.h5 $D/ds91.h5 --output $D/merged.h5
$PY - <<'EOF'
import h5py, numpy as np
from train_cnn_pooled import group_split
from sklearn.model_selection import KFold
with h5py.File("smoke_s91/merged.h5") as f:
    g, h, s = f["meta/group_id"][:], f["meta/halo_id"][:], f["meta/snap"][:]
print("merged rows", len(h), "unique ids", len(np.unique(h)), "snaps", np.unique(s, return_counts=True))
kf = KFold(5, shuffle=True, random_state=42)
old = list(KFold(5, shuffle=True, random_state=42).split(np.arange(352)))
new = list(group_split(g, np.arange(len(g)), kf.split))
same = all(np.array_equal(np.sort(a[a < 352]), np.sort(o[0])) for (a, b), (o) in zip(new, old))
print("snap-99 outer folds identical to the single-snapshot split:", same)
for tr, va in new:
    assert not set(g[tr]) & set(g[va]), "cluster in both train and val"
print("no cluster straddles a fold")
EOF
$PY xray_counts_regrid.py --ntasks 351 --task 0 --check --work-dir $D/xparts
ls $D/xparts
