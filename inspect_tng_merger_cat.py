"""Schema dump for the TNG-Cluster truth merger catalog and the per-snapshot
target-halo catalogs, to design a tree-free merger proxy we can validate.
"""
import os

import h5py
import numpy as np

TARGET = os.path.expanduser(
    "~/data/Chuiyang/targethalo_cat_TNGCluster/targethalo_cat_099/"
    "TargetHalo_MergerCat_099.hdf5")
TRUTH = "TSC_Cutimages/cluster_mergers.hdf5"
TSC = "TSC_Cutimages/TSC_eachhalo_snap99.hdf5"


def dump(path, max_rows=3):
    print(f"\n{'='*70}\n{path}\n{'='*70}")
    with h5py.File(path, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name:<45} {str(obj.shape):<16} {obj.dtype}")
        f.visititems(show)
        print("  -- attrs --")
        for k, v in f.attrs.items():
            print(f"  @{k} = {v}")

        def peek(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.size:
                for k, v in obj.attrs.items():
                    print(f"    {name} @{k} = {v}")
        f.visititems(peek)

        print("  -- sample values --")
        def sample(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.size and obj.ndim <= 2:
                try:
                    print(f"    {name}: {obj[:max_rows]}")
                except Exception as e:
                    print(f"    {name}: <{e}>")
        f.visititems(sample)


for p in (TRUTH, TSC, TARGET):
    if os.path.exists(p):
        dump(p)
    else:
        print(f"MISSING: {p}")

# snapshot coverage of the per-snapshot catalogs
base = os.path.expanduser("~/data/Chuiyang/targethalo_cat_TNGCluster")
snaps = sorted(int(d.split("_")[-1]) for d in os.listdir(base)
               if d.startswith("targethalo_cat_"))
print(f"\ntargethalo snapshots: {snaps[0]}-{snaps[-1]} (n={len(snaps)})")
