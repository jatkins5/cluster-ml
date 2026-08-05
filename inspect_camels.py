"""Dump schemas of Chuiyang's CAMELS radio + halo-mass-history files."""
import glob
import os

import h5py
import numpy as np

BASE = os.path.expanduser("~/data/Chuiyang/Camels")


def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"    {name}: shape={obj.shape} dtype={obj.dtype}")
        for k, v in obj.attrs.items():
            print(f"        .{k} = {v}")
    else:
        print(f"    [group] {name}")
        for k, v in obj.attrs.items():
            print(f"        .{k} = {v}")


radios = sorted(glob.glob(f"{BASE}/Radio/radio_GZ28_*_snap090.hdf5"))
print(f"radio files: {len(radios)}")
with h5py.File(radios[0], "r") as f:
    print(f"\n=== {os.path.basename(radios[0])} ===")
    f.visititems(walk)
    for k, v in f["Header"].attrs.items() if "Header" in f else []:
        pass
    print("  root attrs:", dict(f.attrs))
    for key in f:
        print(f"  {key} attrs:", dict(f[key].attrs))
    # particle stats
    for cand in ["radio/pos", "Radio/pos", "pos"]:
        if cand in f:
            pos = f[cand][:]
            w = f[cand.rsplit("/", 1)[0] + "/w"][:]
            print(f"  N particles={len(pos)}  "
                  f"pos range={pos.min():.1f}..{pos.max():.1f} kpc")
            print(f"  w: min={w.min():.3e} max={w.max():.3e} "
                  f"sum={w.sum():.3e} nonzero={np.count_nonzero(w)}")
            break

hist = sorted(glob.glob(f"{BASE}/Halomass_history/*.hdf5"))
print(f"\n=== halo-mass history: {[os.path.basename(h) for h in hist]} ===")
with h5py.File(hist[0], "r") as f:
    f.visititems(walk)
    print("  root attrs:", dict(f.attrs))
    keys = list(f.keys())
    print("  top-level keys:", keys[:10], "..." if len(keys) > 10 else "")

fof = f"{BASE}/Scripts/GZ28_snap090_FOF_info.hdf5"
if os.path.exists(fof):
    print(f"\n=== {os.path.basename(fof)} ===")
    with h5py.File(fof, "r") as f:
        f.visititems(walk)
        print("  root attrs:", dict(f.attrs))
