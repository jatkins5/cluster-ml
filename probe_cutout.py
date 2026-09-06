"""What is actually in a snap99 cutout, and how long does one take to read?"""
import time
import h5py
import numpy as np

fn = ("/oscar/data/idellant/Chuiyang/TNGCluster_Cutout/snap99/"
      "cutout_sub10008053_FOF18771334.hdf5")
t0 = time.time()
with h5py.File(fn, "r") as f:
    print("top-level:", list(f.keys()))
    print("Header attrs:", dict(f["Header"].attrs) if "Header" in f else "(none)")
    p0 = f["PartType0"]
    print("\nPartType0 fields:")
    for k in p0:
        print(f"  {k:24s} {p0[k].shape} {p0[k].dtype}")
    n = p0["Masses"].shape[0]
    mass = p0["Masses"][:]
    dens = p0["Density"][:]
    mach = p0["Machnumber"][:]
    edis = p0["EnergyDissipation"][:]
print(f"\nread {n} gas cells in {time.time()-t0:.1f}s")

shock = (mach > 1.3) & (edis > 0)
print(f"shock cells: {shock.sum()} ({shock.mean():.1%})")

# Volume in code units (ckpc/h)^3; both mass units cancel.
vol = mass[shock] / dens[shock]
r = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
a, h = 1.0, 0.6774
r_kpc = r * a / h
for q in [5, 25, 50, 75, 95]:
    print(f"  cell radius p{q:<2d}: {np.percentile(r_kpc, q):8.2f} kpc")
print(f"  native map pixel is ~18 kpc; cells larger than that: "
      f"{(r_kpc > 18).mean():.1%}")
