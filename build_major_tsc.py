"""Build major-merger TSC labels from the TNG-Cluster collision catalog.

The shipped label (TSC_eachhalo_snap99.hdf5) counts collisions of *any*
mass ratio -- validate_merger_proxy.py --stage truth reproduces it at
rho=0.999 with no ratio cut, and the median ratio of the event that sets
it is 0.09. Radio relics and haloes are driven by major mergers, so this
writes the same clock restricted to progressively stronger mergers.

Also writes `tsc_any_on_major` -- the original any-ratio label with the
same halos masked out -- so a retrain can separate "different label"
from "different sample".
"""
import h5py
import numpy as np

EVENTS = "TSC_Cutimages/cluster_mergers.hdf5"
BASE = "TSC_Cutimages/TSC_eachhalo_snap99.hdf5"
OUT = "tsc_major_snap99.hdf5"
T0 = 13.797615896896383  # t(snap 99), from the catalog's own Snap/T_coll pairs
VARIANTS = {"tsc_mr033": 1.0 / 3.0, "tsc_mr020": 0.20, "tsc_mr010": 0.10}


def main():
    with h5py.File(BASE, "r") as f:
        halo_id = f["halo_id"][:]
        tsc_any = f["tsc_gyr"][:]
    with h5py.File(EVENTS, "r") as f:
        ev_idx = f["HaloID"][:]        # row index into the target-halo list
        ratio = f["Mass_ratio"][:]
        t_coll = f["T_coll"][:]

    n = len(halo_id)
    out = {}
    for name, cut in VARIANTS.items():
        tsc = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            m = (ev_idx == i) & (ratio > cut) & (t_coll <= T0)
            if m.any():
                tsc[i] = T0 - t_coll[m].max()
        out[name] = tsc
        ok = np.isfinite(tsc)
        print(f"{name} (ratio > {cut:.3f}): n={ok.sum()}/{n}  "
              f"med={np.median(tsc[ok]):.2f} "
              f"IQR=[{np.percentile(tsc[ok], 25):.2f}, "
              f"{np.percentile(tsc[ok], 75):.2f}] "
              f"max={tsc[ok].max():.2f} Gyr  "
              f"(<=2 Gyr: {(tsc[ok] <= 2).sum()})")

    major = np.isfinite(out["tsc_mr033"])
    any_on_major = np.where(major, tsc_any, np.nan)
    print(f"\ncontrol tsc_any_on_major: n={np.isfinite(any_on_major).sum()} "
          f"med={np.nanmedian(any_on_major):.2f} Gyr")

    corr = np.corrcoef(out["tsc_mr033"][major], tsc_any[major])[0, 1]
    print(f"Pearson r(major TSC, any-ratio TSC) on the shared halos: "
          f"{corr:.3f}")

    with h5py.File(OUT, "w") as f:
        f.create_dataset("halo_id", data=halo_id)
        for name, v in out.items():
            f.create_dataset(name, data=v)
        f.create_dataset("tsc_any_on_major", data=any_on_major)
        f.create_dataset("tsc_any", data=tsc_any)
        f.attrs["t0_gyr"] = T0
        f.attrs["description"] = (
            "Time since the last collision above a mass-ratio cut, derived "
            "from cluster_mergers.hdf5. tsc_any reproduces the shipped "
            "TSC_eachhalo_snap99 label (no ratio cut).")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
