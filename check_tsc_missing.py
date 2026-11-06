#!/usr/bin/env python3
"""R22-A evidence: what a NaN catalogue TSC means (read-only on Chuiyang's files).
Catalogue time coverage and selection, and for every NaN target its catalogue
records, at snapshots 99 and 91."""
import h5py
import numpy as np
from astropy.cosmology import Planck15

C = "/oscar/data/idellant/Chuiyang/TSC_new"
with h5py.File(f"{C}/cluster_mergers.hdf5") as f:
    D = {k: f[k][:] for k in ("HaloID", "T_coll", "Snap_coll", "Pre_snap", "Mass_ratio",
                              "Subcluster_mass", "M_Crit200_pre", "TSC_full")}
with h5py.File(f"{C}/TNG-Cluster_Catalog.hdf5") as f:
    hid = f["haloID"][:]
T = D["T_coll"]
print(f"{len(T)} catalogue events for {len(np.unique(D['HaloID']))} of {len(hid)} targets; "
      f"T_coll finite {np.isfinite(T).sum()}, NaN {np.isnan(T).sum()}, negative {(T < 0).sum()}")
fin = np.isfinite(T) & (T >= 0)
print(f"T_coll range {T[fin].min():.3f} - {T[fin].max():.3f} Gyr (age); "
      f"z range {1/Planck15.scale_factor(0) if False else ''}"
      f"Snap_coll range {D['Snap_coll'].min()} - {D['Snap_coll'].max()}")
zmin_age = Planck15.age(1.0).value
print(f"age at z = 1: {zmin_age:.3f} Gyr (Lee et al. 2024 consider z <= 1)")
sm = D["Subcluster_mass"]
print(f"Subcluster_mass: min {np.nanmin(sm):.4g}, median {np.nanmedian(sm):.4g} (units as stored); "
      f"Mass_ratio min {np.nanmin(D['Mass_ratio']):.4g} median {np.nanmedian(D['Mass_ratio']):.3g} max {np.nanmax(D['Mass_ratio']):.3g}")
for snap, z in ((99, 0.0), (91, 0.0994018026302219)):
    t_now = Planck15.age(z).value
    with h5py.File(f"{C}/TSC_eachhalo_snap{snap}.hdf5") as f:
        keys, tc = f["FOF_Halo_ID_At99"][:], f["tsc_gyr_from_Tcoll"][:]
    print(f"\n== snapshot {snap}: t_now {t_now:.3f} Gyr; 4-Gyr window starts at age {t_now - 4:.3f} "
          f"(catalogue covers from {T[fin].min():.3f}: {'yes' if T[fin].min() < t_now - 4 else 'NO'})")
    for k in keys[np.isnan(tc)]:
        row = int(np.where(hid == k)[0][0])          # HaloID is the row index of the target list
        ev = D["HaloID"] == row
        tt = T[ev]
        past = tt[np.isfinite(tt) & (tt <= t_now)]
        fut = tt[np.isfinite(tt) & (tt > t_now)]
        print(f"  FOF{k} (row {row}): {int(ev.sum())} events; finite T_coll {int(np.isfinite(tt).sum())} "
              f"(before t_now {len(past)}, after {len(fut)}: {np.round(fut, 3).tolist()}); "
              f"NaN T_coll {int(np.isnan(tt).sum())}; M200_pre {np.nanmax(D['M_Crit200_pre'][ev]) if ev.any() else float('nan'):.3g}")
    # finite labels: does every finite one come from an event inside the catalogue's coverage?
    print(f"  finite labels: {int(np.isfinite(tc).sum())}; max finite {np.nanmax(tc):.3f} Gyr")
