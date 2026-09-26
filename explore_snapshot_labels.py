"""What would earlier full snapshots (91, 84, 78, 72) add to the training set?

Checked from files already on disk, before downloading anything:
  1. what the per-snapshot label pickle holds (features, halo ids, masses?)
  2. whether the merger-catalogue clock can be recomputed at any snapshot
     from cluster_mergers.hdf5 (collision times), as the z=0 label was
  3. how different each snapshot's label is from the same cluster's z=0
     label -- the leakage / redundancy question
  4. how the progenitor masses compare with the LoVoCCS weak-lensing masses
"""
import pickle

import h5py
import numpy as np
from astropy.cosmology import Planck15
from scipy import stats

FULL = {99: 0.0, 91: 0.1, 84: 0.2, 78: 0.3, 72: 0.4}

with open("feats_labels_dict_tngcluster.pkl", "rb") as f:
    pkl = pickle.load(f)
hid0 = list(pkl)[0]
print(f"pickle: {len(pkl)} halos; snapshots for one halo: "
      f"{sorted(pkl[hid0])[:3]}...{sorted(pkl[hid0])[-3:]}")
e = pkl[hid0][99]
print("keys at snap 99:", [k for k in e if not k.startswith("label_score")][:40])

print("\n======== merger catalogue")
with h5py.File("TSC_Cutimages/cluster_mergers.hdf5") as f:
    def show(g, pre=""):
        for k in g:
            o = g[k]
            if isinstance(o, h5py.Dataset):
                print(f"  {pre}{k}: {o.shape} {o.dtype}")
            else:
                show(o, pre + k + "/")
    show(f)
    cat = {k: f[k][:] for k in f if isinstance(f[k], h5py.Dataset)}

with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5") as f:
    tsc_hid = f["halo_id"][:]
    tsc99 = f["tsc_gyr"][:]

# Recompute the any-ratio collision clock at each full snapshot, as
# validate_merger_proxy.py reproduced it at z=0 (clock = T_coll, HaloID is a
# row index into the 352-target list).
if "HaloID" in cat and "T_coll" in cat:
    t0 = {s: Planck15.age(z).value for s, z in FULL.items()}
    rows = cat["HaloID"].astype(int)
    tc = cat["T_coll"].astype(float)
    print("\n======== any-ratio collision clock at each full snapshot")
    print(f"{'snap':>5}{'z':>5}{'t [Gyr]':>9}{'n valid':>9}{'median TSC':>12}"
          f"{'rho vs z=0':>12}{'same last event':>17}")
    tsc_s = {}
    last_s = {}
    for s, z in FULL.items():
        v, last = np.full(len(tsc_hid), np.nan), np.full(len(tsc_hid), np.nan)
        for i in range(len(tsc_hid)):
            m = (rows == i) & (tc <= t0[s] + 1e-6)
            if m.any():
                last[i] = tc[m].max()
                v[i] = t0[s] - last[i]
        tsc_s[s], last_s[s] = v, last
    for s, z in FULL.items():
        ok = np.isfinite(tsc_s[s]) & np.isfinite(tsc_s[99])
        same = np.mean(np.isclose(last_s[s][ok], last_s[99][ok]))
        print(f"{s:>5}{z:>5.1f}{t0[s]:>9.2f}{ok.sum():>9}"
              f"{np.nanmedian(tsc_s[s]):>12.2f}"
              f"{stats.spearmanr(tsc_s[s][ok], tsc_s[99][ok])[0]:>12.3f}"
              f"{same:>17.1%}")
    ok = np.isfinite(tsc_s[99]) & np.isfinite(tsc99)
    print(f"check: recomputed z=0 clock vs shipped label, Spearman "
          f"{stats.spearmanr(tsc_s[99][ok], tsc99[ok])[0]:.3f}")
else:
    print("merger catalogue lacks HaloID/T_coll; fields:", list(cat))

# Progenitor masses, if the pickle carries them.
mkeys = [k for k in e if "m500" in k.lower() or "mass" in k.lower()
         or "r500" in k.lower()]
print("\nmass-like keys in the pickle:", mkeys)
