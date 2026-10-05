"""Chuiyang's regenerated radio cells (Radio_Snap99, Radio_Snap91; read-only)
against ours, plus her new snap-91 merger-catalogue labels.

1. Recompute a few halos per snapshot with our build_radio_cells.generate at
   her settings (144 MHz, no density cut, snapshot redshift) and compare cell
   by cell: same cells, same weights?
2. All of her snap-99 files against the original 1.4 GHz upstream weights
   (Radio_Data): the ratio must be (0.144/1.4)^(-s/2) per cell, so >= 9.72,
   and the cell sets must be identical.
3. TSC_eachhalo_snap91.hdf5: structure, and agreement with the snap-91
   pseudo-TSC from label_score_pre and with the snap-99 clock.
"""
import glob
import os
import pickle
import re

import h5py
import numpy as np
from scipy import stats

import build_radio_cells as brc

C = "/oscar/data/idellant/Chuiyang"
interp_psi = brc.make_psi("Radio_Data/normalized_psi_table.npz")


def ids(p):
    m = re.search(r"radio_FOF(\d+)_sub(\d+)\.npz", p)
    return int(m.group(1)), int(m.group(2))


def compare(snap, z, n=4):
    files = sorted(glob.glob(f"{C}/Radio_Snap{snap}/radio_FOF*.npz"),
                   key=os.path.getsize)
    pick = files[:: max(1, len(files) // n)][:n]
    brc.set_redshift(z)
    with np.load(pick[0]) as d:
        print(f"  keys in her npz: {list(d.keys())}")
    for p in pick:
        fof, sub = ids(p)
        cut = f"{C}/TNGCluster_Cutout/snap{snap}/cutout_sub{sub}_FOF{fof}.hdf5"
        pos, w, _ = brc.generate(cut, interp_psi, nu_ghz=0.144)
        with np.load(p) as d:
            hp, hw = d["pos"], d["w"]
        same_n = len(hw) == len(w)
        msg = f"  FOF{fof}: cells hers {len(hw)} ours {len(w)}"
        if same_n:
            dp = np.abs(hp - pos).max()
            ok = np.isfinite(hw) & np.isfinite(w) & (hw > 0)
            rel = np.abs(w[ok] - hw[ok]) / hw[ok]
            msg += (f"; max |dpos| {dp:.2e} kpc; weight rel diff max "
                    f"{rel.max():.1e}; total ratio {w[ok].sum() / hw[ok].sum():.6f}")
        print(msg)


print("== snap 99: hers (new, 144 MHz) vs ours recomputed at 144 MHz, no cut")
compare(99, 0.0)

print("\n== snap 99: all of hers vs the original 1.4 GHz weights (Radio_Data)")
r_min, r_med, nmis, nfile = [], [], 0, 0
for p in sorted(glob.glob(f"{C}/Radio_Snap99/radio_FOF*.npz")):
    old = os.path.join("Radio_Data", os.path.basename(p))
    if not os.path.exists(old):
        continue
    nfile += 1
    with np.load(p) as a, np.load(old) as b:
        if len(a["w"]) != len(b["w"]):
            nmis += 1
            continue
        ok = np.isfinite(a["w"]) & np.isfinite(b["w"]) & (b["w"] > 0)
        r = a["w"][ok] / b["w"][ok]
        r_min.append(r.min()); r_med.append(np.median(r))
print(f"  {nfile} halos; cell-count mismatches {nmis}; per-cell ratio min over "
      f"all {min(r_min):.3f} (must be >= 9.72), median of medians "
      f"{np.median(r_med):.2f}")

print("\n== snap 91: hers vs ours recomputed at z = 0.0994, 144 MHz, no cut")
compare(91, 0.0994018026302219)

print("\n== TSC_eachhalo_snap91.hdf5")
def show(f):
    for k in f:
        o = f[k]
        print(f"  {k}: {o.shape} {o.dtype}" if isinstance(o, h5py.Dataset) else f"  {k}/")
    print("  attrs:", dict(f.attrs))
with h5py.File(f"{C}/TSC_Cutimages/TSC_eachhalo_snap91.hdf5") as f:
    show(f)
    t91 = {k: f[k][:] for k in f if isinstance(f[k], h5py.Dataset)}
with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5") as f:
    print("  (snap 99 file for comparison)")
    show(f)
    t99 = dict(zip(f["halo_id"][:], f["tsc_gyr"][:]))
hid = t91.get("FOF_Halo_ID_At99")
tsc = t91.get("tsc_gyr")
if hid is not None and tsc is not None:
    print(f"  exactly 0 (collision at snap 91 itself): {(tsc == 0).sum()}")
    print(f"  {len(hid)} halos; tsc_gyr median {np.nanmedian(tsc):.2f}, "
          f"range {np.nanmin(tsc):.2f}-{np.nanmax(tsc):.2f}, NaN {np.isnan(tsc).sum()}")
    in99 = np.isin(hid, list(t99))
    print(f"  halo ids found among snap-99 ids: {in99.sum()} / {len(hid)}")
    if in99.any():
        a = np.array([t99[h] for h in hid[in99]])
        b = tsc[in99]
        d = a - b
        print(f"  tsc(99) - tsc(91) for the same halo: median {np.nanmedian(d):.2f} Gyr "
              f"(1.32 Gyr elapsed between snaps if no new collision); "
              f"Spearman {stats.spearmanr(a, b, nan_policy='omit')[0]:+.2f}")
    with open("feats_labels_dict_tngcluster.pkl", "rb") as f:
        pkl = pickle.load(f)
    with h5py.File("dataset.h5") as f:
        tau = f["labels/tau_gyr"][:]
    keys = sorted([k for k in pkl[next(iter(pkl))][99] if k.startswith("label_score_pre_tau")],
                  key=lambda k: float(k.split("tau")[1]))
    pt, tt = [], []
    for h, t in zip(hid, tsc):
        e = pkl.get(int(h), {}).get(91)
        if e is None or not all(k in e for k in keys):
            continue
        sc = np.array([e[k] for k in keys])
        above = np.where(sc >= 0.5)[0]
        pt.append(tau[above[0]] if len(above) else tau[-1]); tt.append(t)
    if pt:
        print(f"  vs snap-91 pseudo-TSC (label_score_pre, z=0 ids): n={len(pt)}, "
              f"Spearman {stats.spearmanr(pt, tt, nan_policy='omit')[0]:+.2f}")
