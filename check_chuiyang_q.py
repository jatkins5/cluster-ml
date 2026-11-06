#!/usr/bin/env python3
"""Answers for Chuiyang (2026-11-04), read-only on her files.

1. Her direct TSC (TSC_eachhalo_snap{99,91}.hdf5: tsc_gyr) vs our training
   label (pseudo-TSC from the merger-score curves), per snapshot.
2. How much emission the n_H < 1e-4 cut removes, old (HB07) vs new (synch)
   weights, from the saved cell files (no cuts vs cut, same halos).
3. The gas state (T, n_H, Mach) of the emission the cut removes under the
   NEW weights, from the cutouts of a sample of halos.
"""
import glob
import os
import re

import h5py
import numpy as np
from scipy import stats

import build_radio_cells as B

C = "/oscar/data/idellant/Chuiyang"

print("== 1. her TSC vs our pseudo-TSC")
with h5py.File("dataset_syn_nh4_144_s9991_128.h5") as f:
    ps, snap = f["labels/pseudo_tsc"][:], f["meta/snap"][:]
    hid, gid = f["meta/halo_id"][:], f["meta/group_id"][:]
for s in (99, 91):
    with h5py.File(f"{C}/TSC_Cutimages/TSC_eachhalo_snap{s}.hdf5") as f:
        t = f["tsc_gyr"][:].astype(float)
        key = f["halo_id"][:] if s == 99 else f["FOF_Halo_ID_At99"][:]
    m = snap == s
    pos = {int(k): i for i, k in enumerate(key)}
    ours = gid[m] if s == 91 else hid[m]
    idx = np.array([pos.get(int(k), -1) for k in ours])
    tt, pp = t[idx], ps[m]
    fin = np.isfinite(tt) & (idx >= 0)
    print(f"  snapshot {s}: {m.sum()} rows, joined {int((idx >= 0).sum())}; her tsc NaN {int(np.isnan(tt).sum())}, "
          f"exactly 0: {int((tt == 0).sum())}, negative: {int((tt < 0).sum())}, > 4 Gyr: {int((tt > 4).sum())}")
    print(f"    Spearman(pseudo, tsc) {stats.spearmanr(pp[fin], tt[fin])[0]:.3f}; "
          f"Pearson on capped tsc {stats.pearsonr(pp[fin], np.clip(tt[fin], 0.1, 4))[0]:.3f}; "
          f"median |pseudo - tsc| {np.median(np.abs(pp[fin] - tt[fin])):.3f} Gyr")
    print(f"    medians: pseudo {np.median(pp):.2f}, tsc {np.nanmedian(tt):.2f} Gyr; "
          f"pseudo capped at 4.0: {int((pp == 4.0).sum())}, their tsc there: "
          f"{np.round(np.sort(tt[pp == 4.0])[:8], 2).tolist()}")

print("\n== 2. emission removed by the n_H < 1e-4 cut (cell files, snapshot 99)")
for lab, nocut, cut in (("old HB07", "Radio_Cells_144", "Radio_Cells_nh4_144"),
                        ("new synch", "Radio_Cells_syn_144", "Radio_Cells_syn_nh4_144")):
    kept, top_nc, top_c = [], [], []
    for p in sorted(glob.glob(f"{nocut}/cells_FOF*.npz")):
        q = os.path.join(cut, os.path.basename(p))
        if not os.path.exists(q):
            continue
        a, b = np.load(p)["w"], np.load(q)["w"]
        kept.append(b.sum() / a.sum())
        top_nc.append(a.max() / a.sum())
        top_c.append(b.max() / b.sum())
    kept = np.array(kept)
    print(f"  {lab:<10} {len(kept)} halos: power kept by the cut median {np.median(kept):.3f} "
          f"[{np.percentile(kept, 10):.3f}, {np.percentile(kept, 90):.3f}] (10-90%); "
          f"top-cell share {np.median(top_nc):.3f} -> {np.median(top_c):.3f}")

print("\n== 3. gas state of the emission the cut removes (new weights, 40 halos)")
B.set_redshift(0.0)
interp = B.make_ratio("Radio_Data_new/radio_spectral_ratio_table.npz")
srcs = sorted(glob.glob(f"{C}/Radio_Snap99_new/radio_FOF*.npz"))[::9][:40]
rem = {"w": [], "T": [], "nH": [], "M": []}
kept_w = {"w": [], "T": [], "nH": [], "M": []}
frac = []
for src in srcs:
    fof, sub = re.search(r"radio_FOF(\d+)_sub(\d+)", src).groups()
    cut = f"{C}/TNGCluster_Cutout/snap99/cutout_sub{sub}_FOF{fof}.hdf5"
    out = B.generate(cut, interp, nu_ghz=0.144, phase=True, model="synch")
    if out is None:
        continue
    _, w, _, T, nH, M = out
    r = nH > 1e-4
    frac.append(w[r].sum() / w.sum())
    for d, s_ in ((rem, r), (kept_w, ~r)):
        d["w"].append(w[s_] / w.sum()); d["T"].append(T[s_]); d["nH"].append(nH[s_]); d["M"].append(M[s_])
frac = np.array(frac)
print(f"  {len(frac)} halos: fraction of power in cells with n_H > 1e-4: median {np.median(frac):.3f} "
      f"[{np.percentile(frac, 10):.3f}, {np.percentile(frac, 90):.3f}]")


def wq(d, k, q):
    v, w = np.concatenate(d[k]), np.concatenate(d["w"])
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return v[o][np.searchsorted(c, q)]


for lab, d in (("removed (n_H > 1e-4)", rem), ("kept (n_H <= 1e-4)", kept_w)):
    print(f"  {lab:<22} power-weighted median T {wq(d, 'T', 0.5):.4g} keV "
          f"[{wq(d, 'T', 0.1):.3g}, {wq(d, 'T', 0.9):.3g}], n_H {wq(d, 'nH', 0.5):.3g} cm^-3, "
          f"Mach {wq(d, 'M', 0.5):.3g} [{wq(d, 'M', 0.1):.3g}, {wq(d, 'M', 0.9):.3g}]")
allw = np.concatenate(rem["w"] + kept_w["w"])
allT = np.concatenate(rem["T"] + kept_w["T"])
for tcut in (0.1, 0.5, 1.0):
    print(f"  share of power (all 40 halos pooled) in cells with T < {tcut} keV: "
          f"{allw[allT < tcut].sum() / allw.sum():.3f}")
