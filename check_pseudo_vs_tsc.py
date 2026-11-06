#!/usr/bin/env python3
"""Context for the TSC-hardness experiments (read-only): how pseudo-TSC
relates to the catalogue TSC. S(tau) = sum_mergers exp(-dt/tau); pseudo =
first tau with S >= 0.5 (interpolated on the 0.1..4.0 grid). For one isolated
event pseudo = dt/ln2, so pseudo reaches its 4-Gyr cap at TSC = 4 ln2 = 2.77
Gyr; additional events lower it. Snapshot 99 only."""
import h5py
import numpy as np
from astropy.cosmology import Planck15
from scipy import stats

C = "/oscar/data/idellant/Chuiyang/TSC_new"
with h5py.File(f"{C}/cluster_mergers.hdf5") as f:
    hid_row, tcoll = f["HaloID"][:], f["T_coll"][:]
    mr = f["Mass_ratio"][:]
with h5py.File(f"{C}/TNG-Cluster_Catalog.hdf5") as f:
    targets = f["haloID"][:]
with h5py.File(f"{C}/TSC_eachhalo_snap99.hdf5") as f:
    tk, tsc = f["FOF_Halo_ID_At99"][:], f["tsc_gyr_from_Tcoll"][:]
with h5py.File("dataset_nh4_144_s9991_128.h5") as f:
    snap, gid = f["meta/snap"][:], f["meta/group_id"][:]
    ps_all, tau = f["labels/pseudo_tsc"][:], f["labels/tau_gyr"][:]
    s_all = f["labels/label_score_all"][:]
m99 = snap == 99
pseudo = dict(zip(gid[m99].astype(int), ps_all[m99]))
score = dict(zip(gid[m99].astype(int), s_all[m99]))
t_now = Planck15.age(0).value
rows = []
for row, k in enumerate(targets):
    ev = np.sort(t_now - tcoll[(hid_row == row) & (tcoll <= t_now + 1e-9)])
    S = np.array([np.exp(-ev / t).sum() for t in tau])
    rows.append(dict(k=int(k), ev=ev, S=S, mr=mr[hid_row == row]))
print("== 1. does sum exp(-dt/tau) over the catalogue reproduce the stored merger score?")
err = [np.max(np.abs(r["S"] - score[r["k"]])) for r in rows if r["k"] in score]
print(f"   max |S_reconstructed - S_stored| over halos: median {np.median(err):.3g}, 95% {np.percentile(err, 95):.3g}, "
      f"max {np.max(err):.3g}")
P = np.array([pseudo[r["k"]] for r in rows]); T = np.array([tsc[list(tk).index(r["k"])] for r in rows])
n_ev = np.array([len(r["ev"]) for r in rows])
n2 = np.array([(r["ev"] <= 2.0).sum() for r in rows])
fin = np.isfinite(T)
print("\n== 2. pseudo vs catalogue TSC (T_coll), snapshot 99")
single = T / np.log(2)
pred1 = np.minimum(np.where(fin, single, 4.0), 4.0)
print(f"   pseudo == min(TSC/ln2, 4) within 0.05 Gyr for {np.mean(np.abs(P - pred1)[fin] < 0.05):.0%} of finite-TSC halos")
print(f"   pseudo below that single-event value (extra recent events lower it): "
      f"{np.mean((P < pred1 - 0.05)[fin]):.0%}")
cap_p = P >= 4.0 - 1e-6
print(f"   pseudo capped: {cap_p.sum()} halos; their TSC: min {np.nanmin(T[cap_p]):.2f}, median "
      f"{np.nanmedian(T[cap_p]):.2f} Gyr (single-event cap at 4 ln2 = {4 * np.log(2):.2f})")
band = fin & (T > 4 * np.log(2)) & (T <= 4)
print(f"   halos with 2.77 < TSC <= 4 Gyr (resolved by capped TSC, saturated in pseudo): {band.sum()}")
print(f"   collisions per halo (all recorded, before t_now): median {np.median(n_ev):.0f}, within 2 Gyr: "
      f"median {np.median(n2):.0f}, max {n2.max()}")
# how much of log pseudo is explained by TSC alone vs TSC + recent count
ok = fin & (T > 0) & (P < 4.0 - 1e-6) & (P > 0.1 + 1e-6)
y = np.log(P[ok])
X1 = np.column_stack([np.ones(ok.sum()), np.log(T[ok])])
X2 = np.column_stack([X1, np.log1p(n2[ok])])
for lab, X in (("log TSC", X1), ("log TSC + log(1 + n_2Gyr)", X2)):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"   uncapped, unfloored halos ({ok.sum()}): R2 of log pseudo on {lab}: {r2:.3f}")
print(f"   variances: pseudo {np.var(P):.3f}, capped TSC {np.var(np.where(fin, np.minimum(T, 4), 4)):.3f} Gyr^2; "
      f"Spearman(pseudo, capped TSC) {stats.spearmanr(P, np.where(fin, np.minimum(T, 4), 4))[0]:.3f}")
