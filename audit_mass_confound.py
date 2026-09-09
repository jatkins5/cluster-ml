"""Is the transfer model reading mass instead of merger state?

audit_results.py showed total brightness alone predicts pseudo-TSC at OOF
R2 0.43 and log M500 alone at 0.31. The forward model discards the sim's own
brightness and re-anchors flux to the Cuciti P150-M500 relation, so in the
mocks SNR scales as M500^3.55 at fixed rms: a CNN can recover mass from SNR
and mass predicts TSC. If that is most of the 0.43 mock R2, the real-cluster
ordering is a mass ordering, and A399/A401 coming out youngest means nothing
more than "they are massive".

Checks:
  1. Spearman of mock OOF predictions vs log M500, vs TSC, and the partial
     correlation with TSC controlling for mass.
  2. Real predictions vs LoVoCCS L_X (mass proxy) and vs the real cutouts'
     own brightness statistics.
  3. Mock OOF R2 within mass terciles: a model that only reads mass has
     near-zero R2 inside a narrow mass bin.
"""
import glob

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score


def partial_spearman(x, y, z):
    """rho(x,y | z) from rank residuals."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        A = np.column_stack([b, np.ones_like(b)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return stats.pearsonr(resid(rx, rz), resid(ry, rz))[0]


with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mmap = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))

print("======== mock OOF predictions: mass vs TSC")
print(f"{'run':<22}{'rho(pred,M)':>12}{'rho(pred,TSC)':>14}{'rho(TSC,M)':>11}"
      f"{'partial|M':>11}{'R2 lowM':>9}{'R2 midM':>9}{'R2 highM':>9}")
for p in sorted(glob.glob("*_preds.npz")):
    d = np.load(p, allow_pickle=True)
    if "obs_pred" not in d:
        continue
    pred, tsc, hid = d["oof_cluster"], d["tsc"], d["halo_id"]
    m = np.array([mmap[h] for h in hid])
    q = np.quantile(m, [1 / 3, 2 / 3])
    bins = [m <= q[0], (m > q[0]) & (m <= q[1]), m > q[1]]
    r2b = [r2_score(tsc[b], pred[b]) for b in bins]
    print(f"{p[:-10]:<22}{stats.spearmanr(pred, m)[0]:>12.3f}"
          f"{stats.spearmanr(pred, tsc)[0]:>14.3f}"
          f"{stats.spearmanr(tsc, m)[0]:>11.3f}"
          f"{partial_spearman(pred, tsc, m):>11.3f}"
          + "".join(f"{v:>9.3f}" for v in r2b))

print("\n======== real predictions vs L_X and vs cutout brightness")
df = pd.read_csv("LoVoCCS_target_list - lovoccs.csv")
df["key"] = df["name"].astype(str).str.replace(" ", "")
lx = dict(zip(df["key"], pd.to_numeric(df["lx"], errors="coerce")))
with h5py.File("injected_nh4.h5") as f:
    obs = f["obs/images"][:]
    names = [s.decode() if isinstance(s, bytes) else str(s)
             for s in f["obs/name"][:]]
    z = f["obs/z"][:]
    rms = f["obs/rms"][:]
lin = np.sinh(obs.astype(np.float64))          # back to map/rms
tot = lin.sum(axis=(1, 2))
peak = lin.max(axis=(1, 2))
frac3 = (lin > 3).mean(axis=(1, 2))
lxv = np.array([lx.get(n, np.nan) for n in names])
print(f"L_X available for {np.isfinite(lxv).sum()}/{len(names)} targets")
for p in sorted(glob.glob("injtrans_*_preds.npz")) + \
        sorted(glob.glob("regentrans_nh4_*_preds.npz")):
    d = np.load(p, allow_pickle=True)
    assert [str(s) for s in d["obs_name"]] == names
    pr = d["obs_pred"]
    ok = np.isfinite(lxv)
    print(f"{p[:-10]:<22} rho(pred, L_X) {stats.spearmanr(pr[ok], lxv[ok])[0]:+.2f}"
          f"  rho(pred, total SNR) {stats.spearmanr(pr, tot)[0]:+.2f}"
          f"  rho(pred, peak SNR) {stats.spearmanr(pr, peak)[0]:+.2f}"
          f"  rho(pred, frac>3) {stats.spearmanr(pr, frac3)[0]:+.2f}"
          f"  rho(pred, z) {stats.spearmanr(pr, z)[0]:+.2f}"
          f"  rho(pred, rms) {stats.spearmanr(pr, rms)[0]:+.2f}")
print("\nper-target:")
d = np.load("injtrans_s42_preds.npz", allow_pickle=True)
for i in np.argsort(d["obs_pred"]):
    print(f"  {names[i]:<10} pred {d['obs_pred'][i]:5.2f}  z {z[i]:.3f}  "
          f"L_X {lxv[i]:6.2f}  totSNR {tot[i]:8.0f}  peak {peak[i]:6.1f}  "
          f"frac>3 {frac3[i]:.3f}")
