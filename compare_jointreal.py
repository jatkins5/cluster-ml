"""Joint radio + X-ray with both modalities observationally degraded.

  radio    radio injected into real LoTSS DR3 fields, simulation's own flux
           (b2_sfimg_s*: the recommended transfer configuration)
  xray     X-ray thinned to real Chandra archive depths + soxs sky background
           (xreal_archive_s*)
  joint    both, in one model (jointreal_s*)

Same seeds, grouped folds, inner-split selection and cluster-mean scoring
over three noise realizations. Predictions are aligned by halo_id, since the
radio and X-ray datasets were built separately.
"""
import glob
import re

import h5py
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
PAT = {"radio": "cnn_preds/b2_sfimg_s*_preds.npz",
       "xray": "cnn_preds/xreal_archive_s*_preds.npz",
       "joint": "cnn_preds/jointreal_s*_preds.npz"}
if "--placed" in sys.argv:
    # X-ray at each target's redshift and ACIS-I exposure, with the fixed
    # z=0.05 versions kept alongside for comparison.
    PAT = {"radio": "cnn_preds/b2_sfimg_s*_preds.npz",
           "xray": "cnn_preds/xplaced_s*_preds.npz",
           "joint": "cnn_preds/jointplaced_s*_preds.npz",
           "xray_z05": "cnn_preds/xreal_archive_s*_preds.npz",
           "joint_z05": "cnn_preds/jointreal_s*_preds.npz"}

runs, ref = {}, None
for k, pat in PAT.items():
    runs[k] = {}
    for p in sorted(glob.glob(pat)):
        d = np.load(p, allow_pickle=True)
        s = int(re.search(r"_s(\d+)_preds\.npz$", p).group(1))
        m = dict(zip(d["halo_id"], d["oof_cluster"]))
        t = dict(zip(d["halo_id"], d["tsc"]))
        if ref is None:
            ref = np.array(sorted(m))
            tsc = np.array([t[h] for h in ref], dtype=np.float64)
        runs[k][s] = np.array([m[h] for h in ref])
seeds = sorted(set.intersection(*[set(v) for v in runs.values()]))
print(f"paired seeds {seeds}, {len(ref)} clusters")

with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
logm = np.array([mm[h] for h in ref])

r2 = {k: np.array([r2_score(tsc, v[s]) for s in seeds]) for k, v in runs.items()}
ens = {k: np.mean([v[s] for s in seeds], axis=0) for k, v in runs.items()}
print(f"\n{'seed':>6}" + "".join(f"{k:>10}" for k in runs))
for i, s in enumerate(seeds):
    print(f"{s:>6}" + "".join(f"{r2[k][i]:>10.3f}" for k in runs))
print(f"{'mean':>6}" + "".join(f"{r2[k].mean():>10.3f}" for k in runs))
print(f"{'sd':>6}" + "".join(f"{r2[k].std(ddof=1):>10.3f}" for k in runs))
print(f"{'ens':>6}" + "".join(f"{r2_score(tsc, ens[k]):>10.3f}" for k in runs))

stack = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(
    alphas=np.logspace(-3, 3, 13))), np.column_stack([ens["radio"], ens["xray"]]),
    tsc, cv=KFold(5, shuffle=True, random_state=0))
print(f"\n  stacked radio + X-ray ensembles: {r2_score(tsc, stack):.3f}")

print("\n======== paired differences")
pairs = [("joint", "radio"), ("joint", "xray"), ("xray", "radio")]
if "joint_z05" in runs:
    pairs += [("xray", "xray_z05"), ("joint", "joint_z05")]
for a, b in pairs:
    d = r2[a] - r2[b]
    print(f"  {a:<6} - {b:<6} {d.mean():+.3f} +- "
          f"{d.std(ddof=1) / np.sqrt(len(d)):.3f} (SE)  "
          f"p={stats.ttest_rel(r2[a], r2[b]).pvalue:.4f}  "
          f"{int((d > 0).sum())}/{len(d)} positive")

print("\n======== mass dependence (3-seed ensembles)")
rk = stats.rankdata
def partial(pred):
    def resid(a, b):
        A = np.column_stack([b, np.ones_like(b)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return stats.pearsonr(resid(rk(pred), rk(logm)), resid(rk(tsc), rk(logm)))[0]
q = np.quantile(logm, [1 / 3, 2 / 3])
bins = [("low", logm <= q[0]), ("mid", (logm > q[0]) & (logm <= q[1])),
        ("high", logm > q[1])]
print(f"{'model':<8}{'rho(p,M)':>10}{'partial|M':>11}"
      + "".join(f"{n:>8}" for n, _ in bins))
for k in runs:
    print(f"{k:<8}{stats.spearmanr(ens[k], logm)[0]:>10.3f}"
          f"{partial(ens[k]):>11.3f}"
          + "".join(f"{r2_score(tsc[b], ens[k][b]):>8.3f}" for _, b in bins))

print("\n======== by merger age (RMSE, Gyr)")
for lab, b in [("TSC <= 1", tsc <= 1), ("1-2", (tsc > 1) & (tsc <= 2)),
               ("> 2", tsc > 2)]:
    print(f"  {lab:<9} n={b.sum():3d}  " + "  ".join(
        f"{k} {np.sqrt(np.mean((ens[k][b] - tsc[b]) ** 2)):.3f}" for k in runs))
