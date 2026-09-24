"""Joint radio + X-ray model vs each modality alone.

All five conditions share seeds, folds, protocol (inner-split selection) and
label (pseudo-TSC), so differences are paired by seed:

  radio        radio-only pooled CNN
  xray         X-ray-only, fixed stretch
  joint        radio + X-ray (fixed stretch), trained together
  joint_orig   radio + X-ray with the original, broken stretch -- the setup
               behind the old "X-ray adds nothing" conclusion
  stacked      second-level ridge on the radio and X-ray OOF predictions,
               the upper-bound estimate (+0.054) this run is meant to test
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

PATTERNS = {"radio": "cnn_preds/pooled_nh4_inner_s*.npz",
            "xray": "cnn_preds/xray_scaled_inner_s*.npz",
            "joint": "cnn_preds/joint_scaled_inner_s*.npz",
            "joint_orig": "cnn_preds/joint_orig_inner_s*.npz"}

with h5py.File("dataset_nh4_128.h5") as f:
    hid = f["meta/halo_id"][:]
    tsc = f["labels/pseudo_tsc"][:].astype(np.float64)
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
logm = np.array([mm[h] for h in hid])

runs = {}
for k, pat in PATTERNS.items():
    runs[k] = {}
    for p in sorted(glob.glob(pat)):
        d = np.load(p, allow_pickle=True)
        assert np.array_equal(d["halo_id"], hid), p
        runs[k][int(re.search(r"_s(\d+)\.npz$", p).group(1))] = d["oof_pred"]
seeds = sorted(set.intersection(*[set(v) for v in runs.values()]))
print(f"paired seeds: {seeds}")
r2 = {k: np.array([r2_score(tsc, v[s]) for s in seeds]) for k, v in runs.items()}
ens = {k: np.mean([v[s] for s in seeds], axis=0) for k, v in runs.items()}

print(f"\n{'seed':>6}" + "".join(f"{k:>12}" for k in runs))
for i, s in enumerate(seeds):
    print(f"{s:>6}" + "".join(f"{r2[k][i]:>12.3f}" for k in runs))
print(f"{'mean':>6}" + "".join(f"{r2[k].mean():>12.3f}" for k in runs))
print(f"{'sd':>6}" + "".join(f"{r2[k].std(ddof=1):>12.3f}" for k in runs))
print(f"{'ens':>6}" + "".join(f"{r2_score(tsc, ens[k]):>12.3f}" for k in runs))

stack = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(
    alphas=np.logspace(-3, 3, 13))), np.column_stack([ens["radio"], ens["xray"]]),
    tsc, cv=KFold(5, shuffle=True, random_state=0))
print(f"\n  stacked radio + X-ray ensembles (upper-bound estimate): "
      f"{r2_score(tsc, stack):.3f}")

print("\n======== paired differences")
for a, b in [("joint", "radio"), ("joint", "xray"), ("joint", "joint_orig"),
             ("joint_orig", "radio")]:
    d = r2[a] - r2[b]
    p = stats.ttest_rel(r2[a], r2[b]).pvalue
    print(f"  {a:<11} - {b:<11} {d.mean():+.3f} +- "
          f"{d.std(ddof=1) / np.sqrt(len(d)):.3f} (SE)  p={p:.4f}  "
          f"{int((d > 0).sum())}/{len(d)} positive")

print("\n======== mass dependence and within-mass skill (5-seed ensembles)")
q = np.quantile(logm, [1 / 3, 2 / 3])
bins = [("low", logm <= q[0]), ("mid", (logm > q[0]) & (logm <= q[1])),
        ("high", logm > q[1])]
rk = stats.rankdata
def partial(pred):
    def resid(a, b):
        A = np.column_stack([b, np.ones_like(b)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return stats.pearsonr(resid(rk(pred), rk(logm)), resid(rk(tsc), rk(logm)))[0]
print(f"{'model':<12}{'rho(p,M)':>10}{'partial|M':>11}"
      + "".join(f"{n:>8}" for n, _ in bins))
for k in runs:
    print(f"{k:<12}{stats.spearmanr(ens[k], logm)[0]:>10.3f}"
          f"{partial(ens[k]):>11.3f}"
          + "".join(f"{r2_score(tsc[b], ens[k][b]):>8.3f}" for _, b in bins))

# Where does the joint model gain: recent or old mergers?
print("\n======== by merger age (5-seed ensembles, RMSE in Gyr)")
for lab, b in [("TSC <= 1 Gyr", tsc <= 1), ("1-2 Gyr", (tsc > 1) & (tsc <= 2)),
               ("> 2 Gyr", tsc > 2)]:
    print(f"  {lab:<14} n={b.sum():3d}  " + "  ".join(
        f"{k} {np.sqrt(np.mean((ens[k][b] - tsc[b]) ** 2)):.3f}" for k in runs))
