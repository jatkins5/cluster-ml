"""B1: does the image add anything beyond mass and total brightness?

The audit found that one scalar -- log total linear weight -- reaches OOF
R2 0.432 against pseudo-TSC, and log M500 alone 0.314, while the pooled CNN
(now honestly scored) sits at 0.487. That gap is the entire image-based
claim of the paper, so it needs measuring properly rather than by comparing
two numbers from different scripts.

Reported here, all on the same 352 clusters and the same 5-fold splits:

  1. Scalar baselines, OOF ridge.
  2. The CNN alone (mean of 5 seeds' saved OOF predictions).
  3. Incremental R2: TSC ~ scalars, versus TSC ~ scalars + CNN prediction.
     This is the number a referee will ask for. The CNN column is already
     out-of-fold, so stacking it in a second CV is mildly optimistic; it is
     an upper bound on the increment, which is the conservative direction
     for the question being asked (does the image add *anything*).
  4. R2 within mass terciles, and the partial rank correlation with TSC at
     fixed mass -- the same diagnostics that showed the transfer model is
     mostly a mass regressor.
  5. The reverse direction: can the scalars predict what the CNN predicts?
     A CNN that is only reading brightness is fully explained by them.
"""
import glob

import h5py
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATASET = "dataset_nh4_128.h5"
KF = KFold(5, shuffle=True, random_state=0)


def ridge_oof(X, y):
    if X.ndim == 1:
        X = X[:, None]
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return cross_val_predict(model, X, y, cv=KF)


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        A = np.column_stack([b, np.ones_like(b)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return stats.pearsonr(resid(rx, rz), resid(ry, rz))[0]


# ---------- data ----------
with h5py.File(DATASET) as f:
    hid = f["meta/halo_id"][:]
    r500 = f["meta/r500c_kpc"][:]
    tsc = f["labels/pseudo_tsc"][:].astype(np.float64)
    imgs = f["images"][:]
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mmap = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
logm = np.array([mmap[h] for h in hid])
lin_tot = np.log10(np.sinh(imgs.astype(np.float64)).sum(axis=(1, 2, 3)) + 1)
arc_mean = imgs.mean(axis=(1, 2, 3))
frac_pos = (imgs > 0).mean(axis=(1, 2, 3))
SCALARS = np.column_stack([logm, np.log10(r500), lin_tot, arc_mean, frac_pos])
SCALAR_NAMES = ["log M500", "log r500", "log total w", "mean arcsinh", "filled frac"]

cnn = {}
for p in sorted(glob.glob("cnn_preds/pooled_nh4_inner_s*.npz")):
    d = np.load(p, allow_pickle=True)
    assert np.array_equal(d["halo_id"], hid), p
    assert np.allclose(d["label"], tsc), p
    cnn[int(str(d["seed"]))] = d["oof_pred"]
seeds = sorted(cnn)
cnn_mean = np.mean([cnn[s] for s in seeds], axis=0)
print(f"{len(seeds)} CNN seeds: {seeds};  {len(hid)} clusters\n")

# ---------- 1-2. baselines ----------
print("======== predictors of pseudo-TSC, 5-fold OOF, same splits")
print(f"{'predictor':<34}{'OOF R2':>9}{'Spearman':>10}")
for name, x in zip(SCALAR_NAMES, SCALARS.T):
    print(f"{name:<34}{r2_score(tsc, ridge_oof(x, tsc)):>9.3f}"
          f"{stats.spearmanr(x, tsc)[0]:>10.3f}")
r2_scalars = r2_score(tsc, ridge_oof(SCALARS, tsc))
print(f"{'all 5 scalars (ridge)':<34}{r2_scalars:>9.3f}{'':>10}")
per_seed = [r2_score(tsc, cnn[s]) for s in seeds]
print(f"{'pooled CNN, per seed':<34}{np.mean(per_seed):>9.3f}"
      f"   sd {np.std(per_seed, ddof=1):.3f}")
r2_cnn = r2_score(tsc, cnn_mean)
print(f"{'pooled CNN, 5-seed ensemble':<34}{r2_cnn:>9.3f}"
      f"{stats.spearmanr(cnn_mean, tsc)[0]:>10.3f}")

# ---------- 3. incremental ----------
print("\n======== does the image add anything beyond the scalars?")
combo = np.column_stack([SCALARS, cnn_mean])
r2_combo = r2_score(tsc, ridge_oof(combo, tsc))
print(f"  scalars only            {r2_scalars:+.3f}")
print(f"  scalars + CNN           {r2_combo:+.3f}   increment {r2_combo - r2_scalars:+.3f}")
print(f"  CNN only                {r2_cnn:+.3f}")
mass_only = ridge_oof(logm, tsc)
r2_m = r2_score(tsc, mass_only)
r2_m_cnn = r2_score(tsc, ridge_oof(np.column_stack([logm, cnn_mean]), tsc))
print(f"  mass only               {r2_m:+.3f}")
print(f"  mass + CNN              {r2_m_cnn:+.3f}   increment {r2_m_cnn - r2_m:+.3f}")

# Residual form of the same question: what the scalars cannot explain,
# can the CNN? (Both columns are OOF, so this is an honest scatter plot.)
res_tsc = tsc - ridge_oof(SCALARS, tsc)
res_cnn = cnn_mean - ridge_oof(SCALARS, cnn_mean)
print(f"\n  corr(TSC residual, CNN residual) after removing all 5 scalars: "
      f"Pearson {stats.pearsonr(res_tsc, res_cnn)[0]:+.3f}  "
      f"Spearman {stats.spearmanr(res_tsc, res_cnn)[0]:+.3f}")
print(f"  partial Spearman(CNN, TSC | log M500) = "
      f"{partial_spearman(cnn_mean, tsc, logm):+.3f}   "
      f"(raw {stats.spearmanr(cnn_mean, tsc)[0]:+.3f})")

# ---------- 4. within mass bins ----------
print("\n======== R2 within mass terciles (a pure mass model scores ~0 here)")
q = np.quantile(logm, [1 / 3, 2 / 3])
bins = [("low", logm <= q[0]), ("mid", (logm > q[0]) & (logm <= q[1])),
        ("high", logm > q[1])]
print(f"{'bin':<8}{'n':>5}{'CNN R2':>9}{'scalars R2':>12}{'mass R2':>9}"
      f"{'CNN rho':>9}{'TSC sd':>8}")
for name, b in bins:
    print(f"{name:<8}{b.sum():>5}{r2_score(tsc[b], cnn_mean[b]):>9.3f}"
          f"{r2_score(tsc[b], ridge_oof(SCALARS, tsc)[b]):>12.3f}"
          f"{r2_score(tsc[b], mass_only[b]):>9.3f}"
          f"{stats.spearmanr(cnn_mean[b], tsc[b])[0]:>9.3f}"
          f"{tsc[b].std():>8.3f}")

# ---------- 5. is the CNN just a scalar model? ----------
print("\n======== how much of the CNN's own output do the scalars explain?")
print(f"  CNN ~ all 5 scalars     R2 {r2_score(cnn_mean, ridge_oof(SCALARS, cnn_mean)):+.3f}")
print(f"  CNN ~ log M500          R2 {r2_score(cnn_mean, ridge_oof(logm, cnn_mean)):+.3f}")
print(f"  CNN ~ log total w       R2 {r2_score(cnn_mean, ridge_oof(lin_tot, cnn_mean)):+.3f}")
print("\n  (for reference, the same fits against the TRUE label:)")
print(f"  TSC ~ all 5 scalars     R2 {r2_scalars:+.3f}")
