"""Minkowski functionals of radio maps: topology summary statistics.

Phase 2 of the sim-vs-obs distributional validation. For each map we
binarize at a sweep of area-fraction thresholds (top-f fraction of
pixels), which makes the excursion sets invariant to any monotonic
intensity rescaling -- crucial because the sim radio normalization is
arbitrary. Per threshold we record the 2D Minkowski functionals:
  V0 = area fraction, V1 = boundary length, V2 = Euler characteristic
  (components minus holes), plus component/hole counts.

Uses:
1. Physics-informed features: do MF curves predict pseudo-TSC?
   (5-fold GroupKFold ridge + XGBoost, compare to tabular XGBoost 0.333
   and shallow CNN 0.529.)
2. First-look topology comparison of sim maps vs LoTSS 144 MHz cutouts
   of LoVoCCS clusters. Caveat: no beam/noise forward-modeling yet --
   obs maps are simply resampled to the sim grid, and LoTSS fields
   contain point sources the sims lack.
"""
import argparse
import glob
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

FRACS = np.geomspace(0.005, 0.5, 12)

STRUCT8 = np.ones((3, 3), dtype=bool)
STRUCT4 = ndimage.generate_binary_structure(2, 1)


def binary_mfs(B):
    """V0/V1/V2 + component/hole counts for a binary 2D excursion set."""
    area = B.mean()
    perim = (np.count_nonzero(B[:, 1:] != B[:, :-1])
             + np.count_nonzero(B[1:, :] != B[:-1, :])) / B.size
    _, ncomp = ndimage.label(B, structure=STRUCT8)
    lab_bg, nbg = ndimage.label(~B, structure=STRUCT4)
    border = np.unique(np.concatenate([
        lab_bg[0], lab_bg[-1], lab_bg[:, 0], lab_bg[:, -1]]))
    nholes = nbg - np.count_nonzero(border)
    return area, perim, ncomp, nholes, ncomp - nholes


def mf_curves(img, fracs=FRACS):
    """(len(fracs), 5) MF matrix at area-fraction thresholds."""
    flat = img.ravel()
    out = np.empty((len(fracs), 5))
    for i, f in enumerate(fracs):
        thr = np.quantile(flat, 1.0 - f)
        out[i] = binary_mfs(img > thr)
    return out


def oof_r2(X, y, groups, model_fn, n_folds=5, seed=0):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = model_fn()
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    ss_res = np.sum((y - oof) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    # cluster-averaged predictions
    ug = np.unique(groups)
    ym = np.array([y[groups == g].mean() for g in ug])
    pm = np.array([oof[groups == g].mean() for g in ug])
    r2_cl = 1 - np.sum((ym - pm) ** 2) / np.sum((ym - ym.mean()) ** 2)
    return 1 - ss_res / ss_tot, r2_cl, oof


def load_lotss(pattern, size):
    from astropy.io import fits
    maps, names = [], []
    for path in sorted(glob.glob(pattern)):
        with fits.open(path) as hdul:
            img = np.squeeze(hdul[0].data).astype(np.float64)
        img = np.nan_to_num(img, nan=0.0)
        if min(img.shape) < 64:
            continue
        s = min(img.shape)
        img = img[:s, :s]
        img = ndimage.zoom(img, size / s, order=1)
        maps.append(img)
        names.append(os.path.basename(path))
    return maps, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.h5")
    ap.add_argument("--lotss-glob",
                    default=os.path.expanduser(
                        "~/data/cluster-ml/lotss_images/*.fits"))
    ap.add_argument("--out-prefix", default="minkowski")
    args = ap.parse_args()

    with h5py.File(args.dataset, "r") as f:
        images = f["images"][:]              # (N, 3, H, W)
        halo_ids = f["meta/halo_id"][:]
        tsc = f["labels/pseudo_tsc"][:]
    N, P, H, W = images.shape
    print(f"sim maps: {N} clusters x {P} projections, {H}x{W}")

    # ---- sim MF curves ----
    curves = np.empty((N, P, len(FRACS), 5))
    for i in range(N):
        for p in range(P):
            curves[i, p] = mf_curves(images[i, p])
    np.savez(f"{args.out_prefix}_sim_curves.npz",
             halo_id=halo_ids, fracs=FRACS, curves=curves,
             pseudo_tsc=tsc,
             columns=np.array(["area", "perimeter", "ncomp", "nholes",
                               "euler"]))
    print(f"sim curves saved -> {args.out_prefix}_sim_curves.npz")

    # ---- TSC predictive power ----
    # drop V0 (== requested fraction by construction, up to sparsity)
    feats = curves[:, :, :, 1:].reshape(N * P, -1)
    y = np.repeat(tsc, P).astype(np.float64)
    groups = np.repeat(np.arange(N), P)

    print("\nOOF R² predicting pseudo-TSC from MF curves "
          f"({feats.shape[1]} features):")
    results = {}
    for name, fn in [
        ("ridge", lambda: Ridge(alpha=10.0)),
        ("xgboost", lambda: XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=0,
            n_jobs=2)),
    ]:
        r2, r2_cl, oof = oof_r2(feats, y, groups, fn)
        results[name] = (r2, r2_cl, oof)
        print(f"  {name:8s}: per-projection {r2:+.3f}   "
              f"cluster-mean {r2_cl:+.3f}")
    print("  (references: tabular XGBoost 0.333, shallow CNN 0.529)")

    # per-feature Spearman correlation with TSC (cluster level, proj-mean)
    cl_feats = curves[:, :, :, 1:].mean(axis=1).reshape(N, -1)
    rho = np.array([stats.spearmanr(cl_feats[:, j], tsc).statistic
                    for j in range(cl_feats.shape[1])])
    jbest = np.argmax(np.abs(rho))
    # ordering matches the (fracs, cols) flattening of the reshape above
    fnames = [f"{c}@f={f:.3f}" for f in FRACS
              for c in ["perimeter", "ncomp", "nholes", "euler"]]
    print(f"\nstrongest single-feature Spearman with TSC: "
          f"{fnames[jbest]}  rho={rho[jbest]:+.3f}")

    # ---- LoTSS first-look comparison ----
    obs_maps, obs_names = load_lotss(args.lotss_glob, size=W)
    obs_curves = None
    if obs_maps:
        obs_curves = np.stack([mf_curves(m) for m in obs_maps])
        np.savez(f"{args.out_prefix}_lotss_curves.npz",
                 fracs=FRACS, curves=obs_curves,
                 names=np.array(obs_names))
        print(f"\nLoTSS maps processed: {len(obs_maps)} "
              f"-> {args.out_prefix}_lotss_curves.npz")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    terc = np.quantile(tsc, [1 / 3, 2 / 3])
    tlabels = [f"TSC<{terc[0]:.1f}", "mid", f"TSC>{terc[1]:.1f}"]
    masks = [tsc < terc[0], (tsc >= terc[0]) & (tsc <= terc[1]),
             tsc > terc[1]]
    sim_mean = curves.mean(axis=1)  # projection-averaged, (N, F, 5)

    for ax, col, name in [(axes[0, 0], 1, "perimeter V1"),
                          (axes[0, 1], 4, "Euler char V2")]:
        for m, lab in zip(masks, tlabels):
            mu = sim_mean[m, :, col].mean(axis=0)
            sd = sim_mean[m, :, col].std(axis=0)
            ax.plot(FRACS, mu, label=lab)
            ax.fill_between(FRACS, mu - sd, mu + sd, alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlabel("area fraction threshold f")
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.set_title(f"sim {name} by TSC tercile")

    ax = axes[1, 0]
    best = max(results, key=lambda k: results[k][0])
    r2, r2_cl, oof = results[best]
    ax.scatter(y, oof, s=4, alpha=0.4)
    lims = [y.min(), y.max()]
    ax.plot(lims, lims, "k:", lw=1)
    ax.set_xlabel("pseudo-TSC [Gyr]")
    ax.set_ylabel("OOF prediction")
    ax.set_title(f"{best} on MF features: R²={r2:+.3f}")

    ax = axes[1, 1]
    if obs_curves is not None:
        for arr, lab, c in [(sim_mean, "sim (352)", "C0"),
                            (obs_curves, f"LoTSS ({len(obs_maps)})", "C1")]:
            mu = arr[:, :, 1].mean(axis=0)
            sd = arr[:, :, 1].std(axis=0)
            ax.plot(FRACS, mu, color=c, label=lab)
            ax.fill_between(FRACS, mu - sd, mu + sd, color=c, alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlabel("area fraction threshold f")
        ax.set_ylabel("perimeter V1")
        ax.legend(fontsize=8)
        ax.set_title("sim vs LoTSS V1 (NO beam/noise matching yet)")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=150)
    print(f"figure saved -> {args.out_prefix}.png")


if __name__ == "__main__":
    main()
