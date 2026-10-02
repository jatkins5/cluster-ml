#!/usr/bin/env python3
"""Regenerate the paper figures from the current datasets.

Run from the repository root (via SLURM, like every other script):
    ./venv/bin/python paper/scripts/make_figures.py [name ...]
With no names it makes all of them. Output: paper/figures/fig_<name>.pdf,
and the numbers each figure shows are printed so captions can be checked.

  radio_gallery   clean sim / forward-modelled mock / real LoTSS
  radio_mf        Minkowski functionals of injected mocks vs LoTSS
  mf_tsc          Minkowski functionals of the sim maps vs merger time
  lx_mass         X-ray luminosity and weak-lensing mass, sim vs LoVoCCS
  xray_gallery    real Chandra inputs beside redshift-matched mocks
  relic           relic distance vs merger time (needs relic_catalog_radio_nh4.h5)

Data sources are the current configuration throughout: GroupPos-centred,
n_H < 1e-4 cut radio maps; radio mocks injected into LoTSS DR3 fields with
the simulation's own flux; X-ray mocks at real Chandra depths and target
redshifts, corrected for TNG's L_X excess, with the particle-background-
corrected real data attached.
"""
import os
import re
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.getcwd())

OUT = "paper/figures"
COL, TEXT = 3.32, 6.97            # MNRAS column and text widths, inches
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "savefig.bbox": "tight", "savefig.dpi": 200,
})
C_SIM, C_OBS = "#3b6ea8", "#c2455d"
key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()


def save(fig, name):
    path = os.path.join(OUT, f"fig_{name}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------- radio gallery
def radio_gallery():
    with h5py.File("injected_nh4_simflux.h5") as f:
        mock = f["mock/images"][:, 0, 0]           # realization 0, projection xy
        mh = f["mock/halo_id"][:]
        tsc = f["mock/pseudo_tsc"][:]
        mz = f["mock/z"][:, 0]
        obs = f["obs/images"][:]
        on = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/name"][:]]
        oz = f["obs/z"][:]
    with h5py.File("dataset_nh4_512.h5") as f:
        simh = f["meta/halo_id"][:]
        r500 = f["meta/r500c_kpc"][:]
        order = {h: i for i, h in enumerate(simh)}
        # four clusters spanning merger time
        pick = [int(np.argmin(np.abs(tsc - t))) for t in (0.2, 0.8, 1.8, 4.0)]
        clean = [f["images"][order[mh[i]], 0] for i in pick]
    targets = ["A399", "A2443", "A1650", "A2033"]
    oi = [on.index(t) for t in targets if t in on]
    oi += [k for k in np.argsort(-obs.max(axis=(1, 2))) if k not in oi][:4 - len(oi)]
    print("  real targets shown:", [on[k] for k in oi])

    fig, ax = plt.subplots(3, 4, figsize=(TEXT, TEXT * 0.78))
    vmax = np.arcsinh(30.0)
    for c, i in enumerate(pick):
        H = clean[c].shape[0]
        half = H // 4                                          # +-2 r500
        crop = clean[c][H // 2 - half:H // 2 + half, H // 2 - half:H // 2 + half]
        ax[0, c].imshow(crop, origin="lower", cmap="magma")
        ax[0, c].set_title(f"TSC = {tsc[i]:.1f} Gyr", fontsize=7)
        print(f"  column {c}: halo {mh[i]}, TSC {tsc[i]:.2f}, "
              f"r500 {r500[order[mh[i]]]:.0f} kpc")
        ax[1, c].imshow(mock[i], origin="lower", cmap="magma",
                        vmin=-1, vmax=vmax)
        ax[1, c].set_title(f"mock, z = {mz[i]:.3f}", fontsize=7)
    for c, k in enumerate(oi):
        im = ax[2, c].imshow(obs[k], origin="lower", cmap="magma",
                             vmin=-1, vmax=vmax)
        ax[2, c].set_title(f"{on[k]}, z = {oz[k]:.3f}", fontsize=7)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    ax[0, 0].set_ylabel("simulation")
    ax[1, 0].set_ylabel("mock LoTSS")
    ax[2, 0].set_ylabel("real LoTSS")
    cb = fig.colorbar(im, ax=ax[1:, :], shrink=0.8, pad=0.01)
    cb.set_label(r"arcsinh(S / $\sigma_{\rm rms}$)")
    save(fig, "radio_gallery")


# ---------------------------------------------------------------- radio MFs
def radio_mf():
    from minkowski_functionals import binary_mfs
    ks = np.geomspace(1.0, 20.0, 10)
    with h5py.File("injected_nh4_simflux.h5") as f:
        mock = f["mock/images"][:]
        obs = f["obs/images"][:]
    mock = mock.reshape(-1, *mock.shape[-2:])
    thr = np.arcsinh(ks)                       # images are arcsinh(map/rms)
    M = np.array([[binary_mfs(im > t) for t in thr] for im in mock])
    O = np.array([[binary_mfs(im > t) for t in thr] for im in obs])
    names = ["area fraction", "perimeter", "components", "holes",
             "Euler characteristic"]
    show = [0, 1, 2, 4]
    fig, ax = plt.subplots(2, 4, figsize=(TEXT, TEXT * 0.45),
                           gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
    print("  separation (obs median - mock median) / mock sd:")
    for c, j in enumerate(show):
        lo, med, hi = np.percentile(M[:, :, j], [16, 50, 84], axis=0)
        ax[0, c].fill_between(ks, lo, hi, color=C_SIM, alpha=0.25)
        ax[0, c].plot(ks, med, color=C_SIM, label=f"mocks ({len(M)})")
        olo, omed, ohi = np.percentile(O[:, :, j], [16, 50, 84], axis=0)
        ax[0, c].errorbar(ks, omed, yerr=[omed - olo, ohi - omed], fmt="o",
                          ms=3, color=C_OBS, label=f"LoTSS ({len(O)})")
        ax[0, c].set_title(names[j])
        sep = (np.median(O[:, :, j], 0) - med) / M[:, :, j].std(0)
        ax[1, c].axhspan(-1, 1, color="0.9")
        ax[1, c].plot(ks, sep, "o-", ms=3, color="k")
        ax[1, c].axhline(0, color="0.5", lw=0.6)
        ax[1, c].set_xscale("log")
        ax[1, c].set_xlabel(r"threshold [$\sigma$]")
        print(f"    {names[j]:<22}" + " ".join(f"{v:+.2f}" for v in sep))
    ax[0, 0].legend()
    ax[1, 0].set_ylabel("separation\n[mock s.d.]")
    fig.align_ylabels(ax[:, 0])
    fig.tight_layout()
    save(fig, "radio_mf")


# ---------------------------------------------------------------- MF vs TSC
def mf_tsc():
    from minkowski_functionals import FRACS, mf_curves, oof_r2
    from xgboost import XGBRegressor
    with h5py.File("dataset_nh4_128.h5") as f:
        images = f["images"][:]
        tsc = f["labels/pseudo_tsc"][:]
    N, P = images.shape[:2]
    curves = np.array([[mf_curves(images[i, p]) for p in range(P)]
                       for i in range(N)])
    feats = curves[:, :, :, 1:].reshape(N * P, -1)
    y = np.repeat(tsc, P).astype(float)
    groups = np.repeat(np.arange(N), P)
    r2, r2_cl, oof = oof_r2(feats, y, groups, lambda: XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=0, n_jobs=4))
    print(f"  XGBoost on MF curves: per-projection R2 {r2:+.3f}, "
          f"cluster-mean {r2_cl:+.3f}")
    cm = curves.mean(axis=1)
    rho = stats.spearmanr(cm[:, -1, 2], tsc)[0]
    print(f"  components at f={FRACS[-1]:.2f} vs TSC: Spearman {rho:+.2f}")

    terc = np.quantile(tsc, [1 / 3, 2 / 3])
    masks = [tsc < terc[0], (tsc >= terc[0]) & (tsc <= terc[1]), tsc > terc[1]]
    labs = [f"TSC < {terc[0]:.1f} Gyr", "middle", f"TSC > {terc[1]:.1f} Gyr"]
    cols = ["#c2455d", "#8a8a8a", "#3b6ea8"]
    fig, ax = plt.subplots(1, 3, figsize=(TEXT, TEXT * 0.3))
    for a, j, name in [(ax[0], 2, "connected components"),
                       (ax[1], 4, "Euler characteristic")]:
        for m, lab, c in zip(masks, labs, cols):
            mu, sd = cm[m, :, j].mean(0), cm[m, :, j].std(0)
            a.plot(FRACS, mu, color=c, label=lab)
            a.fill_between(FRACS, mu - sd, mu + sd, color=c, alpha=0.15)
        a.set_xscale("log")
        a.set_xlabel("area-fraction threshold")
        a.set_ylabel(name)
    ax[0].legend()
    ug = np.arange(N)
    pm = np.array([oof[groups == g].mean() for g in ug])
    ax[2].scatter(tsc, pm, s=4, alpha=0.5, color=C_SIM)
    ax[2].plot([0, 4], [0, 4], "k:", lw=0.8)
    ax[2].set_xlabel("pseudo-TSC [Gyr]")
    ax[2].set_ylabel("predicted from topology [Gyr]")
    ax[2].set_title(f"$R^2$ = {r2_cl:.2f} (cluster mean)")
    fig.tight_layout()
    save(fig, "mf_tsc")


# ---------------------------------------------------------------- L_X and mass
def lx_mass():
    with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
        sim_lx = f["xray_0.5-2.0kev"][:] + np.log10(1.65)     # to ~0.1-2.4 keV
        sim_m = f["mhalo_500c"][:]
    tl = pd.read_csv("LoVoCCS_target_list - lovoccs.csv")
    obs_lx = np.log10(pd.to_numeric(tl["lx"], errors="coerce").dropna())
    wl = pd.read_csv("lovoccs_wl_masses.csv")

    # mass-matched X-ray brightness, real vs corrected mocks
    from build_xray_realistic import GRID, aperture_blocks
    with h5py.File("xray_real_placed_lx.h5") as f:
        a = float(f.attrs["stretch_scale"])
        mimg, mz = f["mock/images"][:], f["mock/z"][:]
        mh = f["mock/halo_id"][:]
        oimg, oz = f["obs/images"][:], f["obs/z"][:]
        on = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/name"][:]]
    with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
        mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
    yy, xx = np.mgrid[:GRID, :GRID]
    rr = np.hypot(yy - GRID / 2 + 0.5, xx - GRID / 2 + 0.5)
    tot = lambda im, z: (np.sinh(im.astype(float)) * a)[rr <= aperture_blocks(z)].sum()
    mrate = np.array([[tot(mimg[i, r, 0], mz[i, r]) for r in range(mimg.shape[1])]
                      for i in range(len(mh))])
    mlog = np.array([mm[h] for h in mh])
    wlk = wl.set_index("key")
    pts = []
    for j, n in enumerate(on):
        k = key(n)
        if k not in wlk.index:
            continue
        lm = float(wlk.loc[k, "log_m500c"])
        sel = (np.abs(mz - oz[j]) < 0.02) & (np.abs(mlog - lm) < 0.15)[:, None]
        if sel.sum() < 5:
            continue
        pts.append((lm, tot(oimg[j], oz[j]) / np.median(mrate[sel]),
                    float(wlk.loc[k, "frac_err"])))
    pts = np.array(pts)

    fig, ax = plt.subplots(1, 3, figsize=(TEXT, TEXT * 0.3))
    for v, lab, c in [(sim_lx, f"TNG-Cluster ({len(sim_lx)})", C_SIM),
                      (obs_lx, f"LoVoCCS ({len(obs_lx)})", C_OBS)]:
        v = np.sort(v)
        ax[0].step(v, np.arange(1, len(v) + 1) / len(v), color=c, label=lab)
    ax[0].set_xlabel(r"log $L_X$ (0.1-2.4 keV) [erg s$^{-1}$]")
    ax[0].set_ylabel("cumulative fraction")
    ax[0].legend()
    for v, lab, c, ls in [(sim_m, "TNG-Cluster", C_SIM, "-"),
                          (wl["log_m500c"], f"LoVoCCS WL ({len(wl)})", C_OBS, "-"),
                          (wl.loc[wl.has_lotss, "log_m500c"],
                           f"LOFAR targets ({int(wl.has_lotss.sum())})", C_OBS, "--")]:
        v = np.sort(np.asarray(v))
        ax[1].step(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, label=lab)
    ax[1].set_xlabel(r"log $M_{500}$ [M$_\odot$]")
    ax[1].legend()
    ax[2].errorbar(pts[:, 0], pts[:, 1], xerr=pts[:, 2] / np.log(10), fmt="o",
                   ms=3, color=C_OBS, lw=0.8)
    ax[2].axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"log $M_{500}$ (weak lensing) [M$_\odot$]")
    ax[2].set_ylabel("real / mock X-ray rate")
    fig.tight_layout()
    save(fig, "lx_mass")
    shift = np.median(sim_lx) - np.median(obs_lx)
    s = stats.linregress(pts[:, 0], np.log10(pts[:, 1]))
    print(f"  sim - obs median log L_X: {shift:+.2f} dex (x{10 ** shift:.1f}); "
          f"mass-matched real/mock median {np.median(pts[:, 1]):.2f}, slope "
          f"{s.slope:+.2f} +- {s.stderr:.2f} (n={len(pts)})")


# ---------------------------------------------------------------- X-ray gallery
def xray_gallery():
    with h5py.File("xray_real_placed_lx.h5") as f:
        mock, mz, tsc = f["mock/images"][:], f["mock/z"][:], f["mock/pseudo_tsc"][:]
        obs, oz, ot = f["obs/images"][:], f["obs/z"][:], f["obs/exposure_ks"][:]
        on = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/name"][:]]
    targets = [t for t in ["A401", "A1650", "A119", "A2443", "A1307"] if t in on]
    targets += [n for n in on if n not in targets][:5 - len(targets)]
    print("  real targets shown:", targets)
    rng = np.random.default_rng(1)
    # crop to the largest common aperture (everything outside it is zero)
    lit = np.argwhere((obs != 0).any(axis=0))
    (y0, x0), (y1, x1) = lit.min(0), lit.max(0) + 1
    s = np.s_[max(y0 - 2, 0):y1 + 2, max(x0 - 2, 0):x1 + 2]
    fig, ax = plt.subplots(len(targets), 3, figsize=(COL * 1.2, COL * 2.0))
    vmax = np.percentile(obs, 99.7)
    for r, t in enumerate(targets):
        j = on.index(t)
        ax[r, 0].imshow(obs[j][s], origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax[r, 0].set_title(f"{t}, z = {oz[j]:.3f}, {ot[j]:.0f} ks", fontsize=6.5)
        idx = np.argwhere(np.abs(mz - oz[j]) < 0.005)
        for c in (1, 2):
            i, rr_ = idx[rng.integers(len(idx))]
            ax[r, c].imshow(mock[i, rr_, 0][s], origin="lower", cmap="magma",
                            vmin=0, vmax=vmax)
            ax[r, c].set_title(f"mock, TSC {tsc[i]:.1f} Gyr", fontsize=6.5)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    save(fig, "xray_gallery")


# ---------------------------------------------------------------- relics
def relic():
    """Outermost radio peak distance vs merger-catalogue TSC, per projection,
    against the constant-speed shock expectation (1500 km/s, as in the
    Lee et al. comparison of plot_relic_validation_v3.py)."""
    with h5py.File("relic_catalog_radio_nh4.h5") as f:
        d = f["d_max"][:] / 1000.0                # (N, 3) Mpc, NaN if none
        mt = f["merger_tsc"][:]
        mr = f["mass_ratio"][:]
        n = f["n_relics"][:]
        src = f.attrs["source"]
    with h5py.File(src) as f:
        ext = float(f.attrs["extent_r500"])
    x = d.ravel()
    y = np.repeat(mt, 3)
    m2 = np.repeat(mr, 3)
    act = np.isfinite(x) & np.isfinite(y) & (y <= 2.0)
    sig = act & (m2 >= 0.1)
    dbl = sig & (n.ravel() >= 2)
    print(f"  peaks per projection: median {np.median(n):.0f}, "
          f"{(n >= 2).mean():.0%} have two or more")
    fig, ax = plt.subplots(figsize=(COL, COL * 0.8))
    ax.scatter(y[act & ~sig], x[act & ~sig], s=3, color="0.7",
               label="TSC $\\leq$ 2 Gyr")
    ax.scatter(y[sig], x[sig], s=5, color=C_SIM, label="mass ratio $\\geq$ 0.1")
    t = np.linspace(0, 2.0, 50)
    ax.plot(t, 1500 * 1.0227e-3 * t, color=C_OBS, ls="--",   # km/s -> Mpc/Gyr
            label="1500 km s$^{-1}$ shock")
    ax.set_xlabel("time since collision [Gyr]")
    ax.set_ylabel("outermost radio peak distance [Mpc]")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0, np.nanmax(x[act]) * 1.05)
    ax.legend()
    save(fig, "relic")
    for lab, s_ in [("active (TSC <= 2)", act), ("mass ratio >= 0.1", sig),
                    ("two or more peaks", dbl)]:
        if s_.sum() > 5:
            r, p = stats.spearmanr(x[s_], y[s_])
            print(f"  {lab:<20} n={s_.sum():4d}  Spearman {r:+.3f} (p={p:.2g})")
    print(f"  source {src}, image half-width {ext} r500")


# ---------------------------------------------------------------- augmentation
def aug():
    """Diffusion-sample augmentation vs none, 5-fold OOF with inner-split
    checkpoint selection, paired by seed (train_cnn_aug_oof.py)."""
    import glob
    r2 = lambda p, y: 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    bins = [("all", lambda y: y >= 0), ("TSC $\\leq$ 1 Gyr", lambda y: y <= 1.0),
            ("1-2 Gyr", lambda y: (y > 1.0) & (y <= 2.0)),
            ("> 2 Gyr", lambda y: y > 2.0)]
    # within a narrow bin R^2 is dominated by the bin's tiny label variance,
    # so the bins are shown as RMSE; overall R^2 is printed for the caption
    rmse = lambda p, y: np.sqrt(np.mean((y - p) ** 2))
    res, r2s = {}, {}
    for tag in ("baseline", "aug"):
        rows, rr = [], []
        for p in sorted(glob.glob(f"cnn_aug_oof_128_inner/oof_{tag}_s*.npz")):
            d = np.load(p)
            rows.append([rmse(d["yhat"][m(d["y"])], d["y"][m(d["y"])])
                         for _, m in bins])
            rr.append(r2(d["yhat"], d["y"]))
        res[tag], r2s[tag] = np.array(rows), np.array(rr)
    n = min(len(res["baseline"]), len(res["aug"]))
    b, a = res["baseline"][:n], res["aug"][:n]
    dr = r2s["aug"][:n] - r2s["baseline"][:n]
    print(f"  overall R2: base {r2s['baseline'][:n].mean():.3f}  aug "
          f"{r2s['aug'][:n].mean():.3f}  delta {dr.mean():+.3f} +- "
          f"{dr.std(ddof=1) if n > 1 else np.nan:.3f}")
    fig, ax = plt.subplots(figsize=(COL, COL * 0.75))
    x = np.arange(len(bins))
    for off, v, lab, c in [(-0.17, b, "no augmentation", "0.55"),
                           (0.17, a, "with diffusion samples", C_SIM)]:
        ax.bar(x + off, v.mean(0), 0.32, yerr=v.std(0) if n > 1 else None,
               color=c, label=lab, capsize=2)
        for s in range(n):
            ax.plot(x + off, v[s], "k.", ms=2)
    ax.set_xticks(x, [lab for lab, _ in bins])
    ax.set_ylabel("out-of-fold RMSE [Gyr]")
    ax.legend(loc="upper left")
    save(fig, "aug")
    print(f"  {n} paired seeds")
    for k, (lab, _) in enumerate(bins):
        dlt = a[:, k] - b[:, k]
        print(f"  {lab:<14} base {b[:, k].mean():+.3f}  aug {a[:, k].mean():+.3f}  "
              f"delta {dlt.mean():+.3f} +- {dlt.std(ddof=1) if n > 1 else np.nan:.3f}")


FIGS = {"radio_gallery": radio_gallery, "radio_mf": radio_mf, "mf_tsc": mf_tsc,
        "lx_mass": lx_mass, "xray_gallery": xray_gallery, "relic": relic,
        "aug": aug}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name in (sys.argv[1:] or FIGS):
        print(f"== {name}")
        FIGS[name]()
