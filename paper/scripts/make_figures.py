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
# Realistic inputs (review F1/F5 fixes). Set in one place so every figure
# uses the same forward-model version.
XRAY_H5 = "xray_fix.h5"
RADIO_INJ = "injected_fix_phys_b1000.h5"
key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()


def save(fig, name):
    path = os.path.join(OUT, f"fig_{name}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------- radio gallery
def radio_gallery():
    with h5py.File(RADIO_INJ) as f:
        mock = f["mock/images"][:, 0, 0]           # realization 0, projection xy
        mh = f["mock/halo_id"][:]
        tsc = f["mock/pseudo_tsc"][:]
        mz = f["mock/z"][:, 0]
        obs = f["obs/images"][:]
        on = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/name"][:]]
        oz = f["obs/z"][:]
    with h5py.File("dataset_nh4_144_512.h5") as f:
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
    with h5py.File(RADIO_INJ) as f:
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
    with h5py.File("dataset_nh4_144_128.h5") as f:
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
    with h5py.File(XRAY_H5) as f:
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
    with h5py.File(XRAY_H5) as f:
        mock, mz, tsc = f["mock/images"][:], f["mock/z"][:], f["mock/pseudo_tsc"][:]
        # review M1: match mocks to a real target by stored identity, redshift
        # and exposure (identity alone does NOT fix redshift: placements with
        # no Chandra counterpart borrow a target's identity and exposure)
        if "mock/target" not in f or "mock/exposure_ks" not in f:
            raise SystemExit("xray_gallery needs mock/target and mock/exposure_ks "
                             "(count-model 'real' products); legacy inputs unsupported")
        mtgt = np.vectorize(lambda v: key(v.decode() if isinstance(v, bytes) else v))(
            f["mock/target"][:])
        okey = [key(v.decode() if isinstance(v, bytes) else v) for v in f["obs/key"][:]]
        mexp = f["mock/exposure_ks"][:] if "mock/exposure_ks" in f else None
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
        # review round 3 (M1): identity AND redshift -- a mock placed at a
        # redshift with no Chandra target borrows a random target's identity
        # and exposure but keeps its own redshift
        zok = np.abs(mz - oz[j]) < 5e-4
        idx = np.argwhere((mtgt == okey[j]) & zok)
        if len(idx) == 0:
            raise SystemExit(f"xray_gallery: no mock matches {t} in identity and z")
        bad = [tuple(ix) for ix in idx if abs(mexp[tuple(ix)] - ot[j]) > 0.5]
        if bad:
            raise SystemExit(f"xray_gallery: {t} mocks with exposure != {ot[j]:.1f} ks")
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
    with h5py.File("relic_catalog_radio_nh4_144.h5") as f:
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


# ---------------------------------------------------------------- phase diagram
def phase():
    """Where the shock cells sit in (n_H, T), by number and by radio weight
    (extract_phase.py: 352 clusters, 144 MHz, no cut, each cluster weighted
    equally)."""
    d = np.load("phase_cells_144.npz")
    ext = [d["nh_edges"][0], d["nh_edges"][-1], d["t_edges"][0], d["t_edges"][-1]]
    fig, ax = plt.subplots(1, 2, figsize=(TEXT, TEXT * 0.36), sharey=True)
    for a, H, title in [(ax[0], d["H_n"], "shock cells (by number)"),
                        (ax[1], d["H_w"], "radio weight, 144 MHz")]:
        with np.errstate(divide="ignore"):
            im = a.imshow(np.log10(H.T), origin="lower", extent=ext, aspect="auto",
                          cmap="viridis", vmin=np.log10(H[H > 0]).max() - 4)
        a.axvline(np.log10(float(d["nh_cut"])), color="w", ls="--", lw=1)
        a.set_title(title)
        a.set_xlabel(r"log $n_{\rm H}$ [cm$^{-3}$]")
        fig.colorbar(im, ax=a, pad=0.01, label="log fraction per bin")
    ax[0].set_ylabel(r"log $kT$ [keV]")
    fa = np.median(d["frac_above"])
    ax[1].text(0.03, 0.97, f"{fa:.0%} of the weight\nat $n_{{\\rm H}} > 10^{{-4}}$",
               transform=ax[1].transAxes, ha="left", va="top", color="k",
               fontsize=7)
    fig.tight_layout()
    save(fig, "phase")
    print(f"  weight above the cut: median {fa:.2f} (16-84% "
          f"{np.percentile(d['frac_above'], 16):.2f}-{np.percentile(d['frac_above'], 84):.2f});"
          f" top-cell share {np.median(d['top_all']):.3f} -> {np.nanmedian(d['top_cut']):.3f};"
          f" weighted Mach {np.median(d['mach_all']):.1f} -> {np.nanmedian(d['mach_cut']):.1f}")
    H = d["H_w"]
    tc = 0.5 * (d["t_edges"][1:] + d["t_edges"][:-1])
    nc = 0.5 * (d["nh_edges"][1:] + d["nh_edges"][:-1])
    print(f"  weight-averaged log n_H {np.sum(H.sum(1) * nc) / H.sum():.2f}, "
          f"log kT {np.sum(H.sum(0) * tc) / H.sum():.2f}")


# ---------------------------------------------------------------- labels
def labels():
    with h5py.File("dataset_nh4_144_128.h5") as f:
        p99, h99 = f["labels/pseudo_tsc"][:], f["meta/halo_id"][:]
    with h5py.File("dataset_nh4_144_s91_128.h5") as f:
        p91 = f["labels/pseudo_tsc"][:]
    with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5") as f:
        cat = dict(zip(f["halo_id"][:], f["tsc_gyr"][:]))
    c99 = np.array([cat.get(h, np.nan) for h in h99])
    fig, ax = plt.subplots(1, 2, figsize=(TEXT * 0.75, TEXT * 0.3))
    bins = np.arange(0, 4.26, 0.25)
    ax[0].hist(np.minimum(p99, 4.1), bins, histtype="step", color=C_SIM, lw=1.2,
               label=f"$z = 0$ ({len(p99)})")
    ax[0].hist(np.minimum(p91, 4.1), bins, histtype="step", color=C_OBS, lw=1.2,
               ls="--", label=f"$z = 0.1$ ({len(p91)})")
    ax[0].set_xlabel("pseudo-TSC [Gyr]")
    ax[0].set_ylabel("clusters")
    ax[0].legend()
    m = np.isfinite(c99)
    ax[1].scatter(c99[m], p99[m], s=4, alpha=0.5, color=C_SIM)
    ax[1].plot([0, 4], [0, 4], "k:", lw=0.8)
    ax[1].set_xlabel("merger-catalogue TSC [Gyr]")
    ax[1].set_ylabel("pseudo-TSC [Gyr]")
    rho = stats.spearmanr(c99[m], p99[m])[0]
    ax[1].set_title(f"Spearman {rho:.3f}", fontsize=7)
    fig.tight_layout()
    save(fig, "labels")
    print(f"  pseudo-TSC at cap (4 Gyr): z=0 {np.mean(p99 >= 4):.0%}, z=0.1 "
          f"{np.mean(p91 >= 4):.0%}; medians {np.median(p99):.2f} / {np.median(p91):.2f}")
    print(f"  pseudo vs catalogue (z=0): Spearman {rho:.3f}, n={m.sum()} "
          f"({(~m).sum()} with no recorded collision)")


# ---------------------------------------------------------------- baselines
def _scalars():
    with h5py.File("dataset_nh4_144_128.h5") as f:
        hid, r500 = f["meta/halo_id"][:], f["meta/r500c_kpc"][:]
        tsc = f["labels/pseudo_tsc"][:].astype(float)
        imgs = f["images"][:]
    with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
        mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
    logm = np.array([mm[h] for h in hid])
    lin = np.log10(np.sinh(imgs.astype(float)).sum(axis=(1, 2, 3)) + 1)
    X = np.column_stack([logm, np.log10(r500), lin, imgs.mean(axis=(1, 2, 3)),
                         (imgs > 0).mean(axis=(1, 2, 3))])
    return tsc, logm, lin, X


def _ridge_oof(X, y):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = X[:, None] if X.ndim == 1 else X
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=0))


def _seed_preds(pattern, key="oof_pred"):
    import glob
    return np.array([np.load(p)[key] for p in sorted(glob.glob(pattern))])


def baselines():
    from sklearn.metrics import r2_score
    tsc, logm, lin, X = _scalars()
    cnn = _seed_preds("cnn_preds/pooled_nh4_144_inner_s*.npz")
    shp = _seed_preds("cnn_preds/pooled_shape144_inner_s*.npz")
    ens, sens = cnn.mean(0), shp.mean(0)
    rows = [
        ("mass", r2_score(tsc, _ridge_oof(logm, tsc)), None, "0.6"),
        ("total flux", r2_score(tsc, _ridge_oof(lin, tsc)), None, "0.6"),
        ("5 scalars", r2_score(tsc, _ridge_oof(X, tsc)), None, "0.6"),
        ("morphology only", np.mean([r2_score(tsc, p) for p in shp]),
         np.std([r2_score(tsc, p) for p in shp], ddof=1), C_OBS),
        ("CNN", np.mean([r2_score(tsc, p) for p in cnn]),
         np.std([r2_score(tsc, p) for p in cnn], ddof=1), C_SIM),
        ("CNN, 5-seed mean", r2_score(tsc, ens), None, C_SIM),
        ("5 scalars + CNN", r2_score(tsc, _ridge_oof(np.column_stack([X, ens]), tsc)),
         None, "k"),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(TEXT, TEXT * 0.32),
                           gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, e, c) in zip(y, rows):
        ax[0].barh(yy, v, xerr=e, color=c, capsize=2, height=0.7)
        ax[0].text(v + (e or 0) + 0.01, yy, f"{v:.3f}", va="center", fontsize=6.5)
    ax[0].set_yticks(y, [r[0] for r in rows])
    ax[0].set_xlabel("out-of-fold $R^2$")
    ax[0].set_xlim(0, 0.65)
    q = np.quantile(logm, [1 / 3, 2 / 3])
    bins = [logm <= q[0], (logm > q[0]) & (logm <= q[1]), logm > q[1]]
    x = np.arange(3)
    sc = _ridge_oof(X, tsc)
    for off, pred, lab, c in [(-0.27, _ridge_oof(logm, tsc), "mass", "0.75"),
                              (0.0, sc, "5 scalars", "0.45"),
                              (0.27, ens, "CNN", C_SIM)]:
        v = [r2_score(tsc[b], pred[b]) for b in bins]
        ax[1].bar(x + off, v, 0.25, color=c, label=lab)
        print(f"  terciles {lab:<10}" + " ".join(f"{u:+.3f}" for u in v))
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(x, ["low", "mid", "high"])
    ax[1].set_xlabel(r"$M_{500}$ tercile")
    ax[1].set_ylabel("$R^2$ within tercile")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    save(fig, "baselines")
    for lab, v, e, _ in rows:
        print(f"  {lab:<18} {v:.3f}" + (f" +- {e:.3f}" if e is not None else ""))


# ---------------------------------------------------------------- true vs predicted
def tsc_pred():
    import glob
    from sklearn.metrics import r2_score
    with h5py.File("dataset_nh4_144_128.h5") as f:
        y_c = f["labels/pseudo_tsc"][:]
    p_c = _seed_preds("cnn_preds/pooled_nh4_144_inner_s*.npz").mean(0)
    js = [np.load(p) for p in sorted(glob.glob("cnn_preds/s9991_joint_s*_preds.npz"))]
    m = js[0]["snap"] == 99
    y_r = js[0]["tsc"][m]
    p_r = np.mean([d["oof_cluster"] for d in js], 0)[m]
    fig, ax = plt.subplots(1, 3, figsize=(TEXT, TEXT * 0.32))
    for a, y, p, lab, c in [(ax[0], y_c, p_c, "clean radio (144 MHz)", C_SIM),
                            (ax[1], y_r, p_r, "realistic radio + X-ray", C_OBS)]:
        a.scatter(y, p, s=4, alpha=0.5, color=c)
        a.plot([0, 4.2], [0, 4.2], "k:", lw=0.8)
        a.set_xlim(-0.1, 4.2), a.set_ylim(-0.1, 4.2)
        a.set_xlabel("true pseudo-TSC [Gyr]")
        a.set_title(f"{lab}: $R^2$ = {r2_score(y, p):.2f}", fontsize=7)
    ax[0].set_ylabel("predicted [Gyr]")
    edges = np.array([0, 0.5, 1, 1.5, 2, 3, 4.01])
    mid = 0.5 * (edges[1:] + edges[:-1])
    for y, p, lab, c, dx in [(y_c, p_c, "clean", C_SIM, -0.04),
                             (y_r, p_r, "realistic", C_OBS, 0.04)]:
        r = p - y
        idx = np.digitize(y, edges) - 1
        med = np.array([np.median(r[idx == k]) for k in range(len(mid))])
        lo = np.array([np.percentile(r[idx == k], 16) for k in range(len(mid))])
        hi = np.array([np.percentile(r[idx == k], 84) for k in range(len(mid))])
        ax[2].errorbar(mid + dx, med, yerr=[med - lo, hi - med], fmt="o-", ms=3,
                       color=c, label=lab, capsize=2, lw=0.9)
        print(f"  {lab}: median residual by bin " + " ".join(f"{v:+.2f}" for v in med))
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel("true pseudo-TSC [Gyr]")
    ax[2].set_ylabel("predicted $-$ true [Gyr]")
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    save(fig, "tsc_pred")
    print(f"  clean R2 {r2_score(y_c, p_c):.3f} (5-seed mean); realistic joint "
          f"(99+91 training, snap-99 clusters) {r2_score(y_r, p_r):.3f}")


# ---------------------------------------------------------------- X-ray depth
def xray_depth():
    import glob
    from sklearn.metrics import r2_score
    ks, mu, sd = [], [], []
    for p in sorted(glob.glob("cnn_preds/xdepth_*ks_s42_preds.npz"),
                    key=lambda s: float(re.search(r"xdepth_([\d.]+)ks", s).group(1))):
        k = float(re.search(r"xdepth_([\d.]+)ks", p).group(1))
        r = [r2_score(np.load(q)["tsc"], np.load(q)["oof_cluster"])
             for q in glob.glob(f"cnn_preds/xdepth_{int(k)}ks_s*_preds.npz")]
        ks.append(k), mu.append(np.mean(r)), sd.append(np.std(r))
        print(f"  {k:>6.0f} ks: R2 {np.mean(r):.3f} +- {np.std(r):.3f} ({len(r)} seeds)")
    real = [r2_score(np.load(q)["tsc"], np.load(q)["oof_cluster"])
            for q in glob.glob("cnn_preds/x144_s*_preds.npz")]
    with h5py.File(XRAY_H5) as f:
        texp = f["obs/exposure_ks"][:]
    fig, ax = plt.subplots(figsize=(COL, COL * 0.75))
    ax.errorbar(ks, mu, yerr=sd, fmt="o-", color=C_SIM, ms=4, capsize=2,
                label="every mock at this exposure")
    ax.axhspan(np.mean(real) - np.std(real), np.mean(real) + np.std(real),
               color=C_OBS, alpha=0.2)
    ax.axhline(np.mean(real), color=C_OBS, lw=1,
               label="each mock at its target's real exposure")
    for t in texp:
        ax.axvline(t, ymin=0, ymax=0.06, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Chandra ACIS-I exposure [ks]")
    ax.set_ylabel("out-of-fold $R^2$")
    ax.legend(fontsize=6, loc="upper left")
    save(fig, "xray_depth")
    print(f"  real-exposure reference {np.mean(real):.3f} +- {np.std(real):.3f}; "
          f"15 targets' exposures {np.sort(texp).round(0)}")


# ---------------------------------------------------------------- P-M relation
def pm_relation():
    """The simulation's own 150 MHz power against mass, two ways: total power
    (all shock cells inside +-4 r500, the quantity radio-relic scaling
    relations use) and the power inside the central Mpc (what the injected
    mocks carry). Same global flux constant for both."""
    with h5py.File(RADIO_INJ) as f:
        lp_c = np.nanmean(f["mock/logp150"][:], axis=(1, 2))
        lm = f["mock/log_m500"][:]
        tsc = f["mock/pseudo_tsc"][:]
        hid = f["mock/halo_id"][:]
        off = float(f.attrs["sim_flux_offset_dex"])
    with h5py.File("dataset_nh4_144_512.h5") as f:
        pos = {h: i for i, h in enumerate(f["meta/halo_id"][:])}
        img = f["images"][:]
    tot = np.sinh(img.astype(np.float64)).sum(axis=(2, 3)).mean(axis=1)
    # zero point arbitrary: the flux constant was fixed on the central Mpc
    lp_t = np.log10(tot[[pos[h] for h in hid]]) + off
    fig, ax = plt.subplots(1, 3, figsize=(TEXT, TEXT * 0.3))
    xx = np.linspace(lm.min(), lm.max(), 10)
    out = []
    for a, lp, lab in [(ax[0], lp_t, r"log total $P_{150}$ + const"),
                       (ax[1], lp_c, r"log $P_{150}$, central Mpc [W Hz$^{-1}$]")]:
        ok = np.isfinite(lp)
        fit = stats.linregress(lm[ok], lp[ok])
        res = lp - (fit.intercept + fit.slope * lm)
        sc = a.scatter(lm, lp, c=np.minimum(tsc, 4), s=4, cmap="plasma_r")
        x0, y0 = np.median(lm[ok]), np.median(lp[ok])
        a.plot(xx, fit.intercept + fit.slope * xx, "k-", lw=1,
               label=f"fit: {fit.slope:.2f}, {np.nanstd(res):.2f} dex")
        a.plot(xx, y0 + 2.83 * (xx - x0), "k--", lw=0.8, label="relics: 2.83")
        a.plot(xx, y0 + 3.55 * (xx - x0), "k:", lw=0.8, label="haloes: 3.55")
        a.set_xlabel(r"log $M_{500}$ [M$_\odot$]")
        a.set_ylabel(lab)
        a.legend(fontsize=5.5, loc="upper left")
        out.append((fit, res))
    fig.colorbar(sc, ax=ax[1], pad=0.01, label="pseudo-TSC [Gyr]")
    res = out[0][1]
    ax[2].scatter(np.minimum(tsc, 4), res, s=4, alpha=0.5, color=C_SIM)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel("pseudo-TSC [Gyr]")
    ax[2].set_ylabel("total-power residual [dex]")
    rho, p = stats.spearmanr(tsc, res)
    ax[2].set_title(f"Spearman {rho:+.2f}", fontsize=7)
    fig.tight_layout()
    save(fig, "pm_relation")
    for name, (fit, r) in zip(("total", "central Mpc"), out):
        print(f"  {name:<12} slope {fit.slope:.2f} +- {fit.stderr:.2f}, scatter "
              f"{np.nanstd(r):.2f} dex, residual vs TSC Spearman "
              f"{stats.spearmanr(tsc, r, nan_policy='omit')[0]:+.2f}")


# ---------------------------------------------------------------- real predictions
def real_preds():
    import glob
    wl = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")

    def ens(pat):
        ds = [np.load(p) for p in sorted(glob.glob(pat))]
        names = [key(n) for n in ds[0]["obs_name"]]
        P = np.array([d["obs_pred"] for d in ds])
        return {n: (P[:, i].mean(), P[:, i].std()) for i, n in enumerate(names)}, \
            dict(zip(names, ds[0]["obs_name"]))

    r, nm = ens("cnn_preds/s9991_sfimg_s*_preds.npz")
    x, _ = ens("cnn_preds/s9991_x_s*_preds.npz")
    both = sorted(set(r) & set(x))
    fig, ax = plt.subplots(figsize=(COL, COL * 0.9))
    lmv = [wl.loc[k, "log_m500c"] if k in wl.index else np.nan for k in both]
    vmin, vmax = np.nanmin(lmv), np.nanmax(lmv)
    for k, m in zip(both, lmv):
        c = plt.cm.viridis((m - vmin) / (vmax - vmin)) if np.isfinite(m) else "0.6"
        ax.errorbar(r[k][0], x[k][0], xerr=r[k][1], yerr=x[k][1], fmt="o", ms=4,
                    color=c, mec="k", mew=0.4, lw=0.7, capsize=0)
        lab = str(nm[k]).replace("_", " ").replace("RXC J1217.6+0339", "RXC J1217")
        ax.annotate(lab, (r[k][0], x[k][0]), fontsize=4.5, xytext=(3, -6 if k in
                    ("A399", "A119", "A2033", "RXCJ1217.6+0339") else 2),
                    textcoords="offset points")
    lim = [0, max(max(v[0] for v in r.values()), max(v[0] for v in x.values())) + 0.3]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("radio prediction [Gyr]")
    ax.set_ylabel("X-ray prediction [Gyr]")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax, pad=0.01, label=r"weak-lensing log $M_{500}$")
    rho, p = stats.spearmanr([r[k][0] for k in both], [x[k][0] for k in both])
    ax.set_title(f"{len(both)} LoVoCCS clusters, Spearman {rho:.2f}", fontsize=7)
    save(fig, "real_preds")
    dr = np.array([r[k][0] - x[k][0] for k in both])
    print(f"  radio - X-ray: median {np.median(dr):+.2f} Gyr, {np.mean(dr > 0):.0%} positive; "
          f"radio median {np.median([r[k][0] for k in both]):.2f}, X-ray median "
          f"{np.median([x[k][0] for k in both]):.2f}")
    print(f"  n={len(both)} rho={rho:.2f} p={p:.3f}; grey = no weak-lensing mass: "
          f"{[str(nm[k]) for k, m in zip(both, lmv) if not np.isfinite(m)]}")


# ---------------------------------------------------------------- observed gallery
def obs_gallery():
    """Real targets with both LoTSS and Chandra, on the models' own inputs:
    radio arcsinh(S/rms) over 1 Mpc; X-ray arcsinh(rate/scale), cropped to
    the same 1 Mpc (the X-ray grid is fixed in physical units)."""
    from astropy.cosmology import Planck15
    from build_xray_realistic import BLOCK, GRID
    with h5py.File(RADIO_INJ) as f:
        rimg = f["obs/images"][:]
        rk = [key(s.decode() if isinstance(s, bytes) else s) for s in f["obs/key"][:]]
        rn = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/name"][:]]
        rz = f["obs/z"][:]
    with h5py.File(XRAY_H5) as f:
        ximg = f["obs/images"][:]
        xk = [key(s.decode() if isinstance(s, bytes) else s) for s in f["obs/key"][:]]
        xt = f["obs/exposure_ks"][:]
    kpc_block = BLOCK * 0.492 * Planck15.kpc_proper_per_arcmin(0.05).value / 60
    h = int(round(500.0 / kpc_block))
    s = np.s_[GRID // 2 - h:GRID // 2 + h, GRID // 2 - h:GRID // 2 + h]
    want = ["A401", "A399", "A1650", "A2443", "A119", "A2033"]
    pick = [k for k in want if k in rk and k in xk][:6]
    fig, ax = plt.subplots(2, len(pick), figsize=(TEXT, TEXT * 0.36))
    for c, k in enumerate(pick):
        i, j = rk.index(k), xk.index(k)
        ax[0, c].imshow(rimg[i], origin="lower", cmap="magma", vmin=-1,
                        vmax=np.arcsinh(30))
        ax[0, c].set_title(f"{rn[i]}, z = {rz[i]:.3f}", fontsize=6.5)
        ax[1, c].imshow(ximg[j][s], origin="lower", cmap="magma", vmin=0,
                        vmax=np.percentile(ximg[:, s[0], s[1]], 99.5))
        ax[1, c].set_title(f"Chandra {xt[j]:.0f} ks", fontsize=6.5)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    ax[0, 0].set_ylabel("LoTSS 144 MHz")
    ax[1, 0].set_ylabel("Chandra 0.1-2 keV")
    fig.tight_layout(h_pad=1.2, w_pad=0.3)
    save(fig, "obs_gallery")
    print(f"  targets {pick}; X-ray crop +-{h} blocks of {kpc_block:.1f} kpc")


FIGS = {"radio_gallery": radio_gallery, "radio_mf": radio_mf, "mf_tsc": mf_tsc,
        "lx_mass": lx_mass, "xray_gallery": xray_gallery, "relic": relic,
        "aug": aug, "phase": phase, "labels": labels, "baselines": baselines,
        "tsc_pred": tsc_pred, "xray_depth": xray_depth, "pm_relation": pm_relation,
        "real_preds": real_preds, "obs_gallery": obs_gallery}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name in (sys.argv[1:] or FIGS):
        print(f"== {name}")
        FIGS[name]()
