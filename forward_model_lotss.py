"""Forward-model TNG-Cluster radio maps into mock LoTSS observations.

Pipeline per sim cluster (paired with a random LoTSS-covered LoVoCCS
target for redshift + noise):
1. Flux anchor: total 150 MHz power from the Cuciti et al. 2023
   P150-M500 relation (log10(P/10^24.5 W/Hz) = 1.1 + 3.55 log10(M500 /
   10^14.9 Msun), sigma_raw ~ 0.35 dex, optionally sampled).
2. Crop the central BOX kpc of the 512px (+-4 r500) map, resample to
   the common 128px grid, normalize the box flux to S150(z).
3. Convolve with the 9" LoTSS restoring beam (FITS header BMAJ) scaled
   to kpc at the assigned z; convert Jy/px -> Jy/beam.
4. Add Gaussian noise at the paired target's measured rms.

Obs side: same BOX kpc crop of each LoTSS cutout at the target's own
redshift, resampled to the same grid; rms via sigma-clipped std.

Outputs: Minkowski functional curves at k*sigma thresholds for mocks
and obs, sim-vs-obs comparison + KS tests, and a check that the
MF -> pseudo-TSC signal survives observational degradation.

Known caveats (first pass): no interferometric large-scale flux loss,
no point sources in mocks (obs fields contain AGN), sim native pixel
(11-30 kpc) can exceed the beam at low z.
"""
import argparse
import glob
import os
import re

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy import ndimage, stats
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from minkowski_functionals import binary_mfs

COSMO = FlatLambdaCDM(H0=67.74, Om0=0.3089)  # TNG cosmology
ARCSEC = np.pi / 180.0 / 3600.0

BEAM_FWHM_ARCSEC = 9.0      # BMAJ=BMIN=0.0025 deg in the DR3 cutouts
OBS_PIX_ARCSEC = 1.5
GRID = 128
KSIGMA = np.array([2, 3, 5, 8, 13, 21, 34, 55], dtype=float)
SPEC_INDEX = 1.2            # S ~ nu^-alpha for the k-correction

# Cuciti et al. 2023 (A&A 680, A30), BCES Y|X
P150_LOGP0 = 24.5 + 1.1
P150_SLOPE = 3.55
P150_LOGM_PIVOT = 14.9
P150_SCATTER_DEX = 0.35


def kpc_per_arcsec(z):
    return COSMO.angular_diameter_distance(z).to("kpc").value * ARCSEC


def total_flux_jy(log_m500, z, rng=None):
    logp = P150_LOGP0 + P150_SLOPE * (log_m500 - P150_LOGM_PIVOT)
    if rng is not None:
        logp += rng.normal(0.0, P150_SCATTER_DEX)
    dl_m = COSMO.luminosity_distance(z).to("m").value
    s = 10.0 ** logp * (1 + z) ** (1 - SPEC_INDEX) / (4 * np.pi * dl_m**2)
    return s / 1e-26  # W/m^2/Hz -> Jy


def sigma_clipped_rms(img, nsig=3.0, iters=5):
    x = img[np.isfinite(img)].ravel()
    for _ in range(iters):
        x = x[np.abs(x - np.median(x)) < nsig * x.std()]
    return x.std()


def crop_resample(img, half_px, grid=GRID):
    c0, c1 = img.shape[0] // 2, img.shape[1] // 2
    half_px = int(round(half_px))
    if half_px < 4 or half_px > c0 or half_px > c1:
        return None
    cut = img[c0 - half_px:c0 + half_px, c1 - half_px:c1 + half_px]
    return ndimage.zoom(cut, grid / cut.shape[0], order=1)


def mf_ksigma(img, rms):
    out = np.empty((len(KSIGMA), 5))
    for i, k in enumerate(KSIGMA):
        out[i] = binary_mfs(img > k * rms)
    return out


def make_mock(sim_map, sim_px_kpc, log_m500, z, rms_jyb, box_kpc, rng):
    half_px = box_kpc / 2.0 / sim_px_kpc
    cut = crop_resample(sim_map.astype(np.float64), half_px)
    if cut is None or cut.sum() <= 0:
        return None
    px_kpc = box_kpc / GRID
    s_tot = total_flux_jy(log_m500, z, rng)
    cut *= s_tot / cut.sum()                       # Jy per grid pixel

    kpas = kpc_per_arcsec(z)
    fwhm_px = BEAM_FWHM_ARCSEC * kpas / px_kpc
    cut = ndimage.gaussian_filter(cut, fwhm_px / 2.355, mode="constant")
    beam_area_px = 1.1331 * fwhm_px**2            # pi/(4 ln2) FWHM^2
    cut *= beam_area_px                            # Jy/px -> Jy/beam
    cut += rng.normal(0.0, rms_jyb, cut.shape)
    return cut


def load_obs(lotss_glob, targets_csv, box_kpc):
    df = pd.read_csv(targets_csv)
    df["key"] = df["name"].astype(str).str.replace(" ", "")
    df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
    zmap = dict(zip(df["key"], df["redshift"]))

    obs = []
    for path in sorted(glob.glob(lotss_glob)):
        key = re.sub(r"^lotss_|\.fits$", "", os.path.basename(path))
        z = zmap.get(key, np.nan)
        if not np.isfinite(z):
            print(f"  skip {key}: no redshift in target list")
            continue
        with fits.open(path) as hdul:
            img = np.squeeze(hdul[0].data).astype(np.float64)
        img = np.nan_to_num(img, nan=0.0)
        kpas = kpc_per_arcsec(z)
        half_px = box_kpc / 2.0 / (OBS_PIX_ARCSEC * kpas)
        cut = crop_resample(img, half_px)
        if cut is None:
            print(f"  skip {key}: z={z:.3f}, cutout smaller than "
                  f"{box_kpc} kpc box")
            continue
        rms = sigma_clipped_rms(cut)
        if not np.isfinite(rms) or rms <= 0:
            print(f"  skip {key}: degenerate rms (blank/edge cutout)")
            continue
        obs.append(dict(name=key, z=z, rms=rms, map=cut))
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_512.h5")
    ap.add_argument("--lotss-glob",
                    default=os.path.expanduser(
                        "~/data/cluster-ml/lotss_images/*.fits"))
    ap.add_argument("--targets-csv",
                    default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--catalog", default="Radio_Data/TNG-Cluster_Catalog.hdf5")
    ap.add_argument("--box-kpc", type=float, default=1000.0)
    ap.add_argument("--no-scatter", action="store_true",
                    help="disable the 0.35 dex P150-M500 scatter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="forward_lotss")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- observations ----
    print("processing LoTSS cutouts...")
    obs = load_obs(args.lotss_glob, args.targets_csv, args.box_kpc)
    print(f"usable obs maps: {len(obs)}")
    for o in obs:
        print(f"  {o['name']:12s} z={o['z']:.4f} rms={o['rms']*1e3:.3f} mJy/beam")
    obs_mfs = np.stack([mf_ksigma(o["map"], o["rms"]) for o in obs])

    # ---- sims ----
    with h5py.File(args.dataset, "r") as f:
        images = f["images"][:]
        halo_ids = f["meta/halo_id"][:]
        r500_kpc = f["meta/r500c_kpc"][:]
        tsc = f["labels/pseudo_tsc"][:]
    with h5py.File(args.catalog, "r") as f:
        cat_ids = f["haloID"][:]
        log_m500 = f["mhalo_500c"][:]
    m500_map = dict(zip(cat_ids, log_m500))
    N, P, H, _ = images.shape
    print(f"\nsim maps: {N} x {P} proj, {H}px, +-4 r500")

    scatter_rng = None if args.no_scatter else rng
    pair_idx = rng.integers(0, len(obs), size=N)   # per cluster, not proj
    mock_mfs = np.full((N, P, len(KSIGMA), 5), np.nan)
    mock_z = np.array([obs[j]["z"] for j in pair_idx])
    mock_rms = np.array([obs[j]["rms"] for j in pair_idx])
    example = None
    for i in range(N):
        px_kpc = 8.0 * r500_kpc[i] / H
        for p in range(P):
            mock = make_mock(images[i, p], px_kpc, m500_map[halo_ids[i]],
                             mock_z[i], mock_rms[i], args.box_kpc, rng)
            if mock is None:
                continue
            mock_mfs[i, p] = mf_ksigma(mock, mock_rms[i])
            if example is None:
                example = mock
    ok = np.isfinite(mock_mfs[:, 0, 0, 0])
    print(f"mocks built for {ok.sum()}/{N} clusters")

    np.savez(f"{args.out_prefix}_mfs.npz",
             ksigma=KSIGMA, mock_mfs=mock_mfs, obs_mfs=obs_mfs,
             halo_id=halo_ids, pseudo_tsc=tsc, mock_z=mock_z,
             mock_rms=mock_rms,
             obs_names=np.array([o["name"] for o in obs]),
             obs_z=np.array([o["z"] for o in obs]),
             columns=np.array(["area", "perimeter", "ncomp", "nholes",
                               "euler"]))

    # ---- sim vs obs KS tests on MF summaries ----
    print("\nsim-vs-obs KS tests (projection-averaged mocks):")
    mock_mean = np.nanmean(mock_mfs, axis=1)       # (N, K, 5)
    for k_idx, k in [(1, 3), (3, 8)]:
        for col, cname in [(0, "area"), (1, "perimeter"), (2, "ncomp")]:
            a = mock_mean[ok, k_idx, col]
            b = obs_mfs[:, k_idx, col]
            ks = stats.ks_2samp(a, b)
            print(f"  {cname}@{k}sigma: sim med={np.median(a):.4g} "
                  f"obs med={np.median(b):.4g}  "
                  f"KS D={ks.statistic:.3f} p={ks.pvalue:.3g}")

    # ---- does MF->TSC survive degradation? ----
    feats = mock_mfs[:, :, :, :].reshape(N * P, -1)
    y = np.repeat(tsc, P)
    groups = np.repeat(np.arange(N), P)
    good = np.isfinite(feats).all(axis=1) & np.isfinite(y)
    feats, y, groups = feats[good], y[good], groups[good]
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(feats, y, groups):
        m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=0, n_jobs=2)
        m.fit(feats[tr], y[tr])
        oof[te] = m.predict(feats[te])
    r2 = 1 - np.sum((y - oof) ** 2) / np.sum((y - y.mean()) ** 2)
    ug = np.unique(groups)
    ym = np.array([y[groups == g].mean() for g in ug])
    pm = np.array([oof[groups == g].mean() for g in ug])
    r2_cl = 1 - np.sum((ym - pm) ** 2) / np.sum((ym - ym.mean()) ** 2)
    print(f"\nmock-MF -> pseudo-TSC OOF R²: per-proj {r2:+.3f}, "
          f"cluster-mean {r2_cl:+.3f}  (clean maps: 0.364 / 0.454)")

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    j = int(pair_idx[0]) if ok[0] else 0
    for ax, img, rms, title in [
        (axes[0, 0], obs[0]["map"], obs[0]["rms"],
         f"LoTSS {obs[0]['name']} (z={obs[0]['z']:.3f})"),
        (axes[0, 1], example, mock_rms[np.argmax(ok)],
         "mock (first built cluster)"),
    ]:
        im = ax.imshow(img / rms, vmin=-2, vmax=15, cmap="viridis",
                       origin="lower")
        plt.colorbar(im, ax=ax, label="S/N")
        ax.set_title(title)

    ax = axes[0, 2]
    for arr, lab, c in [(mock_mean[ok], f"mock ({ok.sum()})", "C0"),
                        (obs_mfs, f"LoTSS ({len(obs)})", "C1")]:
        mu, sd = arr[:, :, 1].mean(0), arr[:, :, 1].std(0)
        ax.plot(KSIGMA, mu, color=c, label=lab)
        ax.fill_between(KSIGMA, mu - sd, mu + sd, color=c, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("threshold [k sigma]")
    ax.set_ylabel("perimeter V1")
    ax.legend(fontsize=8)
    ax.set_title("V1 vs threshold")

    ax = axes[1, 0]
    for arr, lab, c in [(mock_mean[ok], "mock", "C0"),
                        (obs_mfs, "LoTSS", "C1")]:
        mu, sd = arr[:, :, 4].mean(0), arr[:, :, 4].std(0)
        ax.plot(KSIGMA, mu, color=c, label=lab)
        ax.fill_between(KSIGMA, mu - sd, mu + sd, color=c, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("threshold [k sigma]")
    ax.set_ylabel("Euler char V2")
    ax.legend(fontsize=8)
    ax.set_title("V2 vs threshold")

    ax = axes[1, 1]
    bins = np.linspace(0, max(mock_mean[ok, 1, 2].max(),
                              obs_mfs[:, 1, 2].max()) + 1, 20)
    ax.hist(mock_mean[ok, 1, 2], bins=bins, alpha=0.5, density=True,
            label="mock")
    ax.hist(obs_mfs[:, 1, 2], bins=bins, histtype="step", lw=2,
            density=True, color="C1", label="LoTSS")
    ax.set_xlabel("n components @ 3 sigma")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("fragmentation at 3 sigma")

    ax = axes[1, 2]
    ax.scatter(y, oof, s=4, alpha=0.4)
    lims = [y.min(), y.max()]
    ax.plot(lims, lims, "k:", lw=1)
    ax.set_xlabel("pseudo-TSC [Gyr]")
    ax.set_ylabel("OOF prediction (mock MFs)")
    ax.set_title(f"TSC from degraded maps: R²={r2:+.3f}")

    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=150)
    print(f"figure saved -> {args.out_prefix}.png")


if __name__ == "__main__":
    main()
