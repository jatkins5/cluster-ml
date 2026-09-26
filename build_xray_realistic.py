#!/usr/bin/env python3
"""Realistic-depth X-ray mocks: Chandra ACIS-I at the exposures real clusters get.

The existing mocks (Chuiyang's pyxsim -> soxs pipeline, read-only) already
model the ACIS-I response, PSF, dither, chip layout, Poisson statistics and
the particle background -- but at a 2 Ms exposure with the sky backgrounds
switched off (foreground=False, ptsrc_bkgnd=False). Real Chandra coverage of
our 26 LOFAR targets is a median 50 ks (range 5-663, query_xray_archive.py),
so the mocks are ~40x deeper than a typical real observation and carry no
background AGN or Galactic foreground at all.

Two facts make a realistic version cheap:

  * The images are photon counts divided by a constant: every pixel is an
    integer multiple of 0.00455 (93% exactly, the rest at exposure-map
    edges), so counts are recoverable.
  * Thinning Poisson counts -- keeping each photon with probability p -- is
    exactly Poisson with mean scaled by p, so Binomial(N_2Ms, t / 2 Ms) *is* a
    t-ks observation of the same source and particle background. Thinning
    commutes with summation, so it can be done on the 128px block-summed
    counts rather than on 4880^2 images, which makes every depth nearly free.

The missing sky background is simulated with soxs itself (Galactic
foreground + unresolved CXB + a random resolved point-source population),
with the particle background off since the mocks already contain it, on the
same instrument, aimpoint, band (0.1-2 keV) and image grid, at 2 Ms so it can
be thinned identically. Each (cluster, realization) draws a different
background realization.

Stages:
  counts   recover 2 Ms block counts from dataset_xray_128_orig.h5
  bkg      make K soxs sky-background realizations, block-summed
  build    thin to a depth (fixed ks, or drawn from the real archive
           exposures) and write a train_cnn_mock.py-compatible h5

Known limits, stated rather than hidden: every cluster stays at z=0.05 (the
mocks were made there; the targets span 0.035-0.12), and the ACIS-I square
covers only about +-500 kpc, i.e. ~0.3-0.6 r500, not the +-1 r500 the old
README describes.
"""
import argparse
import glob
import os

import h5py
import numpy as np

QUANTUM = 0.00455          # image units per detected count (measured)
Z_MOCK = 0.05              # every mock was observed at this redshift
# Radius, at z=0.05, of the circle safely inside the ACIS-I field around the
# cluster (the square is 16.9' across and the cluster sits at the aimpoint on
# I3, slightly off-centre; measured coverage falls to ~half by 6-9').
APERTURE_ARCMIN = 8.0
T_MOCK_KS = 2000.0         # exposure of the existing mocks
RAW, CROP = 4880, 4864     # same centre crop as build_xray_dataset.py
GRID = 128
BLOCK = CROP // GRID        # 38


def block_sum(img):
    m = (RAW - CROP) // 2
    c = img[m:m + CROP, m:m + CROP]
    return c.reshape(GRID, BLOCK, GRID, BLOCK).sum(axis=(1, 3))


# ---------------------------------------------------------------- counts
def stage_counts(args):
    with h5py.File(args.xray_h5, "r") as f:
        stored = f["images"][:].astype(np.float64)
        hid = f["meta/halo_id"][:]
        labels = {k: f[f"labels/{k}"][:] for k in f["labels"]} \
            if "labels" in f else {}
    # block *average* of the rate image -> block *sum* of counts
    counts = np.sinh(stored) * BLOCK * BLOCK / QUANTUM
    frac = np.abs(counts - np.round(counts))
    print(f"recovered counts: distance from integer median {np.median(frac):.3f},"
          f" 99th {np.percentile(frac, 99):.3f} (expmap edges are not exact)")
    counts = np.round(counts).astype(np.int64)
    print(f"total counts per cluster at 2 Ms (sum of 3 projections / 3): "
          f"median {np.median(counts.sum(axis=(2, 3)).mean(axis=1)):.3g}")

    # Cross-check the quantum on a raw FITS against the stored dataset.
    raws = sorted(glob.glob(os.path.join(args.xray_root, "snap99_z",
                                         "halo_*_img.fits")))
    if raws:
        from astropy.io import fits
        i = 0
        p = os.path.join(args.xray_root, "snap99_z", f"halo_{hid[i]}_img.fits")
        with fits.open(p) as h:
            d = h[0].data.astype(np.float64)
        direct = block_sum(np.round(d / QUANTUM))
        rel = np.abs(direct - counts[i, 0]).sum() / direct.sum()
        print(f"check vs raw FITS (halo {hid[i]}, snap99_z): "
              f"relative difference {rel:.2e}")
    with h5py.File(args.counts_h5, "w") as g:
        g.create_dataset("counts", data=counts, compression="gzip")
        g.create_dataset("meta/halo_id", data=hid)
        for k, v in labels.items():
            g.create_dataset(f"labels/{k}", data=v)
        g.attrs["exposure_ks"] = T_MOCK_KS
        g.attrs["quantum"] = QUANTUM
    print(f"wrote {args.counts_h5}")


# ---------------------------------------------------------------- geometry
def cosmo():
    from astropy.cosmology import Planck15   # as in the mock pipeline
    return Planck15


def distance_factor(z):
    """Photons from the same cluster at z relative to z=0.05: 1/D_L^2."""
    c = cosmo()
    return float((c.luminosity_distance(Z_MOCK)
                  / c.luminosity_distance(z)).value ** 2)


def sky_area_factor(z):
    """Sky area of one grid block at z relative to z=0.05. The grid is fixed
    in physical units, so a block at higher redshift covers less sky and
    collects proportionally less background: (D_A(0.05) / D_A(z))^2."""
    c = cosmo()
    return float((c.angular_diameter_distance(Z_MOCK)
                  / c.angular_diameter_distance(z)).value ** 2)


def aperture_blocks(z):
    """Radius, in 128-grid blocks, of the physical aperture covered both by
    the mock (ACIS-I at z=0.05) and by ACIS-I at redshift z.

    The grid is fixed in *physical* units (each block is 38 x 0.492" at
    z=0.05), so a cluster placed at another redshift needs no rebinning --
    only the part of it the detector could have seen changes. Above z=0.05
    the real field would reach further out than the mock contains, so the
    mock's own aperture is the limit; below it the detector sees less."""
    c = cosmo()
    block_arcsec = BLOCK * 0.492
    r0 = APERTURE_ARCMIN * 60.0 / block_arcsec
    ratio = float((c.angular_diameter_distance(z)
                   / c.angular_diameter_distance(Z_MOCK)).value)
    return r0 * min(1.0, ratio)


def aperture_mask(z):
    yy, xx = np.mgrid[:GRID, :GRID]
    rr = np.hypot(yy - GRID / 2 + 0.5, xx - GRID / 2 + 0.5)
    return rr <= aperture_blocks(z)


# ---------------------------------------------------------------- bkg
def stage_bkg(args):
    from soxs.utils import soxs_cfg
    soxs_cfg.set("soxs", "bkgnd_nH", "0.018")   # same as the mock pipeline
    import soxs
    from astropy.io import fits

    os.makedirs(args.work_dir, exist_ok=True)
    out = np.zeros((args.n_bkg, GRID, GRID), dtype=np.int64)
    for k in range(args.n_bkg):
        evt = os.path.join(args.work_dir, f"bkg_{k}_evt.fits")
        exp = os.path.join(args.work_dir, f"bkg_{k}_expmap.fits")
        img = os.path.join(args.work_dir, f"bkg_{k}_img.fits")
        soxs.make_background_file(
            evt, (T_MOCK_KS, "ks"), "chandra_acisi_cy22", (0.0, 0.0),
            overwrite=True, foreground=True, instr_bkgnd=False,
            ptsrc_bkgnd=True, prng=np.random.default_rng(1000 + k))
        soxs.make_exposure_map(evt, exp, energy=1.2, overwrite=True)
        soxs.write_image(evt, img, emin=0.1, emax=2.0, overwrite=True,
                         expmap_file=exp)
        with fits.open(img) as h:
            d = h[0].data.astype(np.float64)
        assert d.shape == (RAW, RAW), f"background grid {d.shape} != mock grid"
        q = d[d > 0].min()
        if abs(q - QUANTUM) / QUANTUM > 0.01:
            print(f"  WARNING: background quantum {q:.4g} != mock {QUANTUM}")
        out[k] = np.round(block_sum(d / q)).astype(np.int64)
        print(f"  background {k}: {out[k].sum():.3g} counts at 2 Ms "
              f"({out[k].sum() / T_MOCK_KS:.2f} counts/ks over the field)")
    with h5py.File(args.bkg_h5, "w") as g:
        g.create_dataset("counts", data=out)
        g.attrs["exposure_ks"] = T_MOCK_KS
        g.attrs["components"] = "galactic foreground + unresolved CXB + " \
                                "resolved point sources; no particle bkg"
    print(f"wrote {args.bkg_h5}")


# ---------------------------------------------------------------- build
def stage_pbkg(args):
    """Particle background alone, at 2 Ms. The mocks already contain it,
    mixed into the source counts; when a cluster is placed further away and
    its photons are thinned by the distance factor, the particle background
    must NOT be dimmed with it, so the shortfall is topped up from these."""
    from soxs.utils import soxs_cfg
    soxs_cfg.set("soxs", "bkgnd_nH", "0.018")
    import soxs
    from astropy.io import fits
    os.makedirs(args.work_dir, exist_ok=True)
    out = np.zeros((args.n_bkg, GRID, GRID), dtype=np.int64)
    for k in range(args.n_bkg):
        evt = os.path.join(args.work_dir, f"pbkg_{k}_evt.fits")
        exp = os.path.join(args.work_dir, f"pbkg_{k}_expmap.fits")
        img = os.path.join(args.work_dir, f"pbkg_{k}_img.fits")
        soxs.make_background_file(
            evt, (T_MOCK_KS, "ks"), "chandra_acisi_cy22", (0.0, 0.0),
            overwrite=True, foreground=False, instr_bkgnd=True,
            ptsrc_bkgnd=False, prng=np.random.default_rng(2000 + k))
        soxs.make_exposure_map(evt, exp, energy=1.2, overwrite=True)
        soxs.write_image(evt, img, emin=0.1, emax=2.0, overwrite=True,
                         expmap_file=exp)
        with fits.open(img) as h:
            d = h[0].data.astype(np.float64)
        q = d[d > 0].min()
        out[k] = np.round(block_sum(d / q)).astype(np.int64)
        print(f"  particle background {k}: {out[k].sum():.3g} counts at 2 Ms")
    with h5py.File(args.pbkg_h5, "w") as g:
        g.create_dataset("counts", data=out)
        g.attrs["exposure_ks"] = T_MOCK_KS
    print(f"wrote {args.pbkg_h5}")


def acis_i_targets(obs_csv, targets_csv):
    """{normalised name: (z, total ACIS-I ks)} from the downloaded archive."""
    import pandas as pd
    import re
    key = lambda x: re.sub(r"[\s_]+", "", str(x)).upper()
    o = pd.read_csv(obs_csv)
    o = o[o["detector"].astype(str).str.strip() == "ACIS-I"]
    ks = (o.groupby(o["target"].map(key))["exposure"].sum() / 1e3).to_dict()
    t = pd.read_csv(targets_csv)
    zmap = dict(zip(t["name"].map(key),
                    pd.to_numeric(t["redshift"], errors="coerce")))
    return {k: (float(zmap[k]), float(v)) for k, v in ks.items()
            if np.isfinite(zmap.get(k, np.nan))}


def archive_exposures(path):
    import pandas as pd
    t = pd.read_csv(path)
    ks = t.loc[t["chandra_n"] > 0, "chandra_ks_total"].to_numpy(dtype=float)
    return ks[ks > 0]


def stage_build(args):
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.counts_h5, "r") as f:
        src = f["counts"][:]
        hid = f["meta/halo_id"][:]
        tsc = f["labels/pseudo_tsc"][:]
    with h5py.File(args.bkg_h5, "r") as f:
        bkg = f["counts"][:]
    N, P = src.shape[:2]
    R = args.realizations

    if args.z_from:
        return build_placed(args, rng, src, hid, tsc, bkg)

    if args.depth == "archive":
        pool = archive_exposures(args.archive_csv)
        print(f"drawing exposures from {len(pool)} real Chandra totals: "
              f"median {np.median(pool):.0f} ks")
    else:
        pool = np.array([float(args.depth)])
    t_ks = np.minimum(rng.choice(pool, size=(N, R)), T_MOCK_KS)

    rate = np.zeros((N, R, P, GRID, GRID), dtype=np.float64)
    for i in range(N):
        for r in range(R):
            p = t_ks[i, r] / T_MOCK_KS
            b = bkg[rng.integers(len(bkg))]
            for j in range(P):
                c = rng.binomial(src[i, j], p) + rng.binomial(b, p)
                rate[i, r, j] = c / t_ks[i, r]            # counts per ks
    a = float(np.median(rate[rate > 0]))
    img = np.arcsinh(rate / a).astype(np.float32)
    per_map = (rate * t_ks[:, :, None, None, None]).sum(axis=(3, 4))
    print(f"depth={args.depth}: exposure median {np.median(t_ks):.0f} ks; "
          f"counts per map median {np.median(per_map):.3g} "
          f"(10-90% {np.percentile(per_map, 10):.3g}-"
          f"{np.percentile(per_map, 90):.3g}); stretch scale {a:.4g}")

    with h5py.File(args.output, "w") as g:
        m = g.create_group("mock")
        m.create_dataset("images", data=img, compression="gzip",
                         compression_opts=4)
        m.create_dataset("halo_id", data=hid)
        m.create_dataset("pseudo_tsc", data=tsc)
        m.create_dataset("exposure_ks", data=t_ks.astype(np.float32))
        g.attrs["depth"] = str(args.depth)
        g.attrs["stretch_scale"] = a
        g.attrs["preprocessing"] = "arcsinh(counts per ks per 38x38 block / scale)"
        g.attrs["realism"] = ("Chandra ACIS-I, Poisson-thinned from 2 Ms, "
                              "plus soxs sky background; z fixed at 0.05")
    print(f"wrote {args.output}: mock {img.shape}")


def build_placed(args, rng, src, hid, tsc, bkg):
    """Put each mock at a real target's redshift and exposure.

    The redshift of realization r of cluster i is taken from the radio mock
    set (--z-from), so a joint model sees one consistent cluster. Exposure is
    the real ACIS-I total of the target at that redshift, or a draw from the
    ACIS-I pool if that target has no ACIS-I data.

      source (+ the particle bkg baked into the mocks)  thinned by
          p_src = (t / 2 Ms) * D_L(0.05)^2 / D_L(z)^2
      particle background top-up    by max(p_t * f_sky - p_src, 0), so its
          level is right for the exposure and block sky area, not dimmed
          with the source's distance
      sky background                by p_t * f_sky, f_sky = sky area of a
          block at z relative to z=0.05
      then zeroed outside the common physical aperture.
    For z < 0.05 the baked-in particle background is over-kept by at most a
    factor ~2 on ~1% of the counts; left as is.
    """
    with h5py.File(args.pbkg_h5, "r") as f:
        pbkg = f["counts"][:]
    with h5py.File(args.z_from, "r") as f:
        rz = f["mock/z"][:]
        rh = f["mock/halo_id"][:]
    pos = {h: i for i, h in enumerate(hid)}
    order = np.array([pos[h] for h in rh])       # radio order
    src, hid, tsc = src[order], hid[order], tsc[order]
    N, P = src.shape[:2]
    R = min(args.realizations, rz.shape[1])

    # Optional source-brightness corrections, both applied as extra thinning
    # of the source photons only: a global L_X scale (TNG-Cluster is ~3.6x
    # over-luminous against the LoVoCCS catalogue L_X, so 1/3.6) and each
    # target's Galactic absorption relative to the mock's single column
    # (xray_absorption.py; median 0.977, down to 0.897 for A399/A401).
    nh_by_z = {}
    if args.nh_csv:
        import pandas as pd
        nh = pd.read_csv(args.nh_csv).dropna(subset=["ratio_kT5"])
        for z, grp in nh.groupby(nh["z"].round(4)):
            nh_by_z[z] = float(grp["ratio_kT5"].mean())
        print(f"absorption factors for {len(nh_by_z)} redshifts; "
              f"L_X scale {args.lx_scale}")
    acis = acis_i_targets(args.obs_csv, args.targets_csv)
    pool = np.array([v[1] for v in acis.values()])
    by_z = {}
    for k, (z, ks) in acis.items():
        by_z.setdefault(round(z, 4), []).append(ks)
    print(f"{len(acis)} targets with ACIS-I data; exposure median "
          f"{np.median(pool):.0f} ks (range {pool.min():.0f}-{pool.max():.0f})")

    rate = np.zeros((N, R, P, GRID, GRID), dtype=np.float64)
    t_ks = np.zeros((N, R))
    z_used = np.zeros((N, R))
    capped = matched = 0
    for i in range(N):
        for r in range(R):
            z = float(rz[i, r])
            cand = by_z.get(round(z, 4))
            if cand:
                matched += 1
                t = float(rng.choice(cand))
            else:
                t = float(rng.choice(pool))
            t = min(t, T_MOCK_KS)
            p_t = t / T_MOCK_KS
            p_src = (p_t * distance_factor(z) * args.lx_scale
                     * nh_by_z.get(round(z, 4), 1.0))
            if p_src > 1.0:
                capped += 1
                p_src = 1.0
            b = bkg[rng.integers(len(bkg))]
            pb = pbkg[rng.integers(len(pbkg))]
            m = aperture_mask(z)
            # Backgrounds scale with the sky area a block covers at this
            # redshift, not just with exposure. Before this factor was added
            # every placed mock kept its z=0.05 background per block, i.e.
            # ~3.6x too much at z=0.1 -- one of the two reasons the higher-
            # redshift real clusters looked emptier than their mocks.
            f_sky = sky_area_factor(z)
            p_bkg = p_t * f_sky
            if p_bkg > 1.0:
                raise ValueError(f"background thinning p={p_bkg:.2f} > 1")
            for j in range(P):
                c = (rng.binomial(src[i, j], p_src)
                     + rng.binomial(pb, max(p_bkg - p_src, 0.0))
                     + rng.binomial(b, p_bkg))
                rate[i, r, j] = np.where(m, c, 0) / t
            t_ks[i, r], z_used[i, r] = t, z
    a = float(np.median(rate[rate > 0]))
    img = np.arcsinh(rate / a).astype(np.float32)
    print(f"placed at radio-mock redshifts {z_used.min():.3f}-{z_used.max():.3f};"
          f" exposure from the matching ACIS-I target for "
          f"{matched}/{N * R} realizations; p_src capped at 1 for {capped}")
    print(f"stretch scale {a:.4g} (real data must use this same constant)")

    with h5py.File(args.output, "w") as g:
        mg = g.create_group("mock")
        mg.create_dataset("images", data=img, compression="gzip",
                          compression_opts=4)
        mg.create_dataset("halo_id", data=hid)
        mg.create_dataset("pseudo_tsc", data=tsc)
        mg.create_dataset("exposure_ks", data=t_ks.astype(np.float32))
        mg.create_dataset("z", data=z_used.astype(np.float32))
        g.attrs["depth"] = "real ACIS-I target exposure"
        g.attrs["lx_scale"] = args.lx_scale
        g.attrs["nh_corrected"] = bool(args.nh_csv)
        g.attrs["stretch_scale"] = a
        g.attrs["aperture_arcmin_at_z005"] = APERTURE_ARCMIN
        g.attrs["preprocessing"] = ("arcsinh(counts per ks per block / scale), "
                                    "zero outside the common aperture")
        g.attrs["realism"] = ("ACIS-I, placed at the radio mocks' redshifts: "
                              "1/D_L^2 dimming, real target exposures, sky + "
                              "particle background, common physical aperture")
    print(f"wrote {args.output}: mock {img.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["counts", "bkg", "pbkg", "build"])
    ap.add_argument("--xray-h5", default="dataset_xray_128_orig.h5")
    ap.add_argument("--xray-root", default="TNGCluster_Xray_Snap99")
    ap.add_argument("--counts-h5", default="xray_counts_2Ms_128.h5")
    ap.add_argument("--bkg-h5", default="xray_skybkg_2Ms_128.h5")
    ap.add_argument("--pbkg-h5", default="xray_partbkg_2Ms_128.h5")
    ap.add_argument("--z-from", default=None,
                    help="radio mock h5 whose mock/z sets each realization's "
                         "redshift (redshift placement)")
    ap.add_argument("--obs-csv", default=os.path.expanduser(
        "~/data/cluster-ml/chandra/observations.csv"))
    ap.add_argument("--targets-csv", default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--lx-scale", type=float, default=1.0,
                    help="global source-brightness factor, e.g. 1/3.6 for the "
                         "TNG-Cluster L_X excess over the LoVoCCS catalogue")
    ap.add_argument("--nh-csv", default=None,
                    help="per-target absorption factors (xray_absorption.py)")
    ap.add_argument("--work-dir", default="xray_bkg_work")
    ap.add_argument("--n-bkg", type=int, default=6)
    ap.add_argument("--depth", default="archive",
                    help="exposure in ks, or 'archive' to draw from the real "
                         "Chandra totals in --archive-csv")
    ap.add_argument("--archive-csv", default="lofar_xray_archive.csv")
    ap.add_argument("--realizations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="xray_real_archive.h5")
    args = ap.parse_args()
    {"counts": stage_counts, "bkg": stage_bkg, "pbkg": stage_pbkg,
     "build": stage_build}[args.stage](args)


if __name__ == "__main__":
    main()
