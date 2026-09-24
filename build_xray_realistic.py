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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["counts", "bkg", "build"])
    ap.add_argument("--xray-h5", default="dataset_xray_128_orig.h5")
    ap.add_argument("--xray-root", default="TNGCluster_Xray_Snap99")
    ap.add_argument("--counts-h5", default="xray_counts_2Ms_128.h5")
    ap.add_argument("--bkg-h5", default="xray_skybkg_2Ms_128.h5")
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
    {"counts": stage_counts, "bkg": stage_bkg, "build": stage_build}[args.stage](args)


if __name__ == "__main__":
    main()
