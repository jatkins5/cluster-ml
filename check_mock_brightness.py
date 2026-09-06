"""Is the mock dynamic-range deficit a normalization or a resolution problem?

Mocks peak at a median 3 sigma while AGN-masked LoTSS cutouts still reach 8+.
The two candidate causes need opposite fixes, so measure before changing
anything:

  too faint      -> total box flux is low, peak/total ratio is normal
  too smooth     -> total box flux matches, peak/total ratio is low

The sim maps are 512px over +-4 r500, so their native pixel is ~16 kpc for a
typical r500, while the 9" LoTSS beam is ~9-14 kpc over the LoVoCCS redshift
range. If the native pixel exceeds the beam the mock is pre-smoothed relative
to the observation and cannot reproduce its peaks at any normalization.
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward_model_lotss import (BEAM_FWHM_ARCSEC, GRID, kpc_per_arcsec,
                                 load_obs, make_mock)


def stats(img, rms, fwhm_px):
    beam_area_px = 1.1331 * fwhm_px ** 2
    flux_jy = img.sum() / beam_area_px
    peak = img.max()
    # Concentration: peak surface brightness per unit total flux. Independent
    # of the overall normalization, so it isolates how smooth the map is.
    conc = peak / flux_jy if flux_jy > 0 else np.nan
    return flux_jy, peak / rms, conc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_512.h5")
    ap.add_argument("--lotss-glob", default=os.path.expanduser(
        "~/data/cluster-ml/lotss_images/*.fits"))
    ap.add_argument("--targets-csv",
                    default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--catalog", default="Radio_Data/TNG-Cluster_Catalog.hdf5")
    ap.add_argument("--box-kpc", type=float, default=1000.0)
    ap.add_argument("--max-beams", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth-px", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
                    help="sim-frame smoothing sigmas (native pixels) to sweep")
    ap.add_argument("--invert-arcsinh", action="store_true",
                    help="undo build_dataset.py's arcsinh stretch (off by "
                         "default, matching forward_model_lotss.py)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mask_rng = np.random.default_rng(args.seed + 1000)

    obs = load_obs(args.lotss_glob, args.targets_csv, args.box_kpc,
                   mask_rng=mask_rng, max_beams=args.max_beams)
    px_kpc = args.box_kpc / GRID
    o_rows = []
    for o in obs:
        fwhm_px = BEAM_FWHM_ARCSEC * kpc_per_arcsec(o["z"]) / px_kpc
        o_rows.append(stats(o["map"], o["rms"], fwhm_px))
    o_rows = np.array(o_rows)

    with h5py.File(args.dataset, "r") as f:
        images = f["images"][:]
        halo_ids = f["meta/halo_id"][:]
        r500_kpc = f["meta/r500c_kpc"][:]
    with h5py.File(args.catalog, "r") as f:
        m500_map = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
    N, P, H, _ = images.shape

    pair = rng.integers(0, len(obs), size=N)
    native_px = np.array([8.0 * r500_kpc[i] / H for i in range(N)])
    beam_kpc = np.array([BEAM_FWHM_ARCSEC * kpc_per_arcsec(obs[pair[i]]["z"])
                         for i in range(N)])

    def col(a, j):
        v = a[:, j]
        return v[np.isfinite(v)]

    names = ["total box flux [Jy]", "peak / rms", "peak per unit flux [1/Jy]"]
    o_med = [np.median(col(o_rows, j)) for j in range(3)]
    print(f"\nobs (masked): " + "  ".join(
        f"{n}={v:.4g}" for n, v in zip(names, o_med)))
    print(f"\n{'smooth_px':>10}{'flux':>12}{'peak/rms':>12}{'conc':>12}"
          f"{'flux ratio':>12}{'peak ratio':>12}{'conc ratio':>12}")

    for sp in args.smooth_px:
        # Same stream per setting, so only the smoothing differs between rows.
        srng = np.random.default_rng(args.seed + 77)
        m_rows = []
        for i in range(N):
            z = obs[pair[i]]["z"]
            rms = obs[pair[i]]["rms"]
            fwhm_px = BEAM_FWHM_ARCSEC * kpc_per_arcsec(z) / px_kpc
            for p in range(P):
                mock = make_mock(images[i, p], native_px[i],
                                 m500_map[halo_ids[i]], z, rms, args.box_kpc,
                                 srng, invert_arcsinh=args.invert_arcsinh,
                                 sim_smooth_px=sp)
                if mock is not None:
                    m_rows.append(stats(mock, rms, fwhm_px))
        m_rows = np.array(m_rows)
        med = [np.median(col(m_rows, j)) for j in range(3)]
        print(f"{sp:>10.2f}" + "".join(f"{v:>12.4g}" for v in med)
              + "".join(f"{m / o:>12.2f}" for m, o in zip(med, o_med)))

    npx, bk = native_px, beam_kpc
    print(f"\nsim native pixel  : median {np.median(npx):.1f} kpc "
          f"[{npx.min():.1f}, {npx.max():.1f}]")
    print(f"LoTSS beam FWHM   : median {np.median(bk):.1f} kpc "
          f"[{bk.min():.1f}, {bk.max():.1f}]")
    print(f"mock effective res: {np.sqrt(np.median(npx)**2 + np.median(bk)**2):.1f}"
          f" kpc vs obs {np.median(bk):.1f} kpc")
    print(f"clusters where native pixel exceeds the beam: "
          f"{(npx > np.median(bk)).mean():.0%}")


if __name__ == "__main__":
    main()
