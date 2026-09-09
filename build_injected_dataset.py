#!/usr/bin/env python3
"""Inject mock diffuse emission into real LoTSS fields ("FullReal" training).

Bottrell et al. (2019, MNRAS 490, 5390) tested three tiers of realism for
merger CNNs: idealized; analytic noise+PSF ("SemiReal"); and simulated sources
inserted into real survey fields ("FullReal"). SemiReal reached ~65% on
survey-realistic data, FullReal 87.1%, and they conclude field insertion was
crucial. Our forward model is exactly SemiReal. Botteon et al. (2023,
A&A 672, A41) do the radio version, injecting mock haloes into real LoTSS data.

Here the sim map is beam-convolved and flux-anchored as before, but instead of
adding synthetic noise it is added to a real DR3 field offset 1-3 degrees from
a LoVoCCS target. The background supplies real correlated noise, real radio
galaxies with the right counts and clustering, real calibration artefacts, and
real crowding -- none of which we have to model or tune.

Both training and inference images then get the identical compact-source mask,
which also retires the old "never mask the mocks" asymmetry: that rule held
only while the mocks contained no sources to mask.
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward_model_lotss import (BEAM_FWHM_ARCSEC, GRID, OBS_PIX_ARCSEC,
                                 crop_resample, kpc_per_arcsec, load_obs,
                                 mask_compact, sigma_clipped_rms,
                                 total_flux_jy)
from scipy import ndimage


def load_backgrounds(field_dir):
    """Real DR3 cutouts, kept at native 1.5 arcsec/px so they can be cropped
    to whatever angular size each injection redshift implies."""
    out = []
    for p in sorted(glob.glob(os.path.join(field_dir, "bg_*.fits"))):
        try:
            with fits.open(p) as h:
                d = np.squeeze(h[0].data).astype(np.float64)
        except Exception:
            continue
        if d.ndim != 2 or min(d.shape) < 200:
            continue
        if np.isfinite(d).mean() < 0.95:
            continue
        out.append((os.path.basename(p)[3:-5], np.nan_to_num(d, nan=0.0)))
    return out


def inject(sim_map, sim_px_kpc, log_m500, z, bg, box_kpc, rng,
           scatter=True, mask=True, max_beams=10.0, mask_rng=None):
    """Beam-convolve and flux-anchor the sim map, then add it to real sky."""
    half_px = box_kpc / 2.0 / sim_px_kpc
    cut = crop_resample(sim_map.astype(np.float64), half_px)
    if cut is None or cut.sum() <= 0:
        return None
    px_kpc = box_kpc / GRID
    s_tot = total_flux_jy(log_m500, z, rng if scatter else None)
    cut *= s_tot / cut.sum()                       # Jy per grid pixel

    kpas = kpc_per_arcsec(z)
    fwhm_px = BEAM_FWHM_ARCSEC * kpas / px_kpc
    cut = ndimage.gaussian_filter(cut, fwhm_px / 2.355, mode="constant")
    cut *= 1.1331 * fwhm_px ** 2                   # Jy/px -> Jy/beam

    # Crop the real field to the same angular size this redshift implies, then
    # resample to the common grid -- the same path load_obs uses, so the
    # background keeps its true noise correlation and source scale.
    half_bg_px = box_kpc / 2.0 / (OBS_PIX_ARCSEC * kpas)
    bgc = crop_resample(bg, half_bg_px)
    if bgc is None:
        return None

    img = cut + bgc
    rms = sigma_clipped_rms(img)
    if not np.isfinite(rms) or rms <= 0:
        return None
    if mask:
        img, _ = mask_compact(img, rms, fwhm_px,
                              mask_rng if mask_rng is not None else rng,
                              max_beams=max_beams)
    return img, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_nh4_512.h5")
    ap.add_argument("--catalog", default="Radio_Data/TNG-Cluster_Catalog.hdf5")
    ap.add_argument("--field-dir", default="lotss_fields")
    ap.add_argument("--lotss-glob", default=os.path.expanduser(
        "~/data/cluster-ml/lotss_images/*.fits"))
    ap.add_argument("--targets-csv", default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--box-kpc", type=float, default=1000.0)
    ap.add_argument("--realizations", type=int, default=3)
    ap.add_argument("--mask-max-beams", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="injected_nh4.h5")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mask_rng = np.random.default_rng(args.seed + 1000)

    bgs = load_backgrounds(args.field_dir)
    if not bgs:
        raise SystemExit(f"no usable backgrounds in {args.field_dir}")
    print(f"{len(bgs)} real background fields")

    obs = load_obs(args.lotss_glob, args.targets_csv, args.box_kpc,
                   mask_rng=mask_rng, max_beams=args.mask_max_beams)
    print(f"{len(obs)} real target cutouts (inference set)")
    obs_z = np.array([o["z"] for o in obs])

    with h5py.File(args.dataset, "r") as f:
        images = f["images"][:]
        halo_ids = f["meta/halo_id"][:]
        r500_kpc = f["meta/r500c_kpc"][:]
        tsc = f["labels/pseudo_tsc"][:]
    with h5py.File(args.catalog, "r") as f:
        m500 = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
    N, P, H, _ = images.shape
    R = args.realizations

    out = np.zeros((N, R, P, GRID, GRID), dtype=np.float32)
    keep = np.ones(N, dtype=bool)
    used_z = np.zeros((N, R), dtype=np.float32)
    used_rms = np.zeros((N, R), dtype=np.float32)
    for i in range(N):
        if i % 50 == 0:
            print(f"  {i}/{N}")
        px_kpc = 8.0 * r500_kpc[i] / H
        for r in range(R):
            # Redshift from the real target sample, background drawn
            # independently so the model cannot key on a fixed pairing.
            z = float(rng.choice(obs_z))
            bg = bgs[rng.integers(len(bgs))][1]
            for p in range(P):
                got = inject(images[i, p], px_kpc, m500[halo_ids[i]], z, bg,
                             args.box_kpc, rng, mask=True,
                             max_beams=args.mask_max_beams, mask_rng=mask_rng)
                if got is None:
                    keep[i] = False
                    continue
                img, rms = got
                out[i, r, p] = np.arcsinh(img / rms)
                used_z[i, r], used_rms[i, r] = z, rms

    print(f"injected for {keep.sum()}/{N} clusters")
    obs_imgs = np.stack([np.arcsinh(o["map"] / o["rms"]) for o in obs]
                        ).astype(np.float32)

    with h5py.File(args.output, "w") as f:
        f.attrs["source_dataset"] = args.dataset
        f.attrs["preprocessing"] = "arcsinh(map / rms)"
        f.attrs["realism"] = "injected into real LoTSS DR3 fields"
        f.attrs["n_backgrounds"] = len(bgs)
        g = f.create_group("mock")
        g.create_dataset("images", data=out[keep], compression="gzip",
                         compression_opts=4)
        g.create_dataset("halo_id", data=halo_ids[keep])
        g.create_dataset("pseudo_tsc", data=tsc[keep])
        g.create_dataset("z", data=used_z[keep])
        g.create_dataset("rms", data=used_rms[keep])
        o = f.create_group("obs")
        o.create_dataset("images", data=obs_imgs, compression="gzip",
                         compression_opts=4)
        o.create_dataset("name", data=np.array([x["name"] for x in obs],
                                               dtype=h5py.string_dtype()))
        o.create_dataset("z", data=obs_z)
        o.create_dataset("rms", data=np.array([x["rms"] for x in obs]))
    print(f"wrote {args.output}: mock {out[keep].shape}, obs {obs_imgs.shape}")


if __name__ == "__main__":
    main()
