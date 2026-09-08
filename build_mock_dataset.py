#!/usr/bin/env python3
"""Build a CNN training set of forward-modelled mock LoTSS observations.

Every CNN number in this project so far was measured on *clean* sim images --
no beam, no noise, no AGN, full dynamic range. Real LoTSS has all of those, so
those numbers are a ceiling, not an estimate of what transfers. This builds the
training set for the experiment that actually tests transfer: train on mocks
that have been through the forward model, then apply to real cutouts.

Outputs one HDF5 with
  mock/images   (N, R, 3, 128, 128)  R noise+pairing realizations per cluster
  mock/...      halo_id, pseudo_tsc, z, rms
  obs/images    (n_obs, 128, 128)    the real LoTSS cutouts, same grid
  obs/...       name, z, rms

Both sides are stored as arcsinh(map / rms), i.e. compressed SNR units. That
is the domain-matching step and it deliberately discards absolute flux
calibration: the sim/obs L_X tilt and the P150-M500 anchor make absolute
brightness the least trustworthy axis, so the CNN should not be asked to
learn it. Structure is what transfers.
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward_model_lotss import GRID, load_obs, make_mock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_nh4_512.h5")
    ap.add_argument("--catalog", default="Radio_Data/TNG-Cluster_Catalog.hdf5")
    ap.add_argument("--lotss-glob", default=os.path.expanduser(
        "~/data/cluster-ml/lotss_images/*.fits"))
    ap.add_argument("--targets-csv",
                    default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--box-kpc", type=float, default=1000.0)
    ap.add_argument("--realizations", type=int, default=3,
                    help="noise/pairing realizations per cluster")
    ap.add_argument("--mask-max-beams", type=float, default=10.0)
    ap.add_argument("--invert-arcsinh", action="store_true")
    ap.add_argument("--correlated-noise", action="store_true",
                    help="beam-correlated mock noise instead of white")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="mock_dataset_nh4.h5")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mask_rng = np.random.default_rng(args.seed + 1000)

    # Obs are AGN-masked; mocks are not, mirroring what a real diffuse-emission
    # analysis does and what the MF work established (never mask the mocks).
    print("processing LoTSS cutouts...")
    obs = load_obs(args.lotss_glob, args.targets_csv, args.box_kpc,
                   mask_rng=mask_rng, max_beams=args.mask_max_beams)
    print(f"usable obs maps: {len(obs)}")

    with h5py.File(args.dataset, "r") as f:
        images = f["images"][:]
        halo_ids = f["meta/halo_id"][:]
        r500_kpc = f["meta/r500c_kpc"][:]
        tsc = f["labels/pseudo_tsc"][:]
    with h5py.File(args.catalog, "r") as f:
        m500_map = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
    N, P, H, _ = images.shape
    R = args.realizations
    print(f"sim maps: {N} x {P} proj, {H}px -> {R} realizations each")

    out = np.zeros((N, R, P, GRID, GRID), dtype=np.float32)
    z_used = np.zeros((N, R), dtype=np.float32)
    rms_used = np.zeros((N, R), dtype=np.float32)
    keep = np.ones(N, dtype=bool)

    for i in range(N):
        if i % 50 == 0:
            print(f"  {i}/{N}")
        px_kpc = 8.0 * r500_kpc[i] / H
        for r in range(R):
            j = rng.integers(0, len(obs))
            z, rms = obs[j]["z"], obs[j]["rms"]
            z_used[i, r], rms_used[i, r] = z, rms
            for p in range(P):
                mock = make_mock(images[i, p], px_kpc, m500_map[halo_ids[i]],
                                 z, rms, args.box_kpc, rng,
                                 invert_arcsinh=args.invert_arcsinh,
                                 correlated_noise=args.correlated_noise)
                if mock is None:
                    keep[i] = False
                    continue
                out[i, r, p] = np.arcsinh(mock / rms)

    print(f"mocks built for {keep.sum()}/{N} clusters")
    obs_imgs = np.stack([np.arcsinh(o["map"] / o["rms"]) for o in obs]
                        ).astype(np.float32)

    with h5py.File(args.output, "w") as f:
        f.attrs["source_dataset"] = args.dataset
        f.attrs["box_kpc"] = args.box_kpc
        f.attrs["preprocessing"] = "arcsinh(map / rms)"
        f.attrs["invert_arcsinh"] = bool(args.invert_arcsinh)
        f.attrs["mask_max_beams"] = args.mask_max_beams
        f.attrs["correlated_noise"] = bool(args.correlated_noise)
        g = f.create_group("mock")
        g.create_dataset("images", data=out[keep], compression="gzip",
                         compression_opts=4)
        g.create_dataset("halo_id", data=halo_ids[keep])
        g.create_dataset("pseudo_tsc", data=tsc[keep])
        g.create_dataset("z", data=z_used[keep])
        g.create_dataset("rms", data=rms_used[keep])
        o = f.create_group("obs")
        o.create_dataset("images", data=obs_imgs, compression="gzip",
                         compression_opts=4)
        o.create_dataset("name", data=np.array([x["name"] for x in obs],
                                               dtype=h5py.string_dtype()))
        o.create_dataset("z", data=np.array([x["z"] for x in obs]))
        o.create_dataset("rms", data=np.array([x["rms"] for x in obs]))
    print(f"wrote {args.output}: mock {out[keep].shape}, obs {obs_imgs.shape}")


if __name__ == "__main__":
    main()
