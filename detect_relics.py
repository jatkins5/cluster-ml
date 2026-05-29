#!/usr/bin/env python3
"""
Detect radio-relic locations in TNG-Cluster halos using simulation
Mach-number maps as ground truth.

For each halo and each of 3 projection axes (xy, yz, xz, matching the radio
dataset's projection order), build a 2D shock surface-brightness map from
gas cells with Mach >= mach_thresh weighted by M^2 * rho, find peaks at
0.3 * R500c < r < FoV, and report distances from a density-weighted center.

Reads cutouts from Chuiyang's snap99 directory; halo IDs and r500c come
from Radio_Data/TNG-Cluster_Catalog.hdf5; pseudo_tsc and mass_ratio are
joined from dataset.h5 for downstream use.

Output relic_catalog.h5:
  halo_id (N,)              int64
  r500c_kpc (N,)            float32
  pseudo_tsc (N,)           float32   (joined from dataset.h5)
  mass_ratio (N,)           float32   (joined from dataset.h5)
  center_xyz (N, 3)         float32   kpc, density-weighted core
  n_relics (N, 3)           int32     per-projection peak counts post-filter
  d_primary (N, 3)          float32   kpc, brightest peak per proj (NaN if none)
  d_secondary (N, 3)        float32   kpc, 2nd brightest
  d_max (N, 3)              float32   kpc, max-r peak (outer-relic proxy)
  flux_primary (N, 3)       float32   summed shock weight at primary peak

Projection axis convention (matches radio dataset):
  proj 0 = xy (view along z)
  proj 1 = yz (view along x)
  proj 2 = xz (view along y)

Usage:
  python detect_relics.py --halo-id 0           # single halo, smoke test
  python detect_relics.py                       # all halos in dataset.h5
"""

import argparse
import glob
import os

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

CUTOUT_DIR = "/oscar/data/idellant/Chuiyang/TNGCluster_Cutout/snap99"
PROJECTIONS = [("xy", 0, 1), ("yz", 1, 2), ("xz", 0, 2)]


def find_cutout(halo_id):
    matches = glob.glob(os.path.join(CUTOUT_DIR, f"cutout_sub*_FOF{halo_id}.hdf5"))
    return matches[0] if matches else None


def robust_center(coord, mass, density, k_frac=0.001, k_min=200):
    """Mass-weighted center of the top-k_frac densest gas cells (X-ray peak proxy)."""
    n_top = max(int(k_frac * len(density)), k_min)
    top = np.argpartition(density, -n_top)[-n_top:]
    c = coord[top]
    m = mass[top]
    return (c * m[:, None]).sum(0) / m.sum()


def find_peaks(img, threshold, min_sep_pix):
    mx = maximum_filter(img, size=2 * min_sep_pix + 1)
    return np.argwhere((img == mx) & (img > threshold))


def detect_relics(hdf5_path, r500c_kpc, mach_thresh=2.0, grid_size=128,
                  fov_r500c=2.5, r_min_frac=0.3, r_max_frac=2.0,
                  peak_thresh_frac=0.05, min_sep_kpc=300, smooth_kpc=80):
    """Returns dict with center, n_relics[3], d_primary[3], d_secondary[3],
    d_max[3], flux_primary[3]. Returns None if no shock cells exist."""
    with h5py.File(hdf5_path, "r") as f:
        M_all = f["PartType0/Machnumber"][:]
        sk = M_all >= mach_thresh
        if not sk.any():
            return None
        coord_all = f["PartType0/Coordinates"][:]
        mass_all = f["PartType0/Masses"][:]
        density_all = f["PartType0/Density"][:]

    center = robust_center(coord_all, mass_all, density_all)
    coord = coord_all[sk]
    rho = density_all[sk]
    M = M_all[sk]

    rel = coord - center
    w_shock = (M ** 2) * rho                      # shock surface-brightness proxy

    half_width = fov_r500c * r500c_kpc
    pixel_kpc = (2 * half_width) / grid_size
    min_sep_pix = max(1, int(np.ceil(min_sep_kpc / pixel_kpc)))
    smooth_pix = max(1.0, smooth_kpc / pixel_kpc)
    edges = np.linspace(-half_width, half_width, grid_size + 1)
    r_min_kpc = r_min_frac * r500c_kpc
    r_max_kpc = r_max_frac * r500c_kpc

    out = {
        "center": center.astype(np.float32),
        "n_relics": np.zeros(3, dtype=np.int32),
        "d_primary": np.full(3, np.nan, dtype=np.float32),
        "d_secondary": np.full(3, np.nan, dtype=np.float32),
        "d_max": np.full(3, np.nan, dtype=np.float32),
        "flux_primary": np.zeros(3, dtype=np.float32),
    }

    for k, (_name, ax0, ax1) in enumerate(PROJECTIONS):
        H, _, _ = np.histogram2d(rel[:, ax0], rel[:, ax1],
                                 bins=edges, weights=w_shock)
        H = gaussian_filter(H, sigma=smooth_pix)
        if H.max() <= 0:
            continue
        thr = peak_thresh_frac * H.max()
        peaks = find_peaks(H, thr, min_sep_pix)
        recs = []
        for (i, j) in peaks:
            x = (i + 0.5) * pixel_kpc - half_width
            y = (j + 0.5) * pixel_kpc - half_width
            r = float(np.hypot(x, y))
            if r < r_min_kpc or r > r_max_kpc:      # exclude core & far-field
                continue
            recs.append((r, float(H[i, j])))
        if not recs:
            continue
        recs.sort(key=lambda t: -t[1])              # brightest first
        out["n_relics"][k] = len(recs)
        out["d_primary"][k] = recs[0][0]
        out["flux_primary"][k] = recs[0][1]
        if len(recs) > 1:
            out["d_secondary"][k] = recs[1][0]
        out["d_max"][k] = max(r for r, _ in recs)
    return out


def main(args):
    with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5", "r") as f:
        r500c_map = {int(h): float(r) * 1000.0
                     for h, r in zip(f["haloID"][:], f["r500c"][:])}
    with h5py.File("dataset.h5", "r") as f:
        ds_halo = f["meta/halo_id"][:]
        ds_pseudo = f["labels/pseudo_tsc"][:]
        ds_mr = f["meta/mass_ratio"][:]
    pseudo_map = {int(h): float(p) for h, p in zip(ds_halo, ds_pseudo)}
    mr_map = {int(h): float(m) for h, m in zip(ds_halo, ds_mr)}
    # merger-TSC from the pre-joined Lee-style catalog (0–7.7 Gyr, ~6/352 NaN)
    with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5", "r") as f:
        mtsc_map = {int(h): float(t) for h, t in zip(f["halo_id"][:], f["tsc_gyr"][:])}

    if args.halo_id is not None:
        halo_ids = [args.halo_id]
    else:
        halo_ids = ds_halo.tolist()

    N = len(halo_ids)
    print(f"Processing {N} halos  mach_thresh={args.mach_thresh}  "
          f"grid={args.grid_size}  fov={args.fov_r500c}xR500c")

    halo_arr = np.array(halo_ids, dtype=np.int64)
    r500_arr = np.zeros(N, dtype=np.float32)
    pseudo_arr = np.full(N, np.nan, dtype=np.float32)
    merger_tsc_arr = np.full(N, np.nan, dtype=np.float32)
    mr_arr = np.full(N, np.nan, dtype=np.float32)
    center_arr = np.zeros((N, 3), dtype=np.float32)
    n_relics = np.zeros((N, 3), dtype=np.int32)
    d_primary = np.full((N, 3), np.nan, dtype=np.float32)
    d_secondary = np.full((N, 3), np.nan, dtype=np.float32)
    d_max = np.full((N, 3), np.nan, dtype=np.float32)
    flux_primary = np.zeros((N, 3), dtype=np.float32)

    for i, hid in enumerate(halo_ids):
        if hid not in r500c_map:
            print(f"[{i}] halo {hid}: not in catalog, skip"); continue
        path = find_cutout(hid)
        if not path:
            print(f"[{i}] halo {hid}: cutout missing, skip"); continue
        r500 = r500c_map[hid]
        r500_arr[i] = r500
        pseudo_arr[i] = pseudo_map.get(hid, np.nan)
        merger_tsc_arr[i] = mtsc_map.get(hid, np.nan)
        mr_arr[i] = mr_map.get(hid, np.nan)
        try:
            res = detect_relics(path, r500,
                                mach_thresh=args.mach_thresh,
                                grid_size=args.grid_size,
                                fov_r500c=args.fov_r500c,
                                r_min_frac=args.r_min_frac,
                                r_max_frac=args.r_max_frac,
                                peak_thresh_frac=args.peak_thresh_frac,
                                min_sep_kpc=args.min_sep_kpc)
        except Exception as e:
            print(f"[{i}] halo {hid}: ERROR {e}"); continue
        if res is None:
            print(f"[{i}] halo {hid}: no Mach>={args.mach_thresh} cells"); continue
        center_arr[i] = res["center"]
        n_relics[i] = res["n_relics"]
        d_primary[i] = res["d_primary"]
        d_secondary[i] = res["d_secondary"]
        d_max[i] = res["d_max"]
        flux_primary[i] = res["flux_primary"]
        if N == 1 or i % 20 == 0:
            print(f"[{i:3d}/{N}] halo {hid:>10d} r500={r500:5.0f}kpc  "
                  f"n_relics={n_relics[i].tolist()}  "
                  f"d_prim={np.round(d_primary[i], 1).tolist()}  "
                  f"d_max={np.round(d_max[i], 1).tolist()}  "
                  f"pseudo_tsc={pseudo_arr[i]:.2f}")

    with h5py.File(args.output, "w") as f:
        f.attrs["mach_thresh"] = args.mach_thresh
        f.attrs["grid_size"] = args.grid_size
        f.attrs["fov_r500c"] = args.fov_r500c
        f.create_dataset("halo_id", data=halo_arr)
        f.create_dataset("r500c_kpc", data=r500_arr)
        f.create_dataset("pseudo_tsc", data=pseudo_arr)
        f.create_dataset("merger_tsc", data=merger_tsc_arr)
        f.create_dataset("mass_ratio", data=mr_arr)
        f.create_dataset("center_xyz", data=center_arr)
        f.create_dataset("n_relics", data=n_relics)
        f.create_dataset("d_primary", data=d_primary)
        f.create_dataset("d_secondary", data=d_secondary)
        f.create_dataset("d_max", data=d_max)
        f.create_dataset("flux_primary", data=flux_primary)
    print(f"saved {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--halo-id", type=int, default=None,
                   help="single halo for smoke test; default = all in dataset.h5")
    p.add_argument("--mach-thresh", type=float, default=2.0)
    p.add_argument("--grid-size", type=int, default=128)
    p.add_argument("--fov-r500c", type=float, default=2.5)
    p.add_argument("--r-min-frac", type=float, default=0.3,
                   help="exclude peaks within this fraction of R500c (core)")
    p.add_argument("--r-max-frac", type=float, default=2.0,
                   help="exclude peaks beyond this fraction of R500c (far-field)")
    p.add_argument("--peak-thresh-frac", type=float, default=0.05,
                   help="keep peaks above this fraction of map max")
    p.add_argument("--min-sep-kpc", type=float, default=300.0,
                   help="minimum separation between peaks")
    p.add_argument("--output", type=str, default="relic_catalog.h5")
    main(p.parse_args())
