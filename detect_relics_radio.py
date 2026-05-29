#!/usr/bin/env python3
"""
Independent relic-peak detection directly in the radio surface-brightness
maps. The radio dataset's pixel values are arcsinh-stretched projections of
the Hoeft & Brueggen synchrotron weights, so radio relics ARE drawn on these
maps by construction. If Lee's relation isn't here either, it isn't in our
TNG-Cluster sample at our resolution and stretch.

Same output schema and projection convention as detect_relics.py so we can
reuse plot_relic_validation_v3.py for the comparison.

Two implementation differences from the Mach detector:
  - Threshold is set from the max within the radial annulus
    (0.3 R500c < r < 2 R500c), not the global map max — the central halo
    is much brighter than any peripheral relic, so a global-max threshold
    would silently kill every outer peak.
  - Reads dataset_512.h5 (or override) which already has r500c, halo_id,
    labels — no cutout I/O needed.
"""

import argparse

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def detect_in_image(img, r500c_kpc, half_width_kpc,
                    r_min_frac=0.3, r_max_frac=2.0,
                    peak_thresh_frac=0.25,
                    min_sep_kpc=600.0, smooth_kpc=50.0):
    """Returns list of (r_kpc, flux) for accepted peaks in the annulus."""
    H, W = img.shape
    pixel_kpc = 2 * half_width_kpc / H
    min_sep_pix = max(1, int(np.ceil(min_sep_kpc / pixel_kpc)))
    smooth_pix = max(1.0, smooth_kpc / pixel_kpc)

    smoothed = gaussian_filter(img, sigma=smooth_pix)

    # pixel-center kpc grid
    yi, xi = np.indices((H, W))
    x_kpc = (xi + 0.5) * pixel_kpc - half_width_kpc
    y_kpc = (yi + 0.5) * pixel_kpc - half_width_kpc
    r_kpc = np.hypot(x_kpc, y_kpc)

    r_min = r_min_frac * r500c_kpc
    r_max = r_max_frac * r500c_kpc
    annulus = (r_kpc >= r_min) & (r_kpc <= r_max)
    if not annulus.any() or smoothed[annulus].max() <= 0:
        return []

    thr = peak_thresh_frac * smoothed[annulus].max()
    mx = maximum_filter(smoothed, size=2 * min_sep_pix + 1)
    is_peak = (smoothed == mx) & (smoothed > thr) & annulus
    coords = np.argwhere(is_peak)

    return [(float(r_kpc[i, j]), float(smoothed[i, j])) for i, j in coords]


def main(args):
    with h5py.File(args.dataset, "r") as f:
        imgs = f["images"][:]                              # (N, 3, H, W)
        halo_ids = f["meta/halo_id"][:]
        r500c = f["meta/r500c_kpc"][:]
        attrs = dict(f.attrs)
    extent_r500 = float(attrs.get("extent_r500", 2.0))

    with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5", "r") as f:
        mtsc_map = {int(h): float(t)
                    for h, t in zip(f["halo_id"][:], f["tsc_gyr"][:])}
    with h5py.File("dataset.h5", "r") as f:
        ds_halo = f["meta/halo_id"][:]
        ds_pseudo = f["labels/pseudo_tsc"][:]
        ds_mr = f["meta/mass_ratio"][:]
    pseudo_map = {int(h): float(p) for h, p in zip(ds_halo, ds_pseudo)}
    mr_map = {int(h): float(m) for h, m in zip(ds_halo, ds_mr)}

    N, _, H, W = imgs.shape
    print(f"Processing {N} halos at {H}x{W}  FoV={extent_r500}xR500c  "
          f"peak>={args.peak_thresh_frac} of annulus-max")

    pseudo_arr = np.array([pseudo_map.get(int(h), np.nan)
                           for h in halo_ids], dtype=np.float32)
    mtsc_arr = np.array([mtsc_map.get(int(h), np.nan)
                         for h in halo_ids], dtype=np.float32)
    mr_arr = np.array([mr_map.get(int(h), np.nan)
                       for h in halo_ids], dtype=np.float32)

    n_relics = np.zeros((N, 3), dtype=np.int32)
    d_primary = np.full((N, 3), np.nan, dtype=np.float32)
    d_secondary = np.full((N, 3), np.nan, dtype=np.float32)
    d_max = np.full((N, 3), np.nan, dtype=np.float32)
    flux_primary = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        half_w = extent_r500 * float(r500c[i])
        for k in range(3):
            recs = detect_in_image(
                imgs[i, k], float(r500c[i]), half_w,
                r_min_frac=args.r_min_frac,
                r_max_frac=args.r_max_frac,
                peak_thresh_frac=args.peak_thresh_frac,
                min_sep_kpc=args.min_sep_kpc,
            )
            if not recs:
                continue
            recs.sort(key=lambda t: -t[1])
            n_relics[i, k] = len(recs)
            d_primary[i, k] = recs[0][0]
            flux_primary[i, k] = recs[0][1]
            if len(recs) > 1:
                d_secondary[i, k] = recs[1][0]
            d_max[i, k] = max(r for r, _ in recs)
        if i % 50 == 0:
            print(f"  [{i:3d}/{N}] halo {int(halo_ids[i]):>10d}  "
                  f"n={n_relics[i].tolist()}  "
                  f"d_prim={np.round(d_primary[i], 0).tolist()}  "
                  f"merger_tsc={mtsc_arr[i]:.2f}")

    with h5py.File(args.output, "w") as f:
        f.attrs["source"] = args.dataset
        f.attrs["peak_thresh_frac"] = args.peak_thresh_frac
        f.attrs["min_sep_kpc"] = args.min_sep_kpc
        f.attrs["r_min_frac"] = args.r_min_frac
        f.attrs["r_max_frac"] = args.r_max_frac
        # the validation script reads 'mach_thresh'; use 0.0 as marker for radio
        f.attrs["mach_thresh"] = 0.0
        f.create_dataset("halo_id", data=halo_ids)
        f.create_dataset("r500c_kpc", data=r500c)
        f.create_dataset("pseudo_tsc", data=pseudo_arr)
        f.create_dataset("merger_tsc", data=mtsc_arr)
        f.create_dataset("mass_ratio", data=mr_arr)
        f.create_dataset("n_relics", data=n_relics)
        f.create_dataset("d_primary", data=d_primary)
        f.create_dataset("d_secondary", data=d_secondary)
        f.create_dataset("d_max", data=d_max)
        f.create_dataset("flux_primary", data=flux_primary)
    print(f"saved {args.output}")
    print(f"  median n_relics per projection: {int(np.median(n_relics))}")
    print(f"  projections with at least one relic: "
          f"{int((n_relics > 0).sum())}/{n_relics.size}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="dataset_512.h5")
    p.add_argument("--r-min-frac", type=float, default=0.3)
    p.add_argument("--r-max-frac", type=float, default=2.0)
    p.add_argument("--peak-thresh-frac", type=float, default=0.25)
    p.add_argument("--min-sep-kpc", type=float, default=600.0)
    p.add_argument("--output", type=str, default="relic_catalog_radio.h5")
    main(p.parse_args())
