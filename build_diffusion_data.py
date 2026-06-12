#!/usr/bin/env python3
"""
Build an unconditional diffusion training tensor from RAW radio NPZ data.

Unlike dataset.h5 (parameter-free np.arcsinh tuned for the CNN), this uses a
tunable, invertible arcsinh(x / a) stretch and stores the stretch + scaling
parameters so generated samples can be mapped back to physical units for
physical evaluation (flux distribution, radial profiles, power spectra).

Pipeline per cluster (mirrors build_dataset.py centering/FoV for consistency):
  1. weight-averaged center, FoV = extent_r500 * r500c
  2. project particles onto xy / yz / xz  -> raw weighted pixel maps
  3. y = arcsinh(x / a)        a = robust scale (median positive pixel)
  4. map [0, y_hi] -> [-1, 1]  y_hi = high percentile (clip brights)

Output (diffusion_radio_<size>.h5):
  images       : (N, 3, S, S) float32 in [-1, 1]
  meta/halo_id : (N,) int64        (for cluster-level train/val split)
  attrs        : arcsinh_a, y_hi, img_size, extent_r500   (for inversion)

Inverse (for evaluation):  y = (img + 1) / 2 * y_hi ;  x = a * sinh(y)

Usage:
  python build_diffusion_data.py --img-size 64 --extent-r500 2.0
"""

import argparse
import os

import h5py
import numpy as np


def project_image(
    pos: np.ndarray,
    weights: np.ndarray,
    center: np.ndarray,
    half_width: float,
    img_size: int,
) -> np.ndarray:
    """Raw (un-stretched) weighted projections onto the xy, yz, xz planes.

    Returns a (3, img_size, img_size) float64 stack — one weighted 2D
    histogram per projection axis pair.
    """
    rel_pos = pos - center
    axis_pairs = [(0, 1), (1, 2), (0, 2)]
    images = np.zeros((3, img_size, img_size), dtype=np.float64)
    edges = np.linspace(-half_width, half_width, img_size + 1)
    for proj, (ax0, ax1) in enumerate(axis_pairs):
        hist, _, _ = np.histogram2d(
            rel_pos[:, ax0], rel_pos[:, ax1], bins=edges, weights=weights
        )
        images[proj] = hist
    return images


def load_r500c_map(catalog_path: str) -> dict[int, float]:
    """Map halo_id -> R500c in kpc (catalog stores Mpc)."""
    with h5py.File(catalog_path, "r") as f:
        halo_ids = f["haloID"][:]
        r500c_mpc = f["r500c"][:]
    return {int(h): float(r) * 1000.0 for h, r in zip(halo_ids, r500c_mpc)}


def main(img_size: int, extent_r500: float, hi_pct: float, output_path: str) -> None:
    radio_dir = "Radio_Data"
    r500c_map = load_r500c_map(os.path.join(radio_dir, "TNG-Cluster_Catalog.hdf5"))

    npz_files = sorted(
        f for f in os.listdir(radio_dir)
        if f.startswith("radio_FOF") and f.endswith(".npz")
    )
    halo_ids = [int(f.replace("radio_FOF", "").split("_sub")[0]) for f in npz_files]
    N = len(npz_files)
    print(f"Found {N} clusters. Projecting at {img_size}px, FoV {extent_r500}xR500c...")

    raw_maps = np.zeros((N, 3, img_size, img_size), dtype=np.float64)
    for i, (fname, hid) in enumerate(zip(npz_files, halo_ids)):
        if i % 50 == 0:
            print(f"  {i}/{N}")
        npz = np.load(os.path.join(radio_dir, fname))
        pos, weights = npz["pos"], npz["w"]
        weight_sum = weights.sum()
        center = ((pos * weights[:, None]).sum(0) / weight_sum
                  if weight_sum > 0 else pos.mean(0))
        half_width = extent_r500 * r500c_map[hid]
        raw_maps[i] = project_image(pos, weights, center, half_width, img_size)

    # robust arcsinh scale: median of strictly-positive pixel values.
    # Puts the linear->log knee at a typical signal pixel (not an arbitrary 1).
    positive_pixels = raw_maps[raw_maps > 0]
    arcsinh_a = float(np.median(positive_pixels))
    stretched = np.arcsinh(raw_maps / arcsinh_a)

    # map [0, y_hi] -> [-1, 1]; y_hi = high percentile so one saturated pixel
    # per cluster doesn't compress all real structure into a thin band.
    y_hi = float(np.percentile(stretched, hi_pct))
    images = np.clip(2.0 * stretched / y_hi - 1.0, -1.0, 1.0).astype(np.float32)

    print(f"arcsinh_a={arcsinh_a:.4e}  y_hi(p{hi_pct})={y_hi:.4f}  "
          f"clipped={(stretched > y_hi).mean()*100:.3f}% of pixels")

    with h5py.File(output_path, "w") as f:
        f.attrs["arcsinh_a"] = arcsinh_a
        f.attrs["y_hi"] = y_hi
        f.attrs["hi_pct"] = hi_pct
        f.attrs["img_size"] = img_size
        f.attrs["extent_r500"] = extent_r500
        f.create_dataset("images", data=images, compression="gzip")
        f.create_dataset("meta/halo_id", data=np.array(halo_ids, dtype=np.int64))
    print(f"Saved {output_path}  images {images.shape}  range "
          f"[{images.min():.3f}, {images.max():.3f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--extent-r500", type=float, default=2.0)
    p.add_argument("--hi-pct", type=float, default=99.9,
                   help="percentile of arcsinh values mapped to +1 (clip above)")
    p.add_argument("--output", type=str, default=None)
    a = p.parse_args()
    out = a.output or f"diffusion_radio_{a.img_size}.h5"
    main(a.img_size, a.extent_r500, a.hi_pct, out)
