#!/usr/bin/env python3
"""
Build X-ray dataset matching the radio dataset.h5 layout.

Loads simulated Chandra ACIS-I images (4880x4880, ~0.492 arcsec/pixel,
2xR500c FoV) from TNGCluster_Xray_Snap99/{snap99_x,snap99_y,snap99_z}/
halo_<id>_img.fits, downsamples via center-crop + block-averaging, and
saves arcsinh-normalised images keyed by halo_id in the same order as
dataset.h5.

Projection-axis alignment with radio dataset.h5 (xy, yz, xz):
    radio proj 0 (xy)  view along z  ->  X-ray snap99_z
    radio proj 1 (yz)  view along x  ->  X-ray snap99_x
    radio proj 2 (xz)  view along y  ->  X-ray snap99_y

Output: dataset_xray.h5
    images        : (352, 3, IMG_SIZE, IMG_SIZE) float32, arcsinh-normalised
    meta/halo_id  : (352,) int64, matches dataset.h5 ordering

Usage:
    python build_xray_dataset.py [--img-size 128] [--output dataset_xray.h5]
"""

import argparse
import os

import h5py
import numpy as np
from astropy.io import fits


# Each X-ray FITS image is 4880x4880. Choosing a center crop that is a
# multiple of the target size keeps block-averaging exact (no fractional
# pixels). For 128: 4864 = 128 * 38. For 256: 4864 = 256 * 19.
RAW_SIZE = 4880

# Radio (xy, yz, xz) -> X-ray dir (view along z, x, y)
PROJECTION_DIRS = ["snap99_z", "snap99_x", "snap99_y"]


def block_average(img, target_size):
    """
    Downsample a 2-D image to (target_size, target_size) by area-averaging.

    Steps:
      1. Center-crop the image to the largest multiple of target_size
         that fits within RAW_SIZE. For RAW_SIZE=4880:
             target=128  -> crop=4864  (factor 38, drops 8 px per side)
             target=256  -> crop=4864  (factor 19, drops 8 px per side)
      2. Reshape to (target, factor, target, factor) and mean over the
         two factor axes. This is exact area-averaging on intensive
         (per-pixel) surface-brightness values, which is what `_img.fits`
         contains (counts/sec/pixel after exposure-map division).

    No interpolation, no filtering — just block-mean. This conserves
    surface brightness and is the standard rebinning operation in X-ray
    astronomy.
    """
    h, w = img.shape
    assert h == w == RAW_SIZE, f"unexpected image shape {img.shape}"

    factor = h // target_size                     # integer divisor
    crop_size = factor * target_size              # largest clean crop
    margin = (h - crop_size) // 2                 # symmetric margin

    cropped = img[margin:margin + crop_size,
                  margin:margin + crop_size]      # (crop_size, crop_size)

    # block-average: reshape to (target, factor, target, factor), mean over
    # the two `factor` axes -> (target, target)
    reduced = cropped.reshape(
        target_size, factor,
        target_size, factor,
    ).mean(axis=(1, 3))

    return reduced.astype(np.float32)


def load_xray_image(xray_root, projection_dir, halo_id):
    """Load one halo_<id>_img.fits as a (4880, 4880) float32 array."""
    path = os.path.join(xray_root, projection_dir, f"halo_{halo_id}_img.fits")
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    return data.astype(np.float32)


def main(img_size, output_path, xray_root, radio_dataset):
    # use the same halo ordering as the radio dataset so indices align
    with h5py.File(radio_dataset, "r") as f:
        halo_ids = f["meta/halo_id"][:]            # (352,) int64

    n_halos = len(halo_ids)
    n_proj = len(PROJECTION_DIRS)
    images = np.zeros((n_halos, n_proj, img_size, img_size), dtype=np.float32)

    print(f"Building X-ray dataset: {n_halos} halos x {n_proj} projections")
    print(f"  raw size:    {RAW_SIZE}x{RAW_SIZE}")
    print(f"  target size: {img_size}x{img_size}")
    print(f"  factor:      {RAW_SIZE // img_size}x")
    print(f"  crop margin: {(RAW_SIZE - (RAW_SIZE // img_size) * img_size) // 2} px per side")
    print()

    for i, hid in enumerate(halo_ids):
        for k, proj_dir in enumerate(PROJECTION_DIRS):
            raw = load_xray_image(xray_root, proj_dir, int(hid))
            reduced = block_average(raw, img_size)
            images[i, k] = np.arcsinh(reduced)
        if (i + 1) % 25 == 0 or (i + 1) == n_halos:
            print(f"  processed {i + 1}/{n_halos} halos")

    print(f"\nWriting {output_path}")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("images", data=images, compression="gzip")
        f.create_dataset("meta/halo_id", data=halo_ids)
        f["images"].attrs["projection_order"] = "xy, yz, xz (matches dataset.h5)"
        f["images"].attrs["xray_dirs"] = ",".join(PROJECTION_DIRS)
        f["images"].attrs["normalization"] = "arcsinh of block-averaged counts/sec/pixel"
        f["images"].attrs["raw_pixel_arcsec"] = 0.492
        f["images"].attrs["fov_r500c"] = 2.0
    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img-size", type=int, default=128,
                        help="Target image size (pixels per side, default 128)")
    parser.add_argument("--output", type=str, default="dataset_xray.h5",
                        help="Output HDF5 path (default dataset_xray.h5)")
    parser.add_argument("--xray-root", type=str, default="TNGCluster_Xray_Snap99",
                        help="Root directory containing snap99_{x,y,z}/")
    parser.add_argument("--radio-dataset", type=str, default="dataset.h5",
                        help="Radio dataset.h5 (used for halo_id ordering)")
    args = parser.parse_args()
    main(args.img_size, args.output, args.xray_root, args.radio_dataset)
