#!/usr/bin/env python3
"""2 Ms X-ray counts for an earlier snapshot, on the snapshot-99 physical grid.

The snapshot-99 mocks were observed at z = 0.05; their training counts are
recovered from Chuiyang's exposure-corrected images as round(pixel/QUANTUM)
per native 0.492" pixel and block-summed 38 x 38 (build_xray_realistic.py
counts stage). The real Chandra inputs are exposure-corrected as well.

Chuiyang's snapshot-91 mocks were observed at the snapshot's own redshift
(0.0994), where one physical block spans 38 * D_A(0.05)/D_A(z) = 20.25
native pixels. Same recipe here, from the snapshot-91 images: per-pixel
counts round(pixel/QUANTUM), then each pixel's counts are split between the
blocks it overlaps, binomially in proportion to the overlap, along x and then
y. That keeps whole counts (the binomial depth-thinning needs them) and
avoids the 20-or-21-pixel aliasing a nearest-block assignment would print on
every map. (Binning the raw event lists instead gives ~11% fewer counts than
the image recipe, because the images are corrected for partial exposure at
chip gaps and dither edges -- so it would not match snapshot 99 or the real
data.)

Same band (0.1-2 keV; the images were made with it), same centre as the
snapshot-99 crop, same projection order as build_xray_dataset.py.
Run as a SLURM array (each task every n-th halo -> part file), then --merge.
"""
import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from astropy.io import fits

from build_xray_realistic import BLOCK, CROP, GRID, QUANTUM, RAW, Z_MOCK, block_sum

C = "/oscar/data/idellant/Chuiyang"
AXES = ["z", "x", "y"]            # radio projections xy, yz, xz
CENTRE = RAW / 2.0                # crop centre, in 0-based pixel-edge units


def block_of_pixels(b):
    """For native pixels 0..RAW-1 (pixel k spans [k, k+1] in edge units):
    the block holding its left part, and the fraction of the pixel in it."""
    e0 = CENTRE - GRID / 2 * b
    left = np.arange(RAW, dtype=np.float64)
    blk = np.floor((left - e0) / b).astype(int)
    frac = np.clip(e0 + (blk + 1) * b - left, 0.0, 1.0)
    return blk, frac


def split_axis(a, blk, frac, rng, axis):
    """Move counts along `axis` from native pixels onto GRID blocks."""
    a = np.moveaxis(a, axis, -1)
    first = rng.binomial(a, frac)                     # part in block blk
    rest = a - first                                  # part in block blk + 1
    out = np.zeros((int(np.prod(a.shape[:-1])), GRID), dtype=np.int64)
    for part, k in ((first, blk), (rest, blk + 1)):
        ok = (k >= 0) & (k < GRID)          # parts off the grid are dropped
        np.add.at(out, (slice(None), k[ok]),
                  part.reshape(len(out), -1)[:, ok])
    return np.moveaxis(out.reshape(a.shape[:-1] + (GRID,)), -1, axis)


def regrid(counts, b, rng):
    """(RAW, RAW) per-pixel counts -> (GRID, GRID) physical-block counts."""
    blk, frac = block_of_pixels(b)
    used = (blk >= -1) & (blk < GRID)                 # can touch the grid
    lo, hi = np.where(used)[0][[0, -1]]
    c = counts[lo:hi + 1, lo:hi + 1]
    bb, ff = blk[lo:hi + 1], frac[lo:hi + 1]
    c = split_axis(c, bb, ff, rng, axis=1)            # x (columns)
    return split_axis(c, bb, ff, rng, axis=0)         # y (rows)


def pixel_counts(path):
    with fits.open(path, memmap=False) as h:
        d = h[0].data.astype(np.float64)
    assert d.shape == (RAW, RAW), d.shape
    return np.round(d / QUANTUM).astype(np.int64)


def scale(z):
    return float((Planck15.angular_diameter_distance(Z_MOCK)
                  / Planck15.angular_diameter_distance(z)).value)


def task(args):
    rng = np.random.default_rng(1000 + args.task)
    hm = pd.read_csv(args.halo_map).sort_values("FOF_Halo_ID_At99")
    b = BLOCK * scale(args.z)
    print(f"z = {args.z}: block {b:.3f} px ({b * 0.492:.2f}\")")
    if args.check:
        # On the native grid (b = 38, pixel-aligned) the split is exact and
        # must reproduce the snapshot-99 recipe bit for bit.
        f91 = int(hm.iloc[0][f"FOF_Halo_ID_At{args.snap}"])
        pc = pixel_counts(f"{args.xray_root}/snap{args.snap}_z/halo_{f91}_img_z.fits")
        a, ref = regrid(pc, float(BLOCK), rng), block_sum(pc)
        print(f"  check on native grid: identical = {np.array_equal(a, ref)}")
        g = regrid(pc, b, rng)
        m = (RAW - CROP) // 2
        e0 = int(np.ceil(CENTRE - GRID / 2 * b)); e1 = int(np.floor(CENTRE + GRID / 2 * b))
        print(f"  conservation: grid {g.sum()} vs pixels fully inside "
              f"{pc[e0:e1, e0:e1].sum()} (edge pixels split)")
    rows = hm.iloc[args.task::args.ntasks]
    out = np.zeros((len(rows), 3, GRID, GRID), dtype=np.int64)
    ok = np.zeros(len(rows), dtype=bool)
    for n, r in enumerate(rows.itertuples()):
        f91 = int(getattr(r, f"FOF_Halo_ID_At{args.snap}"))
        paths = [f"{args.xray_root}/snap{args.snap}_{a}/halo_{f91}_img_{a}.fits"
                 for a in AXES]
        if not all(os.path.exists(p) for p in paths):
            print(f"  FOF{f91}: missing image(s); skipped")
            continue
        for k, p in enumerate(paths):
            out[n, k] = regrid(pixel_counts(p), b, rng)
        ok[n] = True
        print(f"  FOF{f91} -> z=0 {int(r.FOF_Halo_ID_At99)}: "
              f"{out[n].sum(axis=(1, 2))} counts", flush=True)
    os.makedirs(args.work_dir, exist_ok=True)
    with h5py.File(f"{args.work_dir}/part_{args.task:03d}.h5", "w") as g:
        g.create_dataset("counts", data=out[ok], compression="gzip")
        g.create_dataset("halo_id", data=rows["FOF_Halo_ID_At99"].to_numpy()[ok])


def merge(args):
    parts = sorted(glob.glob(f"{args.work_dir}/part_*.h5"))
    cnt, hid = [], []
    for p in parts:
        with h5py.File(p) as f:
            cnt.append(f["counts"][:]); hid.append(f["halo_id"][:])
    cnt, hid = np.concatenate(cnt), np.concatenate(hid)
    with h5py.File(args.labels_from) as f:        # radio dataset, same snapshot
        lh = list(f["meta/halo_id"][:])
        tsc = f["labels/pseudo_tsc"][:]
    pos = {h: i for i, h in enumerate(hid)}
    keep = [h for h in lh if h in pos]
    print(f"{len(parts)} parts, {len(hid)} halos; {len(keep)} in {args.labels_from}")
    with h5py.File(args.output, "w") as g:
        g.create_dataset("counts", data=cnt[[pos[h] for h in keep]], compression="gzip")
        g.create_dataset("meta/halo_id", data=np.array(keep))
        g.create_dataset("labels/pseudo_tsc", data=tsc[[lh.index(h) for h in keep]])
        g.attrs["exposure_ks"] = 2000.0
        g.attrs["z_mock"] = args.z
        g.attrs["snapshot"] = args.snap
        g.attrs["grid"] = "snapshot-99 physical grid (38 x 0.492\" blocks at z=0.05)"
    print(f"wrote {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=int, default=91)
    ap.add_argument("--z", type=float, default=0.0994018026302219)
    ap.add_argument("--xray-root", default=f"{C}/TNGCluster_Xray_Snap91")
    ap.add_argument("--halo-map", default=f"{C}/TNGCluster_Cutout/fof_halo_to_sub91.csv")
    ap.add_argument("--work-dir", default="xray_s91_parts")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--labels-from", default="dataset_nh4_144_s91_128.h5")
    ap.add_argument("--output", default="xray_counts_2Ms_128_s91.h5")
    a = ap.parse_args()
    merge(a) if a.merge else task(a)
