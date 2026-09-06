"""Are the dominant shock cells big enough for volume-spreading to matter?

Spreading only helps if the cells carrying the weight are resolved relative to
the map pixel. If the brightest cell is sub-pixel, its spike survives and the
linear map stays a delta function no matter what kernel is used.
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np

SPHERE_TO_SIGMA = np.sqrt(0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="Radio_Cells/cells_FOF*.npz")
    ap.add_argument("--catalog", default="Radio_Data/TNG-Cluster_Catalog.hdf5")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--extent-r500", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    with h5py.File(args.catalog, "r") as f:
        r500 = {int(h): float(r) * 1000.0
                for h, r in zip(f["haloID"][:], f["r500c"][:])}

    files = sorted(glob.glob(args.cells))
    if args.n:
        files = files[:args.n]
    print(f"{len(files)} clusters, {args.img_size}px over "
          f"+-{args.extent_r500} r500\n")

    rows = []
    for fn in files:
        hid = int(re.search(r"FOF(\d+)", os.path.basename(fn)).group(1))
        d = np.load(fn)
        w, r = d["w"], d["r_kpc"]
        ok = np.isfinite(w) & (w > 0) & np.isfinite(r) & (r > 0)
        w, r = w[ok], r[ok]
        if w.size == 0:
            continue
        px_kpc = 2.0 * args.extent_r500 * r500[hid] / args.img_size
        sig = r * SPHERE_TO_SIGMA / px_kpc
        order = np.argsort(w)[::-1]
        # Weight-weighted sigma: what the map actually feels.
        rows.append((sig[order[0]], np.median(sig[order[:10]]),
                     np.average(sig, weights=w), np.median(sig),
                     (sig > 0.5).mean(), px_kpc,
                     r[order[0]]))
    a = np.array(rows)
    names = ["sigma of top-1 cell [px]", "sigma of top-10 cells [px]",
             "weight-averaged sigma [px]", "median sigma, unweighted [px]",
             "fraction of cells resolved (>0.5px)", "map pixel [kpc]",
             "radius of top-1 cell [kpc]"]
    print(f"{'quantity':<38}{'median':>10}{'10th':>10}{'90th':>10}")
    for j, n in enumerate(names):
        v = a[:, j]
        print(f"{n:<38}{np.median(v):>10.2f}{np.percentile(v, 10):>10.2f}"
              f"{np.percentile(v, 90):>10.2f}")


if __name__ == "__main__":
    main()
