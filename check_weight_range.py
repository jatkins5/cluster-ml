"""How concentrated is the linear radio weight, and is the tail physical?

The forward model has to turn per-cell DSA weights into a surface brightness
map. If a handful of cells carry most of the linear sum, then "normalize the
box flux to the Cuciti anchor" hands nearly all of that flux to those cells and
the mock is a delta function by construction -- no smoothing can fix it.

Reports, per cluster, the fraction of the total linear weight held by the top
1/10/100 cells and the radius of the dominant cells, which distinguishes
"bright shock surface" from "numerical outlier".
"""
import argparse
import glob
import os
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="Radio_Data/radio_FOF*.npz")
    ap.add_argument("--catalog", default="Radio_Data/TNG-Cluster_Catalog.hdf5")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    import h5py
    with h5py.File(args.catalog, "r") as f:
        print("catalog keys:", list(f.keys()))
        # r500c is stored in Mpc; pos is in kpc (see build_dataset.load_catalog)
        r500 = {int(h): float(r) * 1000.0
                for h, r in zip(f["haloID"][:], f["r500c"][:])}

    files = sorted(glob.glob(args.glob))[:args.n]
    print(f"\n{len(files)} clusters")
    print("npz keys:", list(np.load(files[0]).keys()), "\n")
    print(f"{'halo':>8}{'ncell':>9}{'log w min':>11}{'log w max':>11}"
          f"{'top1 %':>9}{'top10 %':>9}{'top100 %':>10}"
          f"{'r_top/r500':>12}{'d_centroid':>13}")
    rows, rads = [], []
    for fn in files:
        hid = int(re.search(r"FOF(\d+)", os.path.basename(fn)).group(1))
        d = np.load(fn)
        w = d["w"].astype(np.float64)
        pos = d["pos"].astype(np.float64)
        good = np.isfinite(w) & (w > 0)
        w, pos = w[good], pos[good]
        if w.size == 0:
            continue
        # Unweighted median, not the weight-weighted mean: the latter is pulled
        # onto the very cell whose radius we are trying to measure.
        pos = pos - np.median(pos, axis=0)
        tot = w.sum()
        order = np.argsort(w)[::-1]
        f1 = w[order[:1]].sum() / tot
        f10 = w[order[:10]].sum() / tot
        f100 = w[order[:100]].sum() / tot
        rr = np.linalg.norm(pos[order[:10]], axis=1) / r500[hid]
        # build_dataset.py centres each image on the weight-averaged position.
        # With one cell holding ~30% of the weight that centroid tracks the
        # brightest shock cell, so measure how far it sits from the halo.
        cen = (pos * w[:, None]).sum(axis=0) / tot
        d_cen = np.linalg.norm(cen) / r500[hid]
        rows.append((f1, f10, f100))
        rads.append((np.median(rr), d_cen))
        print(f"{hid:>8}{w.size:>9}{np.log10(w.min()):>11.1f}"
              f"{np.log10(w.max()):>11.1f}{100 * f1:>9.2f}"
              f"{100 * f10:>9.2f}{100 * f100:>10.2f}"
              f"{rr[0]:>12.2f}{d_cen:>13.2f}")

    a = np.array(rows)
    print(f"\nmedian fraction of total linear weight:")
    for j, n in enumerate([1, 10, 100]):
        print(f"  top {n:>3} cells: {100 * np.median(a[:, j]):.2f}%")
    rd = np.array(rads)
    print(f"median r/r500 of the top-10 cells: {np.median(rd[:, 0]):.2f}")
    print(f"weight-centroid offset from halo [r500]: "
          f"median {np.median(rd[:, 1]):.2f}, "
          f"90th pct {np.percentile(rd[:, 1], 90):.2f}")


if __name__ == "__main__":
    main()
