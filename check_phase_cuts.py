"""Which gas-phase cut breaks the weight concentration?

The weight is carried by cold, ~380x overdense, Mach-26 cells (stripped ISM
of infalling galaxies) rather than the diffuse ICM. A cut is only worth a
full regeneration if it (a) removes those cells, (b) leaves the emission
spread over many cells rather than one, and (c) does not throw away the
genuine ICM shock emission with them.

Reports, per candidate cut: the fraction of the original weight retained, the
share still held by the single brightest cell, and the phase of what remains.
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_radio_cells import generate, make_psi

CUTS = [
    ("no cut (upstream model)",      dict()),
    ("T > 0.1 keV",                  dict(min_T_keV=0.1)),
    ("T > 0.5 keV",                  dict(min_T_keV=0.5)),
    ("T > 1.0 keV",                  dict(min_T_keV=1.0)),
    ("n_H < 1e-3",                   dict(max_nH=1e-3)),
    ("n_H < 1e-4",                   dict(max_nH=1e-4)),
    ("no star-forming",              dict(exclude_sf=True)),
    ("T > 0.5 keV & n_H < 1e-3",     dict(min_T_keV=0.5, max_nH=1e-3)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutout-dir", default="/oscar/data/idellant/Chuiyang/"
                                            "TNGCluster_Cutout/snap99")
    ap.add_argument("--radio-dir", default="Radio_Data")
    ap.add_argument("--psi-table", default="Radio_Data/normalized_psi_table.npz")
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    interp_psi = make_psi(args.psi_table)
    srcs = sorted(glob.glob(os.path.join(args.radio_dir, "radio_FOF*.npz")))[:args.n]
    cutouts = []
    for src in srcs:
        m = re.search(r"radio_FOF(\d+)_sub(\d+)\.npz", os.path.basename(src))
        p = os.path.join(args.cutout_dir,
                         f"cutout_sub{m.group(2)}_FOF{m.group(1)}.hdf5")
        if os.path.exists(p):
            cutouts.append(p)
    print(f"{len(cutouts)} clusters\n")

    base = {}
    print(f"{'cut':<28}{'weight kept':>13}{'top-1 cell':>12}"
          f"{'top-10':>10}{'cells kept':>12}")
    for name, kw in CUTS:
        kept_w, top1, top10, kept_n = [], [], [], []
        for cut in cutouts:
            out = generate(cut, interp_psi, **kw)
            if out is None:
                continue
            _, w, _ = out
            ok = np.isfinite(w) & (w > 0)
            w = w[ok]
            if w.size == 0:
                continue
            tot = w.sum()
            if not kw:
                base[cut] = (tot, ok.sum())
            b_tot, b_n = base.get(cut, (tot, ok.sum()))
            s = np.sort(w)[::-1]
            kept_w.append(tot / b_tot)
            top1.append(s[0] / tot)
            top10.append(s[:10].sum() / tot)
            kept_n.append(ok.sum() / b_n)
        print(f"{name:<28}{np.median(kept_w):>12.1%}{np.median(top1):>12.1%}"
              f"{np.median(top10):>10.1%}{np.median(kept_n):>12.1%}")


if __name__ == "__main__":
    main()
