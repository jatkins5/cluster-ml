"""Are the dominant shock cells inside the Psi lookup table's domain?

Radio_generation.py builds its interpolator with fill_value=None, which makes
RegularGridInterpolator *linearly extrapolate* outside the tabulated grid
rather than returning NaN. If the cells carrying the weight sit outside the
table, their Psi -- and so their weight -- is an extrapolation artefact, not
the Hoeft & Bruggen model.

Domain: s in (2, 8], log10(e_min) in [-10, log10 3], with
e_min = 10 * T_keV / 511.
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.constants import k_B
from astropy.cosmology import Planck15

GAMMA = 5.0 / 3.0
XH = 0.76
A, H0 = 1.0, Planck15.h


def s_of_M(M):
    M2 = M * M
    r = (GAMMA + 1.0) * M2 / ((GAMMA - 1.0) * M2 + 2.0)
    return (r + 2.0) / (r - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutout-dir", default="/oscar/data/idellant/Chuiyang/"
                                            "TNGCluster_Cutout/snap99")
    ap.add_argument("--radio-dir", default="Radio_Data")
    ap.add_argument("--psi-table", default="Radio_Data/normalized_psi_table.npz")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    t = np.load(args.psi_table)
    s_lo, s_hi = t["s_grid"].min(), t["s_grid"].max()
    e_lo, e_hi = t["loge_grid"].min(), t["loge_grid"].max()
    print(f"Psi table domain: s in [{s_lo:.3f}, {s_hi:.3f}], "
          f"log10(e_min) in [{e_lo:.3f}, {e_hi:.3f}]")
    print(f"Psi values: min {t['normalized_psi_grid'].min():.3e}, "
          f"max {t['normalized_psi_grid'].max():.3e}\n")

    srcs = sorted(glob.glob(os.path.join(args.radio_dir, "radio_FOF*.npz")))[:args.n]
    out_top, out_all, wfrac_out, negpsi = [], [], [], []
    for src in srcs:
        m = re.search(r"radio_FOF(\d+)_sub(\d+)\.npz", os.path.basename(src))
        fof, sub = m.group(1), m.group(2)
        cut = os.path.join(args.cutout_dir, f"cutout_sub{sub}_FOF{fof}.hdf5")
        if not os.path.exists(cut):
            continue
        with h5py.File(cut, "r") as f:
            p0 = f["PartType0"]
            mach = p0["Machnumber"][:]
            edis = p0["EnergyDissipation"][:]
            sh = (mach > 1.3) & (edis > 0)
            elec = p0["ElectronAbundance"][sh]
            uint = p0["InternalEnergy"][sh]
            mach = mach[sh]
        mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * elec) * c.m_p.cgs.value
        T = (GAMMA - 1.0) * uint / c.k_B.cgs.value * 1e10 * mu
        T_keV = (k_B * (T * u.K)).to(u.keV).value
        loge = np.log10(10.0 * T_keV / 511.0)
        s = s_of_M(mach.astype(np.float64))
        outside = (s < s_lo) | (s > s_hi) | (loge < e_lo) | (loge > e_hi)

        w = np.load(src)["w"]
        ok = np.isfinite(w) & (w > 0)
        if ok.sum() == 0:
            continue
        order = np.argsort(np.where(ok, w, -np.inf))[::-1]
        out_top.append(outside[order[:10]].mean())
        out_all.append(outside.mean())
        wfrac_out.append(w[ok & outside].sum() / w[ok].sum())
        negpsi.append((np.load(src)["w"] < 0).mean())

    for name, v in [("cells outside the table, all", out_all),
                    ("cells outside the table, top-10 by weight", out_top),
                    ("FRACTION OF TOTAL WEIGHT from outside-table cells",
                     wfrac_out),
                    ("negative weights (Psi extrapolated below 0)", negpsi)]:
        v = np.array(v)
        print(f"{name:<52} median {np.median(v):>7.1%}  "
              f"90th {np.percentile(v, 90):>7.1%}")


if __name__ == "__main__":
    main()
