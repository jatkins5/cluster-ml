"""Where does the DSA radio weight actually come from, physically?

The weight is concentrated in the smallest cells, which suggested the
emission sits in dense knots rather than in the low-density outskirts where
real radio relics live. That was an inference from w ~ B**(1+0.5s); this
measures it directly, comparing the properties of the cells that carry the
weight against the shock-cell population as a whole.

Real relics, for reference: r ~ 1-2 r500, Mach 2-4, n_e ~ 1e-4 cm^-3,
B ~ 0.5-2 uG. If the dominant cells instead sit at high density, low Mach
and high B, the model is putting the emission in the wrong gas.
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

GAMMA, XH = 5.0 / 3.0, 0.76
A, H0 = 1.0, Planck15.h
UNIT_B_G = (np.sqrt(1e10 * u.Msun / u.kpc) * (u.km / u.s) / u.kpc).to(
    u.g ** 0.5 / (u.cm ** 0.5 * u.s)).value
# Density code unit (1e10 Msun/h)/(ckpc/h)^3 -> g/cm^3, then to n_H in cm^-3.
RHO_CGS = ((1e10 * u.Msun / u.kpc ** 3).to(u.g / u.cm ** 3).value
           * H0 ** 2 / A ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutout-dir", default="/oscar/data/idellant/Chuiyang/"
                                            "TNGCluster_Cutout/snap99")
    ap.add_argument("--cells", default="Radio_Cells/cells_FOF*.npz")
    ap.add_argument("--groupcat", default="/oscar/data/idellant/Chuiyang/"
                                          "groupcat_classification/groupcat_099")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    # Halo centres + r500, same source as check_centering.py.
    files = sorted(glob.glob(os.path.join(args.groupcat, "fof_subhalo_tab_*.hdf5")),
                   key=lambda p: int(re.search(r"\.(\d+)\.hdf5$", p).group(1)))
    gp, gr, box = [], [], None
    for fn in files:
        with h5py.File(fn, "r") as f:
            if box is None:
                box = float(f["Header"].attrs["BoxSize"]) * A / H0
            if f["Header"].attrs["Ngroups_ThisFile"] == 0:
                continue
            gp.append(f["Group/GroupPos"][:])
            gr.append(f["Group/Group_R_Crit500"][:])
    gp = np.concatenate(gp) * A / H0
    gr = np.concatenate(gr) * A / H0

    srcs = sorted(glob.glob(args.cells))[:args.n]
    top, allc = [], []
    wfrac_in, wfrac_out = [], []
    for src in srcs:
        m = re.search(r"cells_FOF(\d+)_sub(\d+)\.npz", os.path.basename(src))
        fof, sub = int(m.group(1)), m.group(2)
        cut = os.path.join(args.cutout_dir, f"cutout_sub{sub}_FOF{fof}.hdf5")
        if not os.path.exists(cut):
            continue
        with h5py.File(cut, "r") as f:
            p0 = f["PartType0"]
            mach = p0["Machnumber"][:]
            edis = p0["EnergyDissipation"][:]
            sh = (mach > 1.3) & (edis > 0)
            dens = p0["Density"][sh].astype(np.float64)
            elec = p0["ElectronAbundance"][sh].astype(np.float64)
            uint = p0["InternalEnergy"][sh].astype(np.float64)
            bfld = p0["MagneticField"][sh].astype(np.float64)
            mach = mach[sh].astype(np.float64)
        w = np.load(src)["w"]
        pos = np.load(src)["pos"]
        rel = pos - gp[fof]
        rel -= box * np.round(rel / box)
        rr = np.linalg.norm(rel, axis=1) / gr[fof]

        mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * elec) * c.m_p.cgs.value
        T_keV = (k_B * (((GAMMA - 1.0) * uint / c.k_B.cgs.value * 1e10 * mu)
                        * u.K)).to(u.keV).value
        nH = dens * RHO_CGS * XH / c.m_p.cgs.value
        B = np.sqrt((bfld ** 2).sum(axis=1)) * H0 / A ** 2 * UNIT_B_G * 1e6

        ok = np.isfinite(w) & (w > 0)
        if ok.sum() < 20:
            continue
        order = np.argsort(np.where(ok, w, -np.inf))[::-1][:10]
        cols = lambda i: (nH[i], mach[i], B[i], T_keV[i], rr[i])
        top.append([np.median(v) for v in cols(order)])
        allc.append([np.median(v) for v in cols(np.where(ok)[0])])
        tot = w[ok].sum()
        wfrac_in.append(w[ok & (rr < 0.5)].sum() / tot)
        wfrac_out.append(w[ok & (rr > 1.0)].sum() / tot)

    t, a = np.array(top), np.array(allc)
    names = ["n_H [cm^-3]", "Mach number", "B [uG]", "T [keV]", "r / r500"]
    print(f"{len(t)} clusters\n")
    print(f"{'quantity':<16}{'top-10 by weight':>20}{'all shock cells':>20}"
          f"{'ratio':>10}")
    for j, n in enumerate(names):
        mt, ma = np.median(t[:, j]), np.median(a[:, j])
        print(f"{n:<16}{mt:>20.4g}{ma:>20.4g}{mt / ma:>10.2f}")

    print(f"\nfraction of total weight from r < 0.5 r500 (core): "
          f"median {np.median(wfrac_in):.1%}")
    print(f"fraction of total weight from r > 1.0 r500 (relic zone): "
          f"median {np.median(wfrac_out):.1%}")


if __name__ == "__main__":
    main()
