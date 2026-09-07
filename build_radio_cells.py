#!/usr/bin/env python3
"""Regenerate the DSA radio weights, keeping each shock cell's physical size.

Radio_Data/radio_FOF*.npz stores only (pos, w), so the map builder has no
choice but to deposit each cell as a point. That is what makes the linear map
a delta function: one cell carries a median 30% of the total weight (see the
weight-concentration finding), and a point deposit gives it a single pixel.

TNG-Cluster gas cells in the ICM are 15 kpc in radius against an 18 kpc map
pixel, so the missing ingredient is the cell's own volume. Masses and Density
are already present in the cutouts (Chuiyang's Radio_generation.py reads both
and uses neither), so V = M / rho gives an equivalent-sphere radius per cell.

Reproduces the weight formula from
/oscar/data/idellant/Chuiyang/Radio_Data/Radio_generation.py exactly -- the
--validate flag checks the recomputed w against the stored npz -- and adds
r_kpc. Chuiyang's files are read-only; output goes to Radio_Cells/.
"""
import argparse
import glob
import os
import re
import time

import h5py
import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.constants import k_B
from astropy.cosmology import Planck15
from scipy.interpolate import RegularGridInterpolator

GAMMA = 5.0 / 3.0
XH = 0.76
Z = 0.0
A = 1.0 / (1.0 + Z)
H0 = Planck15.h

# Code-unit conversions, transcribed from Radio_generation.py.
UNIT_B_G = (np.sqrt(1e10 * u.Msun / u.kpc) * (u.km / u.s) / u.kpc).to(
    u.g ** 0.5 / (u.cm ** 0.5 * u.s)).value
ECONV = ((1e10 * u.Msun / u.kpc) * (u.km / u.s) ** 3).to(u.erg / u.s).value
# Density code unit (1e10 Msun/h)/(ckpc/h)^3 -> g/cm^3, then n_H = rho*XH/m_p.
RHO_CGS = ((1e10 * u.Msun / u.kpc ** 3).to(u.g / u.cm ** 3).value
           * H0 ** 2 / A ** 3)


def r_of_M(M):
    M2 = M * M
    return (GAMMA + 1.0) * M2 / ((GAMMA - 1.0) * M2 + 2.0)


def s_of_M(M):
    r = r_of_M(M)
    return (r + 2.0) / (r - 1.0)


def bcmb_uG(z):
    return 3.24 * (1.0 + z) ** 2


def make_psi(table):
    d = np.load(table)
    return RegularGridInterpolator(
        (d["s_grid"], d["loge_grid"]), d["normalized_psi_grid"],
        bounds_error=False, fill_value=None)


def generate(cutout, interp_psi, min_T_keV=0.0, max_nH=np.inf,
             exclude_sf=False):
    with h5py.File(cutout, "r") as f:
        p0 = f["PartType0"]
        mach = p0["Machnumber"][:]
        edis = p0["EnergyDissipation"][:]
        shock = (mach > 1.3) & (edis > 0)
        if not shock.any():
            return None
        coords = p0["Coordinates"][shock]
        dens = p0["Density"][shock]
        mass = p0["Masses"][shock]
        elec = p0["ElectronAbundance"][shock]
        uint = p0["InternalEnergy"][shock]
        bfld = p0["MagneticField"][shock]
        sfr = p0["StarFormationRate"][shock] if exclude_sf else None
        mach = mach[shock]
        edis = edis[shock]

    mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * elec) * c.m_p.cgs.value
    temperature = (GAMMA - 1.0) * uint / c.k_B.cgs.value * 1e10 * mu
    T_keV = (k_B * (temperature * u.K)).to(u.keV).value

    # Gas-phase cuts. The upstream model selects on Mach number alone, which
    # lets the weight be carried by cold, ~380x overdense, Mach-26 cells --
    # ram-pressure stripped ISM of infalling galaxies, not the diffuse ICM in
    # which radio relics form. Excluding cold and star-forming gas is standard
    # in DSA relic modelling; these cuts are off by default so the unmodified
    # model is reproduced exactly.
    nH = dens.astype(np.float64) * RHO_CGS * XH / c.m_p.cgs.value
    keep = (T_keV >= min_T_keV) & (nH <= max_nH)
    if exclude_sf:
        keep &= (sfr <= 0)
    if not keep.all():
        coords, dens, mass = coords[keep], dens[keep], mass[keep]
        elec, uint, bfld = elec[keep], uint[keep], bfld[keep]
        mach, edis, T_keV = mach[keep], edis[keep], T_keV[keep]
        if mach.size == 0:
            return None

    pos = coords * A / H0                                   # physical kpc
    M = mach.astype(np.float64)
    E = edis.astype(np.float64) / A * ECONV * 1e-44
    B_uG = np.sqrt((bfld.astype(np.float64) ** 2).sum(axis=1)) * H0 / A ** 2 \
        * UNIT_B_G * 1e6
    s = s_of_M(M)
    phi = interp_psi(np.column_stack([s, np.log10(T_keV * 10.0 / 511.0)]))
    w = 5.2e23 * E * B_uG ** (1.0 + 0.5 * s) / (B_uG ** 2 + bcmb_uG(Z) ** 2) * phi

    # Equivalent-sphere radius of each Voronoi cell. Mass units cancel, so
    # only the length unit needs converting to physical kpc.
    vol = mass.astype(np.float64) / dens.astype(np.float64)
    r_kpc = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0) * A / H0
    return pos, w, r_kpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutout-dir",
                    default="/oscar/data/idellant/Chuiyang/TNGCluster_Cutout/snap99")
    ap.add_argument("--radio-dir", default="Radio_Data")
    ap.add_argument("--out-dir", default="Radio_Cells")
    ap.add_argument("--psi-table", default="Radio_Data/normalized_psi_table.npz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--validate", action="store_true",
                    help="compare recomputed w against the stored npz")
    ap.add_argument("--min-temp-kev", type=float, default=0.0,
                    help="drop shock cells colder than this (ICM is ~2 keV; "
                         "the dominant cells today are ~0.006 keV)")
    ap.add_argument("--max-nh", type=float, default=float("inf"),
                    help="drop shock cells denser than this n_H [cm^-3] "
                         "(typical shock cell is ~5e-5)")
    ap.add_argument("--exclude-sf", action="store_true",
                    help="drop star-forming cells")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    interp_psi = make_psi(args.psi_table)

    srcs = sorted(glob.glob(os.path.join(args.radio_dir, "radio_FOF*.npz")))
    if args.limit:
        srcs = srcs[:args.limit]
    print(f"{len(srcs)} clusters")

    worst = 0.0
    t0 = time.time()
    for i, src in enumerate(srcs):
        m = re.search(r"radio_FOF(\d+)_sub(\d+)\.npz", os.path.basename(src))
        fof, sub = m.group(1), m.group(2)
        cutout = os.path.join(args.cutout_dir, f"cutout_sub{sub}_FOF{fof}.hdf5")
        if not os.path.exists(cutout):
            print(f"  MISSING cutout for FOF{fof}")
            continue
        out = generate(cutout, interp_psi,
                       min_T_keV=args.min_temp_kev,
                       max_nH=args.max_nh,
                       exclude_sf=args.exclude_sf)
        if out is None:
            print(f"  FOF{fof}: no shock cells")
            continue
        pos, w, r_kpc = out

        if args.validate:
            # Only meaningful with no cuts applied.
            ref = np.load(src)
            rw = ref["w"]
            if len(rw) != len(w):
                print(f"  FOF{fof}: LENGTH MISMATCH {len(rw)} vs {len(w)}")
            else:
                both = np.isfinite(rw) & np.isfinite(w) & (rw > 0)
                rel = np.abs(w[both] - rw[both]) / rw[both]
                worst = max(worst, np.nanmax(rel) if rel.size else 0.0)
                if i < 3 or np.nanmax(rel) > 1e-6:
                    print(f"  FOF{fof}: n={len(w)} max rel diff "
                          f"{np.nanmax(rel):.3e}")

        np.savez_compressed(
            os.path.join(args.out_dir, f"cells_FOF{fof}_sub{sub}.npz"),
            pos=pos.astype(np.float64), w=w.astype(np.float64),
            r_kpc=r_kpc.astype(np.float64))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(srcs)}  ({time.time()-t0:.0f}s)")

    if args.validate:
        print(f"\nworst relative difference vs stored weights: {worst:.3e}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
