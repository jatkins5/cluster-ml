"""Does the radio model, as coded, keep Hoeft & Brueggen (2007)'s scaling --
and does our density cut change it?

HB07 eq. 32 (and Lee et al. 2024 eq. 9, the energy-dissipation form the
code follows) is, per shocked cell,

    dP/dnu = 5.2e23 W/Hz (xi_e/0.05) (E_diss/1e44 erg/s)
             B^(1+s/2) / (B^2 + B_CMB^2) (nu_obs/1.4 GHz)^(-s/2) Psi(M, T)

Chuiyang's Radio_generation.py (and our build_radio_cells.py, which copies
it) omits the (nu/1.4 GHz)^(-s/2) factor, i.e. it computes 1.4 GHz emission.
We compare with LOFAR at 144 MHz, where that factor is 9.7^(s/2) -- ~10 for a
strong shock (s=2) but ~7000 for the weakest (M=1.3, s=7.8) -- so it changes
the relative weight of cells, not just the overall normalisation.

For every cluster, with and without the n_H < 1e-4 cut, at 1.4 GHz and at
144 MHz: total power, the share carried by the brightest cell, the
emission-weighted Mach number, the share from weak (M<3) shocks, the
integrated spectral index, and across clusters the P-M500 slope and scatter.
"""
import glob
import os
import re

import h5py
import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.constants import k_B
from scipy import stats

from build_radio_cells import (A, ECONV, GAMMA, H0, RHO_CGS, UNIT_B_G, XH, Z,
                               bcmb_uG, make_psi, s_of_M)

NU_LOFAR = 0.144          # GHz
NH_CUT = 1e-4
CUT_DIR = "/oscar/data/idellant/Chuiyang/TNGCluster_Cutout/snap99"


def cells(path, psi):
    with h5py.File(path, "r") as f:
        p0 = f["PartType0"]
        mach = p0["Machnumber"][:]
        edis = p0["EnergyDissipation"][:]
        sh = (mach > 1.3) & (edis > 0)
        dens = p0["Density"][sh]
        elec = p0["ElectronAbundance"][sh]
        uint = p0["InternalEnergy"][sh]
        bfld = p0["MagneticField"][sh]
        mach, edis = mach[sh], edis[sh]
    mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * elec) * c.m_p.cgs.value
    T_keV = (k_B * ((GAMMA - 1.0) * uint / c.k_B.cgs.value * 1e10 * mu * u.K)
             ).to(u.keV).value
    nH = dens.astype(np.float64) * RHO_CGS * XH / c.m_p.cgs.value
    M = mach.astype(np.float64)
    E = edis.astype(np.float64) / A * ECONV * 1e-44
    B = np.sqrt((bfld.astype(np.float64) ** 2).sum(axis=1)) * H0 / A ** 2 \
        * UNIT_B_G * 1e6
    s = s_of_M(M)
    phi = psi(np.column_stack([s, np.log10(T_keV * 10.0 / 511.0)]))
    w14 = 5.2e23 * E * B ** (1 + 0.5 * s) / (B ** 2 + bcmb_uG(Z) ** 2) * phi
    w015 = w14 * (NU_LOFAR / 1.4) ** (-0.5 * s)
    return dict(M=M, nH=nH, w14=w14, w015=w015)


def summarise(c_, keep):
    out = {}
    for tag in ("w14", "w015"):
        w = c_[tag][keep]
        tot = w.sum()
        out[f"logP_{tag}"] = np.log10(tot) if tot > 0 else np.nan
        out[f"top1_{tag}"] = w.max() / tot if tot > 0 else np.nan
        order = np.argsort(c_["M"][keep])
        cw = np.cumsum(w[order]) / tot if tot > 0 else None
        out[f"mach_med_{tag}"] = (c_["M"][keep][order][np.searchsorted(cw, 0.5)]
                                  if tot > 0 else np.nan)
        out[f"weak_{tag}"] = w[c_["M"][keep] < 3].sum() / tot if tot > 0 else np.nan
    out["alpha"] = ((out["logP_w14"] - out["logP_w015"])
                    / np.log10(1.4 / NU_LOFAR))
    return out


def main():
    psi = make_psi("Radio_Data/normalized_psi_table.npz")
    with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
        m500 = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
    rows = {"none": [], "nh4": []}
    files = sorted(glob.glob(os.path.join(CUT_DIR, "cutout_sub*_FOF*.hdf5")))
    for k, p in enumerate(files):
        hid = int(re.search(r"_FOF(\d+)\.hdf5", p).group(1))
        if hid not in m500:
            continue
        cc = cells(p, psi)
        for name, keep in (("none", np.ones(len(cc["M"]), bool)),
                           ("nh4", cc["nH"] < NH_CUT)):
            d = summarise(cc, keep)
            d["logM"] = m500[hid]
            rows[name].append(d)
        if k % 50 == 0:
            print(f"  {k}/{len(files)}", flush=True)

    for name in ("none", "nh4"):
        R = {key: np.array([r[key] for r in rows[name]]) for key in rows[name][0]}
        print(f"\n======== {'no cut' if name == 'none' else 'n_H < 1e-4'}  "
              f"({len(R['logM'])} clusters)")
        print(f"{'':<22}{'1.4 GHz':>12}{'144 MHz':>12}")
        for lab, key in [("top cell share", "top1"),
                         ("emission-wtd Mach", "mach_med"),
                         ("share from M<3", "weak")]:
            print(f"{lab:<22}{np.nanmedian(R[f'{key}_w14']):>12.3f}"
                  f"{np.nanmedian(R[f'{key}_w015']):>12.3f}")
        for tag, lab in (("w14", "1.4 GHz"), ("w015", "144 MHz")):
            ok = np.isfinite(R[f"logP_{tag}"])
            x, y = R["logM"][ok] - 14.9, R[f"logP_{tag}"][ok]
            fit = stats.linregress(x, y)
            print(f"P-M500 at {lab:<8} slope {fit.slope:.2f} +- {fit.stderr:.2f}"
                  f"   scatter {np.std(y - fit.intercept - fit.slope * x):.2f} dex"
                  f"   (Cuciti+23 150 MHz: 3.55, 0.35 dex)")
        a = R["alpha"][np.isfinite(R["alpha"])]
        print(f"integrated spectral index 144 MHz-1.4 GHz: median {np.median(a):.2f}"
              f" (10-90% {np.percentile(a, 10):.2f}-{np.percentile(a, 90):.2f});"
              f" observed relics ~1.0-1.5, the pipeline k-correction assumed 1.2")


if __name__ == "__main__":
    main()
