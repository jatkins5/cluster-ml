#!/usr/bin/env python3
"""What changed between Chuiyang's Radio_Snap{99,91} and Radio_Snap{99,91}_new
(read-only on her directories).

Old weight (HB07 Psi, normalised table, 144 MHz via (nu/1.4 GHz)^(-s/2)):
  w_old = 5.2e23 E B^(1+s/2)/(B^2+Bcmb^2) psi_norm(s, e_min) (0.144/1.4)^(-s/2)
New weight (first-principles synchrotron coefficients, cal_coefficients.ipynb):
  w_new = 8.70e34 (xi_e/0.05) E SPECTRAL_BASE^(s/2) (I_Psi/I_spec)(s, e_min)
          B^(1+s/2)/(B^2+Bcmb^2)
E, B and Bcmb cancel in w_new/w_old, so the per-cell change is a function of
(Mach, T) only: f(M, T). This prints f on a grid, then compares the stored
weights halo by halo (same cells, same order).
"""
import glob
import os

import numpy as np
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

C = "/oscar/data/idellant/Chuiyang"
GAMMA = 5.0 / 3.0
POWER_PREF, SPECTRAL_BASE = 8.701249739473983e34, 2.290130219242185e-08


def s_of_M(M):
    r = (GAMMA + 1) * M ** 2 / ((GAMMA - 1) * M ** 2 + 2)
    return (r + 2) / (r - 1)


old = np.load(f"{C}/Radio_Snap99/normalized_psi_table.npz")
new = np.load(f"{C}/Radio_Snap99_new/radio_spectral_ratio_table.npz")
print("new table keys:", new.files, "e_max", float(new["e_max"]), "gamma", float(new["gamma"]))
f_old = RegularGridInterpolator((old["s_grid"], old["loge_grid"]), old["normalized_psi_grid"],
                                bounds_error=False, fill_value=None)
f_new = RegularGridInterpolator((new["s_grid"], new["loge_grid"]), new["log10_spectral_ratio"],
                                bounds_error=False, fill_value=None)


def ratio(M, T_keV):
    s = s_of_M(M)
    le = np.log10(T_keV * 10.0 / 511.0)
    pts = np.column_stack([np.broadcast_to(s, np.shape(le)).ravel(), np.ravel(le)])
    w_old = 5.2e23 * f_old(pts) * (0.144 / 1.4) ** (-0.5 * s)
    w_new = POWER_PREF * SPECTRAL_BASE ** (0.5 * s) * 10 ** f_new(pts)
    return (w_new / w_old).reshape(np.shape(le))


Ms = np.array([1.3, 1.5, 2, 2.5, 3, 4, 5, 8, 15, 30])
Ts = np.array([0.006, 0.1, 1.0, 5.0])
print("\nf(M, T) = w_new / w_old  (rows Mach, columns T [keV])")
print("  M     " + "".join(f"{t:>11.3g}" for t in Ts))
for M in Ms:
    print(f"  {M:<6}" + "".join(f"{ratio(M, np.array([t]))[0]:>11.3e}" for t in Ts))
spread = max(np.ptp([ratio(M, np.array([t]))[0] for t in Ts]) / ratio(M, np.array([1.0]))[0] for M in Ms)
print(f"\nlargest relative spread of f across the sampled T at fixed Mach: {spread:.4f} "
      f"(approximately Mach-dependent over these temperatures, not exactly)")
ref = ratio(4.0, np.array([1.0]))[0]
print("\nnormalised to M=4, T=1 keV (the shape change that matters for maps):")
for M in Ms:
    print(f"  M={M:<5}" + "".join(f"{ratio(M, np.array([t]))[0] / ref:>11.3g}" for t in Ts))

for snap in (99, 91):
    od, nd = f"{C}/Radio_Snap{snap}", f"{C}/Radio_Snap{snap}_new"
    files = sorted(os.path.basename(p) for p in glob.glob(f"{od}/radio_FOF*.npz"))
    newf = sorted(os.path.basename(p) for p in glob.glob(f"{nd}/radio_FOF*.npz"))
    print(f"\n== snapshot {snap}: {len(files)} old, {len(newf)} new files; "
          f"same names: {files == newf}")
    rows = []
    for fn in files:
        a, b = np.load(f"{od}/{fn}"), np.load(f"{nd}/{fn}")
        if a["w"].shape != b["w"].shape or not np.allclose(a["pos"], b["pos"]):
            print(f"  {fn}: cells differ ({a['w'].shape} vs {b['w'].shape})")
            continue
        wo, wn = a["w"].astype(float), b["w"].astype(float)
        ok = (wo > 0) & (wn > 0) & np.isfinite(wo) & np.isfinite(wn)
        lr = np.log10(wn[ok] / wo[ok])
        rho = stats.spearmanr(wo[ok], wn[ok])[0] if ok.sum() > 2 else np.nan
        top_o, top_n = wo.max() / wo.sum(), wn.max() / wn.sum()
        same_top = int(np.argmax(wo) == np.argmax(wn))
        co = (a["pos"] * wo[:, None]).sum(0) / wo.sum()
        cn = (b["pos"] * wn[:, None]).sum(0) / wn.sum()
        rows.append((wn.sum() / wo.sum(), np.median(lr), np.percentile(lr, 5),
                     np.percentile(lr, 95), rho, top_o, top_n, same_top,
                     np.linalg.norm(cn - co), int((~ok).sum())))
    r = np.array(rows)
    print(f"  same top cell in {int(r[:, 7].sum())} of {len(r)} halos")
    names = ["total ratio new/old", "median log10 cell ratio", "5% log10 ratio",
             "95% log10 ratio", "Spearman(w_old, w_new)", "top-cell fraction old",
             "top-cell fraction new", "same top cell", "weighted-centroid shift [kpc]",
             "non-positive/non-finite cells"]
    for j, nm in enumerate(names):
        print(f"  {nm:<32} median {np.median(r[:, j]):.4g}   "
              f"[{np.percentile(r[:, j], 5):.4g}, {np.percentile(r[:, j], 95):.4g}] (5-95%)")
