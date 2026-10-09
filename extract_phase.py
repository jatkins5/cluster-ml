"""Gas state of the radio-emitting shock cells, for the phase-diagram figure.

For every snapshot-99 cluster, at 144 MHz and with no density cut: a 2-D
histogram over (log n_H, log T) of shock-cell count and of radio weight,
each normalised per cluster so that every cluster counts equally; plus each
cluster's weight fraction above the n_H < 1e-4 cut, its top-cell share with
and without the cut, and the weight-averaged Mach number either way.
"""
import glob
import os
import re

import numpy as np

import build_radio_cells as brc

CUT = "/oscar/data/idellant/Chuiyang/TNGCluster_Cutout/snap99"
NH_EDGES = np.linspace(-7.0, 0.0, 71)        # log10 n_H [cm^-3]
T_EDGES = np.linspace(-3.5, 1.5, 51)         # log10 kT [keV]
NH_CUT = 1e-4

psi = brc.make_psi("Radio_Data/normalized_psi_table.npz")
brc.set_redshift(0.0)
H_n = np.zeros((len(NH_EDGES) - 1, len(T_EDGES) - 1))
H_w = np.zeros_like(H_n)
frac_above, top_all, top_cut, mach_all, mach_cut, hids = [], [], [], [], [], []
for src in sorted(glob.glob("Radio_Data/radio_FOF*.npz")):
    fof, sub = re.search(r"radio_FOF(\d+)_sub(\d+)", src).groups()
    path = os.path.join(CUT, f"cutout_sub{sub}_FOF{fof}.hdf5")
    if not os.path.exists(path):
        continue
    out = brc.generate(path, psi, nu_ghz=0.144, phase=True)
    if out is None:
        continue
    _, w, _, T, nH, M = out
    ok = np.isfinite(w) & (w > 0) & (nH > 0) & (T > 0)
    w, T, nH, M = w[ok], T[ok], nH[ok], M[ok]
    x, y = np.log10(nH), np.log10(T)
    hn, _, _ = np.histogram2d(x, y, [NH_EDGES, T_EDGES])
    hw, _, _ = np.histogram2d(x, y, [NH_EDGES, T_EDGES], weights=w)
    H_n += hn / hn.sum()
    H_w += hw / hw.sum()
    keep = nH < NH_CUT
    frac_above.append(w[~keep].sum() / w.sum())
    top_all.append(w.max() / w.sum())
    top_cut.append(w[keep].max() / w[keep].sum() if keep.any() else np.nan)
    mach_all.append((w * M).sum() / w.sum())
    mach_cut.append((w[keep] * M[keep]).sum() / w[keep].sum() if keep.any() else np.nan)
    hids.append(int(fof))
    if len(hids) % 50 == 0:
        print(f"  {len(hids)} clusters", flush=True)

n = len(hids)
np.savez("phase_cells_144.npz", H_n=H_n / n, H_w=H_w / n, nh_edges=NH_EDGES,
         t_edges=T_EDGES, nh_cut=NH_CUT, halo_id=np.array(hids),
         frac_above=np.array(frac_above), top_all=np.array(top_all),
         top_cut=np.array(top_cut), mach_all=np.array(mach_all),
         mach_cut=np.array(mach_cut))
print(f"{n} clusters; weight above the cut: median {np.median(frac_above):.2f}; "
      f"top-cell share {np.median(top_all):.3f} -> {np.nanmedian(top_cut):.3f}; "
      f"weight-averaged Mach {np.median(mach_all):.1f} -> {np.nanmedian(mach_cut):.1f}")
