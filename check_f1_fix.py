"""Review F1 acceptance checks for the new radio injection recipe.

1. Flux conservation of rebin_flux: for every cluster and projection, the
   rebinned box must hold exactly the linear emission inside the box. The
   reference is an independent sum on 8x-supersampled native pixels; boxes
   have fractional edges, so it must handle partial pixels.
2. Legacy vs new power estimator, for 1/2/3 Mpc boxes, under BOTH averaging
   conventions: the figure's mean over projections of log10(power), and
   log10 of the mean. Reports the offset, its scatter, the residual slope
   against log r500, and the power-mass slope and scatter.
"""
import h5py
import numpy as np
from scipy import stats

from forward_model_lotss import crop_resample, rebin_flux

with h5py.File("dataset_nh4_144_512.h5") as f:
    img = f["images"][:].astype(np.float64)
    r500 = f["meta/r500c_kpc"][:]
    lm = f["meta/log_m500"][:] if "meta/log_m500" in f else None
    hid = f["meta/halo_id"][:]
if lm is None:
    with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
        mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
    lm = np.array([mm[h] for h in hid])
N, P, H, _ = img.shape
lin = np.sinh(img)

print("======== 1. flux conservation")
# Independent reference: supersample each native pixel ss x ss, keep the
# sub-pixels whose centres fall inside the box. Its edge error is <= 1/(2 ss)
# of a pixel per edge, so if rebin_flux is exact the measured difference must
# shrink ~in proportion to 1/ss.
for ss in (8, 64, 512):
    worst, n = 0.0, 0
    for i in range(0, N, 7):                   # every 7th cluster, all projections
        px = 8 * r500[i] / H
        for box in (1000.0, 2000.0, 3000.0):
            hp = box / 2 / px
            c = (np.arange(H * ss) + 0.5) / ss
            inside = (c >= H / 2 - hp) & (c < H / 2 + hp)
            frac = inside.reshape(H, ss).mean(axis=1)          # per native px
            for p in range(P):
                out = rebin_flux(lin[i, p], hp)
                ref = (frac[:, None] * frac[None, :] * lin[i, p]).sum()
                worst = max(worst, abs(out.sum() - ref) / ref)
                n += 1
    print(f"  reference at 1/{ss} pixel: {n} boxes, max relative difference {worst:.2e}")

import sys
if "--conservation-only" in sys.argv:
    sys.exit(0)
print("\n======== 2. power estimators, legacy vs new")
for box in (1000.0, 2000.0, 3000.0):
    leg, new = np.zeros((N, P)), np.zeros((N, P))
    for i in range(N):
        hp = box / 2 / (8 * r500[i] / H)
        for p in range(P):
            leg[i, p] = np.sinh(np.clip(crop_resample(img[i, p], hp), 0, 40)).sum()
            new[i, p] = rebin_flux(lin[i, p], hp).sum()
    print(f"  box {box / 1000:.0f} Mpc")
    for conv, f_ in (("mean of log10 (figure)", lambda a: np.log10(a).mean(1)),
                     ("log10 of mean", lambda a: np.log10(a.mean(1)))):
        L, Nw = f_(leg), f_(new)
        d = L - Nw
        sr = stats.linregress(np.log10(r500), d)
        fl, fn = stats.linregress(lm, L), stats.linregress(lm, Nw)
        sc = lambda y, fit: np.std(y - fit.intercept - fit.slope * lm)
        print(f"    {conv:<22} legacy - new: median {np.median(d):+.2f}, sd "
              f"{d.std():.2f} dex, vs log r500 {sr.slope:+.2f} +- {sr.stderr:.2f}; "
              f"P-M slope legacy {fl.slope:.2f} ({sc(L, fl):.2f} dex) new "
              f"{fn.slope:.2f} ({sc(Nw, fn):.2f} dex)")
