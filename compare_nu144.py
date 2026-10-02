"""How much does per-cell 144 MHz emission differ from rescaled 1.4 GHz?

HB07's (nu / 1.4 GHz)^(-s/2) depends on each cell's Mach number, so going
to 144 MHz brightens weak-shock cells relative to strong ones. Both cell sets
are in 1.4 GHz units (the 144 MHz weights were divided by the sample-median
ratio, --w-scale), so a pure global factor would show up here as ratio = 1.

Reports, per cluster and projection:
  - total-flux ratio 144/1400: its median (should be ~1) and scatter in dex;
    the scatter is the only flux change the sim-flux mocks can see
  - flux displaced: half the L1 distance between the two flux-normalised
    maps, i.e. the fraction of emission that moves pixel
  - Pearson r of the arcsinh images the CNN sees
and whether the flux ratio tracks merger time.
"""
import h5py
import numpy as np
from scipy import stats

for size in (512, 128):
    with h5py.File(f"dataset_nh4_{size}.h5") as f:
        a = f["images"][:].astype(np.float64)
        tsc = f["labels/pseudo_tsc"][:]
        h1 = f["meta/halo_id"][:]
    with h5py.File(f"dataset_nh4_144_{size}.h5") as f:
        b = f["images"][:].astype(np.float64)
        h2 = f["meta/halo_id"][:]
    assert np.array_equal(h1, h2)
    la, lb = np.sinh(a), np.sinh(b)
    ta, tb = la.sum(axis=(2, 3)), lb.sum(axis=(2, 3))
    ratio = np.log10(tb / ta)                                   # (N, 3)
    na = la / ta[..., None, None]
    nb = lb / tb[..., None, None]
    moved = 0.5 * np.abs(na - nb).sum(axis=(2, 3))
    r = np.array([[np.corrcoef(a[i, p].ravel(), b[i, p].ravel())[0, 1]
                   for p in range(3)] for i in range(len(a))])
    print(f"== {size}px  ({len(a)} clusters x 3 projections)")
    print(f"  log10 flux ratio 144/1400 (in 1.4 GHz units): median "
          f"{np.median(ratio):+.3f}, sd {ratio.std():.3f} dex, 5-95% "
          f"{np.percentile(ratio, 5):+.3f} .. {np.percentile(ratio, 95):+.3f}")
    print(f"  flux displaced between pixels: median {np.median(moved):.3f}, "
          f"95% {np.percentile(moved, 95):.3f}, max {moved.max():.3f}")
    print(f"  arcsinh image Pearson r: median {np.median(r):.4f}, "
          f"5% {np.percentile(r, 5):.4f}, min {r.min():.4f}")
    rho, p = stats.spearmanr(ratio.mean(1), tsc)
    print(f"  flux ratio vs pseudo-TSC: Spearman {rho:+.3f} (p={p:.2g})")

with h5py.File("injected_nh4_simflux.h5") as f:
    pa = f["mock/logp150"][:]
with h5py.File("injected_nh4_144_simflux.h5") as f:
    pb = f["mock/logp150"][:]
    off = float(f.attrs["sim_flux_offset_dex"])
d = pb - pa
print(f"== injected mocks: logP150 change median {np.median(d):+.3f}, "
      f"sd {d.std():.3f} dex (new global offset {off:.3f})")
