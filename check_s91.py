"""Sanity checks on the assembled snapshot-91 sets, before training.

The same clusters 1.3 Gyr earlier should look similar but not identical:
slightly lower masses, labels that move, comparable radio power and X-ray
brightness once placed at the same redshift. Gross differences (factors of
several in counts or power, image extents off) would mean a units or
geometry slip in the new path, not physics.
"""
import h5py
import numpy as np
from scipy import stats


def load(p, grp):
    with h5py.File(p) as f:
        return {k: f[f"{grp}/{k}"][:] for k in f[grp]}, dict(f.attrs)


def by_halo(a, b, key="halo_id"):
    ia = {h: i for i, h in enumerate(a[key])}
    common = [h for h in b[key] if h in ia]
    return (np.array([ia[h] for h in common]),
            np.array([list(b[key]).index(h) for h in common]))


print("== clean datasets (128 px)")
d99, a99 = load("dataset_nh4_144_128.h5", "meta")
d91, a91 = load("dataset_nh4_144_s91_128.h5", "meta")
print(f"  snap 91: {len(d91['halo_id'])} clusters, z = {a91.get('redshift')}")
i, j = by_halo(d99, d91)
with h5py.File("dataset_nh4_144_128.h5") as f:
    t99, im99 = f["labels/pseudo_tsc"][:], f["images"][:]
with h5py.File("dataset_nh4_144_s91_128.h5") as f:
    t91, im91, lm91 = f["labels/pseudo_tsc"][:], f["images"][:], f["meta/log_m500"][:]
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    cm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
lm99 = np.array([cm[h] for h in d99["halo_id"]])
print(f"  log M500 (91 - 99): median {np.median(lm91[j] - lm99[i]):+.3f} dex; "
      f"R500 ratio {np.median(d91['r500c_kpc'][j] / d99['r500c_kpc'][i]):.3f}")
print(f"  pseudo-TSC: 99 median {np.median(t99):.2f} (cap {np.mean(t99 >= 4):.0%}), "
      f"91 median {np.median(t91):.2f} (cap {np.mean(t91 >= 4):.0%}); "
      f"Spearman {stats.spearmanr(t99[i], t91[j])[0]:+.2f}")
tot = lambda im: np.log10(np.sinh(im.astype(np.float64)).sum(axis=(2, 3)).mean(1))
d = tot(im91[j]) - tot(im99[i])
print(f"  total radio weight per map, log (91 - 99): median {np.median(d):+.2f}, "
      f"sd {d.std():.2f}")
fill = lambda im: (im > 0).mean(axis=(1, 2, 3))
print(f"  pixel fill fraction: 99 {np.median(fill(im99)):.3f}, 91 {np.median(fill(im91)):.3f}")

print("\n== injected radio mocks")
r99, b99 = load("injected_nh4_144_simflux.h5", "mock")
r91, b91 = load("injected_nh4_144_s91_simflux.h5", "mock")
print(f"  flux offsets: 99 {b99.get('sim_flux_offset_dex'):.3f}, "
      f"91 {b91.get('sim_flux_offset_dex'):.3f} (must be equal)")
i, j = by_halo(r99, r91)
lp = lambda r: np.nanmean(r["logp150"], axis=(1, 2))
d = lp(r91)[j] - lp(r99)[i]
print(f"  log P150 (91 - 99): median {np.nanmedian(d):+.2f}, sd {np.nanstd(d):.2f} dex")
for s, r in (("99", r99), ("91", r91)):
    print(f"  snap {s}: {len(r['halo_id'])} clusters, z draws "
          f"{r['z'].min():.3f}-{r['z'].max():.3f}, mock image median "
          f"{np.median(r['images'][:, 0]):.3f}")
with h5py.File("injected_nh4_simflux.h5") as f:
    z14 = f["mock/z"][:]
print(f"  1.4 GHz vs 144 MHz (snap 99) mocks use identical redshift draws: "
      f"{np.array_equal(z14, r99['z'])}")

print("\n== X-ray")
with h5py.File("xray_counts_2Ms_128.h5") as f:
    c99, h99 = f["counts"][:], f["meta/halo_id"][:]
with h5py.File("xray_counts_2Ms_128_s91.h5") as f:
    c91, h91 = f["counts"][:], f["meta/halo_id"][:]
    zm = f.attrs["z_mock"]
p99 = {h: k for k, h in enumerate(h99)}
k99 = np.array([p99[h] for h in h91])
from build_xray_realistic import distance_factor
dimming = distance_factor(zm)          # D_L(0.05)^2 / D_L(z_mock)^2
ratio = c91.sum(axis=(2, 3)).mean(1) / c99[k99].sum(axis=(2, 3)).mean(1)
print(f"  2 Ms counts, 91 / 99 for the same cluster: median {np.median(ratio):.3f} "
      f"(distance alone predicts {dimming:.3f}; the rest is 1.3 Gyr of evolution "
      f"and the 91 mocks covering more of the cluster)")
for p in ("xray_placed_144.h5", "xray_placed_144_s91.h5"):
    with h5py.File(p) as f:
        im = f["mock/images"][:]
        print(f"  {p}: z_mock {f.attrs.get('z_mock')}, stretch "
              f"{f.attrs['stretch_scale']:.4g}, image median {np.median(im):.3f}, "
              f"99th {np.percentile(im, 99):.3f}")
with h5py.File("xray_placed_144_s9991.h5") as f:
    print(f"  merged X-ray: {f['mock/images'].shape}, obs "
          f"{f['obs/images'].shape if 'obs' in f else None}")
