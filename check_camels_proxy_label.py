"""Sanity-check the CAMELS proxy TSC before it is used for training.

The combine stage reported a maximum TSC of 21.9 Gyr, which is older than
the universe in any standard cosmology, so this checks whether the per-zoom
cosmologies are the cause and how the label compares with the TNG one it
has to be trained alongside.
"""
import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

TNG_LABEL = "TSC_Cutimages/TSC_eachhalo_snap99.hdf5"


def main():
    d = np.load("camels_proxy_parts/zooms_0000_0048.npz")
    print("cosmology across the first 48 zooms:")
    for k in ("om0", "hub"):
        v = d[k]
        print(f"  {k}: min={np.nanmin(v):.4f} med={np.nanmedian(v):.4f} "
              f"max={np.nanmax(v):.4f}")

    om = np.concatenate([np.load(f"camels_proxy_parts/zooms_{a:04d}_{a+48:04d}"
                                 ".npz")["om0"] for a in range(0, 768, 48)])
    hb = np.concatenate([np.load(f"camels_proxy_parts/zooms_{a:04d}_{a+48:04d}"
                                 ".npz")["hub"] for a in range(0, 768, 48)])
    print(f"\nall 768 zooms:  Omega0 [{np.nanmin(om):.3f}, "
          f"{np.nanmax(om):.3f}]   h [{np.nanmin(hb):.3f}, "
          f"{np.nanmax(hb):.3f}]")

    age0 = np.array([FlatLambdaCDM(H0=100 * h, Om0=o).age(0).to("Gyr").value
                     for o, h in zip(om, hb)])
    print(f"implied age of universe at z=0: [{age0.min():.2f}, "
          f"{age0.max():.2f}] Gyr, median {np.median(age0):.2f}")
    print(f"zooms with age > 13.8 Gyr: {(age0 > 13.8).sum()}/{len(age0)}")

    with h5py.File("camels_proxy_tsc.hdf5", "r") as f:
        tsc = f["tsc_proxy"][:]
        span = f["span_gyr"][:]
    print(f"\nTSC vs its own zoom's age: max ratio "
          f"{np.nanmax(tsc / age0):.3f} (must be <= 1)")
    print(f"zooms with TSC > 13.8 Gyr: {(tsc > 13.8).sum()}")
    print(f"  their h:      {np.round(hb[tsc > 13.8][:8], 3)}")
    print(f"  their Omega0: {np.round(om[tsc > 13.8][:8], 3)}")
    print(f"  their age:    {np.round(age0[tsc > 13.8][:8], 2)}")

    with h5py.File(TNG_LABEL, "r") as f:
        tng = f["tsc_gyr"][:]
    tng = tng[np.isfinite(tng)]
    print(f"\n{'':<12}{'median':>8}{'IQR25':>8}{'IQR75':>8}{'max':>8}"
          f"{'<=2 Gyr':>9}")
    for name, v in (("TNG label", tng), ("CAMELS proxy", tsc)):
        v = v[np.isfinite(v)]
        print(f"{name:<12}{np.median(v):>8.2f}{np.percentile(v, 25):>8.2f}"
              f"{np.percentile(v, 75):>8.2f}{v.max():>8.2f}"
              f"{100 * (v <= 2).mean():>8.1f}%")

    from scipy import stats
    ks = stats.ks_2samp(tng, tsc[np.isfinite(tsc)])
    print(f"\nKS distance TNG vs CAMELS proxy: D={ks.statistic:.3f} "
          f"(mass-jump label managed D=0.169)")
    clip = np.clip(tsc[np.isfinite(tsc)], 0, tng.max())
    print(f"KS after clipping CAMELS at the TNG max ({tng.max():.2f} Gyr): "
          f"D={stats.ks_2samp(tng, clip).statistic:.3f}")


if __name__ == "__main__":
    main()
