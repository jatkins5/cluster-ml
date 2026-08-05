"""Joint mass/TSC diagnostic deciding how CAMELS can be combined with TNG.

Two problems surfaced by the threshold sweep:
1. CAMELS zooms span log10 M200 = 9.7-15.7 (median 13.9) while
   TNG-Cluster is a selected massive sample -- most zooms are not
   cluster-scale at all.
2. A 10% mass jump between consecutive snapshots fires ~32 times per
   history, so it is tracking ordinary accretion rather than major
   mergers. A 1:3 major merger implies ~33% growth, so the physically
   motivated raw threshold is much higher.

This compares CAMELS TSC distributions under mass cuts and thresholds
against the TNG-Cluster merger-catalog TSC we actually train on.
"""
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from camels_labels import HIST, TARGET_SNAP, derive_tsc, zoom_cosmology

MASS_CUTS = [0.0, 13.5, 14.0, 14.5]
RAW_THR = [0.10, 0.20, 0.33, 0.50]
RATE_THR = [0.10, 0.20, 0.33]


def main():
    with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5", "r") as f:
        tng_tsc = f["tsc_gyr"][:]
    tng_tsc = tng_tsc[np.isfinite(tng_tsc)]
    print(f"TNG-Cluster TSC: n={len(tng_tsc)} med={np.median(tng_tsc):.2f} "
          f"IQR=[{np.percentile(tng_tsc, 25):.2f}, "
          f"{np.percentile(tng_tsc, 75):.2f}] max={tng_tsc.max():.2f}")

    # cache per-zoom history once
    zooms, hist = [], {}
    with h5py.File(HIST, "r") as f:
        for z in sorted(f.keys()):
            g = f[z]
            snap = g["SnapNum"][:]
            i90 = np.where(snap == TARGET_SNAP)[0]
            if len(i90) == 0:
                continue
            om0, h = zoom_cosmology(z)
            hist[z] = dict(m200=g["M200_Msun"][:], snap=snap,
                           z=g["Redshift"][:],
                           passes=g["passes_contamination_cut"][:],
                           om0=om0, h=h,
                           logm=np.log10(g["M200_Msun"][i90[0]]))
            zooms.append(z)
    logm = np.array([hist[z]["logm"] for z in zooms])
    print(f"\nCAMELS log10 M200 at snap {TARGET_SNAP}: "
          f"min={logm.min():.2f} med={np.median(logm):.2f} "
          f"max={logm.max():.2f}")
    for c in MASS_CUTS[1:]:
        print(f"  zooms with log10 M200 > {c}: {(logm > c).sum()}/{len(logm)}")

    def tsc_for(thr, use_rate):
        out = np.empty(len(zooms))
        njump = np.empty(len(zooms))
        for i, z in enumerate(zooms):
            d = hist[z]
            t, nj, _ = derive_tsc(d["m200"], d["snap"], d["z"], d["passes"],
                                  d["om0"], d["h"], threshold=thr,
                                  use_rate=use_rate)
            out[i], njump[i] = t, nj
        return out, njump

    print("\n=== raw fractional jump ===")
    print(f"{'thr':>6} {'masscut':>8} {'n':>5} {'med TSC':>9} {'IQR':>15} "
          f"{'med jumps':>10} {'KS vs TNG':>10}")
    results = {}
    for thr in RAW_THR:
        tsc, nj = tsc_for(thr, False)
        for c in MASS_CUTS:
            m = logm > c
            if m.sum() < 20:
                continue
            t = tsc[m]
            ks = stats.ks_2samp(t, tng_tsc)
            print(f"{thr:>6.2f} {c:>8.1f} {m.sum():>5} "
                  f"{np.median(t):>9.2f} "
                  f"{f'[{np.percentile(t,25):.2f}, {np.percentile(t,75):.2f}]':>15} "
                  f"{int(np.median(nj[m])):>10} "
                  f"{ks.statistic:>10.3f}")
            results[("raw", thr, c)] = t

    print("\n=== growth rate per Gyr ===")
    print(f"{'thr':>6} {'masscut':>8} {'n':>5} {'med TSC':>9} {'IQR':>15} "
          f"{'med jumps':>10} {'KS vs TNG':>10}")
    for thr in RATE_THR:
        tsc, nj = tsc_for(thr, True)
        for c in MASS_CUTS:
            m = logm > c
            if m.sum() < 20:
                continue
            t = tsc[m]
            ks = stats.ks_2samp(t, tng_tsc)
            print(f"{thr:>6.2f} {c:>8.1f} {m.sum():>5} "
                  f"{np.median(t):>9.2f} "
                  f"{f'[{np.percentile(t,25):.2f}, {np.percentile(t,75):.2f}]':>15} "
                  f"{int(np.median(nj[m])):>10} "
                  f"{ks.statistic:>10.3f}")
            results[("rate", thr, c)] = t

    best = min(results, key=lambda k: stats.ks_2samp(results[k],
                                                     tng_tsc).statistic)
    print(f"\nclosest match to the TNG TSC distribution: {best} "
          f"(KS D={stats.ks_2samp(results[best], tng_tsc).statistic:.3f}, "
          f"n={len(results[best])})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.hist(logm, bins=40)
    for c in MASS_CUTS[1:]:
        ax.axvline(c, ls=":", color="k")
    ax.set_xlabel("log10 M200 [Msun] at snap 90")
    ax.set_ylabel("zooms")
    ax.set_title("CAMELS halo masses (TNG-Cluster: 14.2-15.5 in M200)")

    ax = axes[1]
    bins = np.linspace(0, 14, 40)
    ax.hist(tng_tsc, bins=bins, density=True, histtype="step", lw=2,
            color="k", label="TNG merger-catalog")
    for key in [("raw", 0.10, 0.0), ("raw", 0.33, 14.0),
                ("rate", 0.10, 14.0), best]:
        if key in results:
            ax.hist(results[key], bins=bins, density=True, alpha=0.35,
                    label=f"{key[0]} thr={key[1]} M>{key[2]}")
    ax.set_xlabel("TSC [Gyr]")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.set_title("TSC distributions")

    ax = axes[2]
    tsc10, _ = tsc_for(0.33, False)
    ax.scatter(logm, tsc10, s=5, alpha=0.4)
    ax.set_xlabel("log10 M200")
    ax.set_ylabel("TSC [Gyr] (raw thr=0.33)")
    ax.set_title("mass vs TSC")
    fig.tight_layout()
    fig.savefig("camels_mass_tsc.png", dpi=150)
    print("figure saved -> camels_mass_tsc.png")


if __name__ == "__main__":
    main()
