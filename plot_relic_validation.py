#!/usr/bin/env python3
"""
Validate the Mach-derived relic catalog against Lee's relic-distance / TSC
relation.

Plots merger-TSC vs d_max (the most-outer detected peak), stratified by:
  - mass_ratio (major vs minor mergers)
  - n_relics (clean: <=2 / complex: >=3 per projection)
overlays v_sh = 1500 km/s × TSC reference, and reports correlations per
stratum. The diagnostic verdict is: does the *major-merger / clean* subset
show a positive correlation matching the reference slope?

Reads relic_catalog.h5 (or --catalog override). Writes relic_validation.png
and relic_count_hist.png.
"""

import argparse
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KPC_PER_MPC = 1000.0
KMS_TO_KPC_PER_GYR = 1.022


def corr(x, y, mask):
    return float(np.corrcoef(x[mask], y[mask])[0, 1]) if mask.sum() >= 5 else np.nan


def main(args):
    with h5py.File(args.catalog, "r") as f:
        attrs = dict(f.attrs)
        n_rel = f["n_relics"][:]
        d_pri = f["d_primary"][:]
        d_max = f["d_max"][:]
        pseudo = f["pseudo_tsc"][:]
        merger = f["merger_tsc"][:]
        mr = f["mass_ratio"][:]

    d_max_mean = np.nanmean(d_max, axis=1)
    d_pri_mean = np.nanmean(d_pri, axis=1)
    n_mean = n_rel.mean(axis=1)

    # use merger-TSC as primary label (Lee's relation is in observational TSC)
    tsc = merger
    valid = np.isfinite(tsc) & np.isfinite(d_max_mean)
    clean = valid & (n_rel.max(axis=1) <= 2)
    major = valid & (mr >= 0.25)
    minor = valid & (mr < 0.25)
    major_clean = major & clean

    print(f"detector attrs: {attrs}")
    print(f"valid (merger-TSC + d_max): {valid.sum()}/{len(tsc)}")
    print(f"  major (mr>=0.25)     : {major.sum()}")
    print(f"  minor (mr<0.25)      : {minor.sum()}")
    print(f"  clean (n_rel max<=2) : {clean.sum()}")
    print(f"  major + clean        : {major_clean.sum()}")
    print(f"n_relics median across projections: {np.median(n_rel)}")

    print(f"\nCorrelation merger-TSC vs d_max:")
    print(f"  all valid        : {corr(tsc, d_max_mean, valid):+.3f}")
    print(f"  major mergers    : {corr(tsc, d_max_mean, major):+.3f}")
    print(f"  minor mergers    : {corr(tsc, d_max_mean, minor):+.3f}")
    print(f"  clean (any mr)   : {corr(tsc, d_max_mean, clean):+.3f}")
    print(f"  major + clean    : {corr(tsc, d_max_mean, major_clean):+.3f}")

    print(f"\nCorrelation merger-TSC vs d_primary:")
    print(f"  major mergers    : {corr(tsc, d_pri_mean, major):+.3f}")
    print(f"  major + clean    : {corr(tsc, d_pri_mean, major_clean):+.3f}")

    # reference: d = v_sh × TSC
    v_sh = 1500.0
    slope = v_sh * KMS_TO_KPC_PER_GYR
    tau = np.linspace(0, 8, 100)
    d_ref = slope * tau

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    for k, (d, label) in enumerate([(d_max_mean, "d_max (outermost peak)"),
                                     (d_pri_mean, "d_primary (brightest)")]):
        a = ax[k]
        # background: minor + complex (gray)
        bg = valid & ~major_clean
        a.scatter(tsc[bg], d[bg] / KPC_PER_MPC, s=18, alpha=0.3, c="C7",
                  label=f"other [{bg.sum()}]")
        # foreground: major + clean (the relic-bearing analog)
        sc = a.scatter(tsc[major_clean], d[major_clean] / KPC_PER_MPC,
                       s=70, alpha=0.85, c=mr[major_clean], cmap="plasma",
                       edgecolor="k", linewidth=0.5,
                       label=f"major (mr≥0.25) + clean (n≤2) [{major_clean.sum()}]")
        # all-major in between
        mid = major & ~clean
        a.scatter(tsc[mid], d[mid] / KPC_PER_MPC, s=30, alpha=0.6, c="C0",
                  marker="x", label=f"major + complex [{mid.sum()}]")
        a.plot(tau, d_ref / KPC_PER_MPC, "k--", lw=1.5,
               label=f"d = {v_sh:.0f} km/s × TSC")
        a.set_xlabel("merger-TSC (Gyr)")
        a.set_ylabel("relic distance (Mpc, averaged across projections)")
        a.set_title(f"{label}\n"
                    f"corr major+clean={corr(tsc, d, major_clean):+.2f}  "
                    f"major={corr(tsc, d, major):+.2f}  "
                    f"all={corr(tsc, d, valid):+.2f}")
        a.set_xlim(0, 8)
        a.set_ylim(0, 6)
        a.legend(loc="upper right", fontsize=8)
        if k == 0:
            cb = plt.colorbar(sc, ax=a)
            cb.set_label("mass_ratio")

    title_extra = (f"  (mach≥{attrs.get('mach_thresh','?')}, "
                   f"peak≥{attrs.get('peak_thresh_frac', '?')}, "
                   f"sep≥{attrs.get('min_sep_kpc', '?')}kpc)")
    fig.suptitle("TNG-Cluster relic distance vs merger-TSC" + title_extra,
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.outprefix + "_validation.png", dpi=120)
    print(f"saved {args.outprefix}_validation.png")

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.hist(n_rel.flatten(), bins=np.arange(0, 30) - 0.5,
             color="C0", edgecolor="k")
    ax2.set_xlabel("n_relics detected per projection")
    ax2.set_ylabel("count (over 352 × 3 projections)")
    ax2.set_title(f"detector cleanness  "
                  f"(mach≥{attrs.get('mach_thresh','?')})")
    ax2.axvline(2.5, color="r", ls="--", label="clean / complex split")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(args.outprefix + "_count_hist.png", dpi=120)
    print(f"saved {args.outprefix}_count_hist.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=str, default="relic_catalog.h5")
    p.add_argument("--outprefix", type=str, default="relic")
    main(p.parse_args())
