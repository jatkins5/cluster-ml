#!/usr/bin/env python3
"""
Refined relic-catalog validation: per-projection treatment, restricted to
the bow-shock-active TSC regime, with relaxed major-merger threshold.

The v1/v2 validations averaged d across the 3 projection axes and required
mass_ratio >= 0.25, which both shrank statistics and mixed orientations.
Here each (halo × projection) is one observational analog (3x the sample),
we restrict to merger-TSC in [tsc_lo, tsc_hi] where Lee's bow-shock linear
relation is expected to hold, and relax to mr >= 0.1.

Plots 4 strata of increasing strictness so the trend (or its absence) is
visible at a glance. The decisive panel is the "single-relic, active,
significant" one — that's the cleanest observational analog of Lee's
single-relic / BCG-distance subsample.
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


def panel(ax, tsc, d, mask, mr, tag, ref):
    n = int(mask.sum())
    if n < 1:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        ax.set_title(f"{tag}  [n=0]")
        return
    c = corr(tsc, d, mask)
    sc = ax.scatter(tsc[mask], d[mask] / KPC_PER_MPC, s=22, alpha=0.7,
                    c=mr[mask], cmap="plasma", edgecolor="k", linewidth=0.3)
    tau = np.linspace(0, 8, 100)
    ax.plot(tau, ref * tau / KPC_PER_MPC, "k--", lw=1.2, label="Lee 1500 km/s")
    ax.set_xlabel("merger-TSC (Gyr)")
    ax.set_ylabel("d (Mpc)")
    ax.set_title(f"{tag}  [n={n}]  corr={c:+.3f}")
    ax.set_xlim(0, 8); ax.set_ylim(0, 4)
    ax.legend(loc="upper right", fontsize=8)
    return sc


def main(args):
    with h5py.File(args.catalog, "r") as f:
        attrs = dict(f.attrs)
        n_rel = f["n_relics"][:]
        d_pri = f["d_primary"][:]
        d_max = f["d_max"][:]
        merger = f["merger_tsc"][:]
        mr = f["mass_ratio"][:]
        hid = f["halo_id"][:]

    # flatten to per-projection rows
    N = len(hid)
    tsc_p = np.repeat(merger, 3)
    mr_p = np.repeat(mr, 3)
    n_p = n_rel.reshape(-1).astype(np.int32)
    d_pri_p = d_pri.reshape(-1)
    d_max_p = d_max.reshape(-1)

    # use d_primary (brightest peak per projection) as the d statistic - it's
    # closer to "the relic an observer would identify" than the outermost peak
    d = d_pri_p

    valid = np.isfinite(d) & np.isfinite(tsc_p)
    active = valid & (tsc_p >= args.tsc_lo) & (tsc_p <= args.tsc_hi)
    sig = active & (mr_p >= args.mr_thresh)
    single = sig & (n_p == 1)
    double = sig & (n_p == 2)

    print(f"catalog attrs: {attrs}")
    print(f"per-projection rows: {len(tsc_p)}")
    print(f"  valid                : {valid.sum()}")
    print(f"  active (TSC {args.tsc_lo}-{args.tsc_hi} Gyr) : {active.sum()}")
    print(f"  + significant (mr>={args.mr_thresh}): {sig.sum()}")
    print(f"  + single relic (n=1) : {single.sum()}")
    print(f"  + double relic (n=2) : {double.sum()}")

    print(f"\nCorrelation merger-TSC vs d_primary (per projection):")
    print(f"  all valid                       : {corr(tsc_p, d, valid):+.3f}")
    print(f"  active                          : {corr(tsc_p, d, active):+.3f}")
    print(f"  active + significant            : {corr(tsc_p, d, sig):+.3f}")
    print(f"  active + significant + single   : {corr(tsc_p, d, single):+.3f}")
    print(f"  active + significant + double   : {corr(tsc_p, d, double):+.3f}")
    print(f"\nAlso d_max:")
    print(f"  active + significant            : {corr(tsc_p, d_max_p, sig):+.3f}")
    print(f"  active + significant + single   : {corr(tsc_p, d_max_p, single):+.3f}")

    v_sh_kms = 1500.0
    ref = v_sh_kms * KMS_TO_KPC_PER_GYR

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    panel(ax[0, 0], tsc_p, d, valid, mr_p,
          "all per-projection (no restriction)", ref)
    panel(ax[0, 1], tsc_p, d, active, mr_p,
          f"active (TSC ∈ [{args.tsc_lo}, {args.tsc_hi}] Gyr)", ref)
    panel(ax[1, 0], tsc_p, d, sig, mr_p,
          f"active + significant (mr ≥ {args.mr_thresh})", ref)
    sc = panel(ax[1, 1], tsc_p, d, single, mr_p,
               "active + significant + single relic (n=1)", ref)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax[1, 1])
        cb.set_label("mass_ratio")

    fig.suptitle("Refined validation: per-projection, bow-shock-active TSC, "
                 "relaxed major threshold\n"
                 f"detector: mach≥{attrs.get('mach_thresh','?')}, "
                 f"peak≥{attrs.get('peak_thresh_frac', 0.25)}, "
                 f"sep≥{attrs.get('min_sep_kpc', 600)}kpc",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=str, default="relic_catalog_v2.h5")
    p.add_argument("--tsc-lo", type=float, default=0.3)
    p.add_argument("--tsc-hi", type=float, default=3.0)
    p.add_argument("--mr-thresh", type=float, default=0.1)
    p.add_argument("--out", type=str, default="relic_v3_validation.png")
    main(p.parse_args())
