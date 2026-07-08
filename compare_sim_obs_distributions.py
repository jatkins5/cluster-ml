"""Distributional comparison: TNG-Cluster sims vs LoVoCCS observations.

Phase 1 of the sim-vs-obs validation suggested by the PI:
ECDF + QQ comparison of X-ray luminosities, LoVoCCS selection-cut pass
fraction on the sim sample, KS/AD tests, and importance-reweighting
factors so the training set can be tilted toward the observed L_X
distribution.

Band caveat: LoVoCCS selection L_X is 0.1-2.4 keV (ROSAT); the TNG
catalog stores 0.5-2.0 keV. We apply an approximate band-conversion
factor for a ~5 keV APEC-like ICM spectrum and report results both raw
and converted.
"""
import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# L(0.1-2.4 keV) / L(0.5-2.0 keV) for kT ~ 4-8 keV, Z ~ 0.3 Zsun thermal
# spectrum; ~1.5-1.8 across that kT range. Single factor, so shapes of the
# ECDFs are unaffected -- only the horizontal offset.
BAND_FACTOR = 1.65

LOVOCCS_CUT = 1e44  # erg/s, 0.1-2.4 keV


def load_obs(csv_path):
    df = pd.read_csv(csv_path)
    df = df[["name", "redshift", "lx"]].copy()
    df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
    df["lx"] = pd.to_numeric(df["lx"], errors="coerce")
    df = df.dropna(subset=["lx"])
    df = df[df["lx"] > 0]
    return df


def load_sim(cat_path):
    with h5py.File(cat_path, "r") as f:
        halo_id = f["haloID"][:]
        log_lx = f["xray_0.5-2.0kev"][:]
        log_m500 = f["mhalo_500c"][:]
        log_m200 = f["mhalo_200c"][:]
    return halo_id, 10.0 ** log_lx, 10.0 ** log_m500, 10.0 ** log_m200


def ecdf(x):
    xs = np.sort(x)
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def qq(a, b, n=100):
    q = np.linspace(0.01, 0.99, n)
    return np.quantile(a, q), np.quantile(b, q)


def importance_weights(sim_vals, obs_vals, n_bins=15):
    """Histogram-ratio weights so weighted sim log-L_X matches obs."""
    log_sim, log_obs = np.log10(sim_vals), np.log10(obs_vals)
    lo = min(log_sim.min(), log_obs.min()) - 1e-6
    hi = max(log_sim.max(), log_obs.max()) + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    p_sim, _ = np.histogram(log_sim, bins=edges, density=True)
    p_obs, _ = np.histogram(log_obs, bins=edges, density=True)
    idx = np.clip(np.digitize(log_sim, edges) - 1, 0, n_bins - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_sim > 0, p_obs / p_sim, 0.0)
    w = ratio[idx]
    if w.sum() > 0:
        w *= len(w) / w.sum()
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="Radio_Data/TNG-Cluster_Catalog.hdf5")
    ap.add_argument("--obs-csv", default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--out-prefix", default="sim_obs_dist")
    args = ap.parse_args()

    obs = load_obs(args.obs_csv)
    halo_id, sim_lx_raw, m500, m200 = load_sim(args.catalog)
    sim_lx_conv = sim_lx_raw * BAND_FACTOR

    print(f"obs targets with valid lx: {len(obs)}  "
          f"(z {obs['redshift'].min():.3f}-{obs['redshift'].max():.3f})")
    print(f"sim clusters: {len(sim_lx_raw)}")
    print(f"obs  lx  [0.1-2.4 keV]: {obs['lx'].min():.2e} - {obs['lx'].max():.2e}")
    print(f"sim  lx  [0.5-2.0 keV]: {sim_lx_raw.min():.2e} - {sim_lx_raw.max():.2e}")
    print(f"sim  lx  converted x{BAND_FACTOR}: "
          f"{sim_lx_conv.min():.2e} - {sim_lx_conv.max():.2e}")

    # LoVoCCS selection cut applied to sims
    frac_raw = np.mean(sim_lx_raw > LOVOCCS_CUT)
    frac_conv = np.mean(sim_lx_conv > LOVOCCS_CUT)
    print(f"\nsim fraction passing LX > 1e44: raw band {frac_raw:.1%}, "
          f"converted {frac_conv:.1%} "
          f"({int(frac_conv * len(sim_lx_conv))}/{len(sim_lx_conv)} clusters)")

    # two-sample tests (log space), full sim sample and cut sample
    obs_lx = obs["lx"].values
    for label, sim_sample in [
        ("all sim (converted)", sim_lx_conv),
        ("sim > cut (converted)", sim_lx_conv[sim_lx_conv > LOVOCCS_CUT]),
    ]:
        ks = stats.ks_2samp(np.log10(sim_sample), np.log10(obs_lx))
        ad = stats.anderson_ksamp(
            [np.log10(sim_sample), np.log10(obs_lx)],
            method=stats.PermutationMethod(n_resamples=999, random_state=0))
        med_ratio = np.median(sim_sample) / np.median(obs_lx)
        print(f"\n{label}: n={len(sim_sample)}")
        print(f"  median L_X sim/obs = {med_ratio:.2f}")
        print(f"  KS D={ks.statistic:.3f} p={ks.pvalue:.2e}")
        print(f"  AD stat={ad.statistic:.2f} p={ad.pvalue:.3f}")

    # importance weights: match cut sim sample to obs
    weights = importance_weights(sim_lx_conv, obs_lx)
    np.savez(f"{args.out_prefix}_weights.npz",
             halo_id=halo_id, weight=weights,
             lx_converted=sim_lx_conv, band_factor=BAND_FACTOR)
    print(f"\nweights saved -> {args.out_prefix}_weights.npz  "
          f"(min {weights.min():.2f}, max {weights.max():.2f}, "
          f"zero-weight halos: {(weights == 0).sum()})")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    for vals, lab, style in [
        (sim_lx_raw, "sim 0.5-2.0 keV (raw)", "C0--"),
        (sim_lx_conv, f"sim x{BAND_FACTOR} (~0.1-2.4 keV)", "C0-"),
        (obs_lx, "LoVoCCS 0.1-2.4 keV", "C1-"),
    ]:
        xs, ys = ecdf(np.log10(vals))
        ax.plot(xs, ys, style, label=lab)
    ax.axvline(44, color="gray", ls=":", label="LoVoCCS cut 1e44")
    ax.set_xlabel("log10 L_X [erg/s]")
    ax.set_ylabel("ECDF")
    ax.legend(fontsize=8)
    ax.set_title("X-ray luminosity ECDF")

    ax = axes[0, 1]
    qs, qo = qq(np.log10(sim_lx_conv), np.log10(obs_lx))
    ax.plot(qo, qs, "C0.-", ms=3)
    lims = [min(qo.min(), qs.min()), max(qo.max(), qs.max())]
    ax.plot(lims, lims, "k:", lw=1)
    ax.set_xlabel("obs quantiles: log10 L_X")
    ax.set_ylabel("sim (converted) quantiles: log10 L_X")
    ax.set_title("QQ plot, sim vs LoVoCCS")

    ax = axes[1, 0]
    for vals, lab in [(m500, "M500c"), (m200, "M200c")]:
        xs, ys = ecdf(np.log10(vals))
        ax.plot(xs, ys, label=lab)
    ax.set_xlabel("log10 M [Msun]")
    ax.set_ylabel("ECDF")
    ax.legend()
    ax.set_title("Sim halo mass ECDF (obs WL masses pending)")

    ax = axes[1, 1]
    log_lx = np.log10(sim_lx_conv)
    bins = np.linspace(log_lx.min(), log_lx.max(), 20)
    ax.hist(log_lx, bins=bins, alpha=0.5, density=True, label="sim unweighted")
    ax.hist(log_lx, bins=bins, weights=weights, alpha=0.5, density=True,
            label="sim reweighted")
    xs, _ = ecdf(np.log10(obs_lx))
    ax.hist(np.log10(obs_lx), bins=bins, histtype="step", density=True,
            color="C1", lw=2, label="LoVoCCS")
    ax.set_xlabel("log10 L_X [erg/s] (converted)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("Importance reweighting check")

    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=150)
    print(f"figure saved -> {args.out_prefix}.png")


if __name__ == "__main__":
    main()
