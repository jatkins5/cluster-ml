"""Collect LoVoCCS weak-lensing masses for the LOFAR targets (checklist B2).

Source: /oscar/data/idellant/Clusters/gen3_processing/<cluster>/mass_fit_output/
mass_peak_statistics.csv, written by lovoccs_pipe's mass_fit.py. Read-only --
nothing here writes into the processing tree.

What the pipeline's numbers mean, established by reading the fitting code
rather than assuming:

  * Single-peak NFW fit (`num_peaks = 1`), so median_mass is the main
    cluster, not a deblended component.
  * Despite the docstring saying "200 x mean-overdensity", shear_profiles.py
    line 66 uses `rho_crit = cosmo.critical_density(zL)`, so it is **M200c**.
  * Msun, no h factor, on FlatLambdaCDM(H0=70, Om0=0.3).
  * Concentration is fixed by Child+18 as c = 75.4 (1+z)^-0.422 M200^-0.089,
    so the M200c -> M500c conversion can use the *same* concentration the fit
    assumed rather than an independent c(M) relation. That consistency is the
    reason to convert here rather than downstream.
  * mass_upper_ci / mass_lower_ci bracket the median from 2000 bootstrap
    realizations; we carry them through as the per-cluster uncertainty, which
    B2 needs in order to inject matched noise on the mass input at training
    time.

The sim side (`Radio_Data/TNG-Cluster_Catalog.hdf5: mhalo_500c`) is log10
M500c in Msun, so M500c is the quantity to match on.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import brentq

GEN3 = "/oscar/data/idellant/Clusters/gen3_processing"
TARGETS = "LoVoCCS_target_list - lovoccs.csv"
LOTSS_GLOB = os.path.expanduser("~/data/cluster-ml/lotss_images/*.fits")
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)     # the fit's cosmology, not TNG's


def key(name):
    """Collapse the three spellings in play: target list 'MKW 3s', gen3 dir
    'MKW3s', FITS file 'lotss_MKW_3s.fits'."""
    return re.sub(r"[\s_]+", "", str(name)).upper()


def concentration(m200, z):
    """Child+18, exactly as shear_profiles.py applies it."""
    return 75.4 * (1.0 + z) ** (-0.422) * m200 ** (-0.089)


def m200c_to_m500c(m200, z):
    """Convert under NFW using the fit's own concentration."""
    c200 = concentration(m200, z)
    mu = lambda x: np.log(1.0 + x) - x / (1.0 + x)
    rho_c = COSMO.critical_density(z).to("Msun/Mpc3").value
    r200 = (3.0 * m200 / (4.0 * np.pi * 200.0 * rho_c)) ** (1.0 / 3.0)
    rs = r200 / c200
    # M(<r) = M200 mu(r/rs)/mu(c200); want it to equal 500 rho_c (4/3) pi r^3
    def f(x):
        m_enc = m200 * mu(x) / mu(c200)
        r = x * rs
        return m_enc - 500.0 * rho_c * (4.0 / 3.0) * np.pi * r ** 3
    x500 = brentq(f, 1e-3, c200)
    return m200 * mu(x500) / mu(c200), c200, x500 * rs


def main():
    # ---- the peak catalogue, for provenance of the fitted centres ----
    xl = glob.glob(os.path.join(GEN3, "Peak catalogue", "*.xlsx"))
    if xl:
        try:
            pk = pd.read_excel(xl[0])
            print(f"peak catalogue: {os.path.basename(xl[0])}")
            print(f"  {len(pk)} rows, columns: {list(pk.columns)}")
            print(pk.head(5).to_string(max_colwidth=22))
        except Exception as e:
            print(f"could not read peak catalogue ({e}); continuing")
    print()

    # ---- redshifts ----
    df = pd.read_csv(TARGETS)
    df["k"] = df["name"].map(key)
    zmap = dict(zip(df["k"], pd.to_numeric(df["redshift"], errors="coerce")))
    lxmap = dict(zip(df["k"], pd.to_numeric(df["lx"], errors="coerce")))

    # ---- which clusters we actually need ----
    lotss = sorted({key(re.sub(r"^lotss_|\.fits$", "", os.path.basename(p)))
                    for p in glob.glob(LOTSS_GLOB)})
    print(f"{len(lotss)} LoTSS cutouts on disk\n")

    rows = []
    for path in sorted(glob.glob(os.path.join(
            GEN3, "*", "mass_fit_output", "mass_peak_statistics.csv"))):
        cluster = path.split(os.sep)[-3]
        if cluster.endswith("_OLD"):
            continue                      # superseded reprocessing
        s = pd.read_csv(path).iloc[0]
        k = key(cluster)
        z = zmap.get(k, np.nan)
        if not np.isfinite(z):
            continue                      # not a LoVoCCS target we can place
        m200 = float(s["median_mass"])
        m500, c200, r500_mpc = m200c_to_m500c(m200, z)
        lo, hi = float(s["mass_lower_ci"]), float(s["mass_upper_ci"])
        rows.append(dict(
            cluster=cluster, key=k, z=z,
            m200c=m200, m200c_lo=lo, m200c_hi=hi,
            m500c=m500,
            m500c_lo=m200c_to_m500c(lo, z)[0],
            m500c_hi=m200c_to_m500c(hi, z)[0],
            log_m500c=np.log10(m500),
            frac_err=0.5 * (hi - lo) / m200,
            c200=c200, r500c_mpc=r500_mpc,
            chi2=float(s["reduced_chi_square"]),
            has_lotss=k in lotss, lx=lxmap.get(k, np.nan)))

    t = pd.DataFrame(rows).sort_values("m500c", ascending=False)
    print(f"{len(t)} LoVoCCS clusters with a weak-lensing mass fit")
    print(f"  M500c range {t.m500c.min():.2e} - {t.m500c.max():.2e} Msun")
    print(f"  median fractional error on M200c: {t.frac_err.median():.1%}")
    print(f"  concentration range {t.c200.min():.2f} - {t.c200.max():.2f}")

    have = t[t.has_lotss]
    print(f"\n======== the LOFAR sample: {len(have)}/{len(lotss)} have masses")
    cols = ["cluster", "z", "m200c", "m500c", "log_m500c", "frac_err", "chi2"]
    print(have[cols].to_string(index=False,
                               float_format=lambda v: f"{v:,.4g}"))
    missing = sorted(set(lotss) - set(t[t.has_lotss].key))
    print(f"\nLoTSS cutouts with no mass fit ({len(missing)}): "
          f"{', '.join(missing)}")

    # Sanity check against the sim distribution we would condition on.
    import h5py
    with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
        sim = f["mhalo_500c"][:]
    print(f"\n======== overlap with TNG-Cluster (log10 M500c)")
    print(f"  sim   : {sim.min():.2f} - {sim.max():.2f}  median {np.median(sim):.2f}")
    print(f"  LoVoCCS WL: {t.log_m500c.min():.2f} - {t.log_m500c.max():.2f}  "
          f"median {t.log_m500c.median():.2f}")
    print(f"  LOFAR subset: {have.log_m500c.min():.2f} - "
          f"{have.log_m500c.max():.2f}  median {have.log_m500c.median():.2f}")
    inside = ((have.log_m500c >= sim.min()) & (have.log_m500c <= sim.max())).mean()
    print(f"  LOFAR targets inside the sim's mass range: {inside:.0%}")

    t.to_csv("lovoccs_wl_masses.csv", index=False)
    print(f"\nwrote lovoccs_wl_masses.csv ({len(t)} clusters)")


if __name__ == "__main__":
    main()
