"""An independent observational check on the real-cluster predictions.

The project has no ground-truth merger time for the LOFAR targets, so every
statement about real data has so far been "in range and stable", never
"correct". The LoVoCCS peak catalogue offers the one handle we have: weak
lensing measures the *total* mass distribution, so the number of significant
shear peaks and their separation is a dynamical-state indicator derived from
a completely different observable than the radio image the model looks at.

A cluster caught mid-merger should show multiple mass peaks, or one clearly
elongated peak; a relaxed cluster a single round one. If the model's
predicted time-since-collision is doing anything real, clusters with more
shear substructure should be predicted *younger*, and the correlation should
be negative.

This is a weak test and it is worth being honest about why: 17-25 clusters,
peak detection that depends on survey depth, and projection effects on both
sides. A null result would not be evidence the model fails. A clear result
in the right direction would be the only external validation available.

Peak catalogue: gen3_processing/Peak catalogue/*.xlsx, sectioned by cluster
(a "Cluster <name>" row, then that cluster's peaks). Parsed with the stdlib
via read_peak_catalogue.read_sheet -- no openpyxl in the venv.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_peak_catalogue import read_sheet

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)      # the mass fit's cosmology
KEY = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()


def parse_catalogue(path):
    """-> {cluster key: DataFrame of its peaks}."""
    rows = read_sheet(path)
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = rows[0]
    # Section rows carry the literal "Cluster" in the column where the
    # per-peak rows carry their centroid; the name sits one column right.
    try:
        marker = header.index("Cluster")
    except ValueError:
        marker = 4
    out, current, buf = {}, None, []
    colnames = [c for c in rows[0] if c] or None
    for r in rows[1:]:
        if str(r[marker]).strip() == "Cluster":
            if current and buf:
                out[current] = buf
            current = KEY(r[marker + 1])
            buf = []
        elif current is not None and str(r[0]).strip():
            buf.append(r)
    if current and buf:
        out[current] = buf

    # Column positions from the per-peak header line (row index 1).
    hdr = rows[1]
    idx = {name: i for i, name in enumerate(hdr) if name}
    def col(name, default=None):
        for k, v in idx.items():
            if k.strip().rstrip(",") == name:
                return v
        return default
    cols = {n: col(n) for n in
            ["SN_peak", "sn_fwhm", "sn_ellip", "area", "ra", "dec"]}
    frames = {}
    for k, body in out.items():
        recs = []
        for r in body:
            rec = {}
            ok = True
            for name, i in cols.items():
                if i is None or i >= len(r):
                    ok = False
                    break
                try:
                    rec[name] = float(str(r[i]).replace("E-", "e-"))
                except ValueError:
                    ok = False
                    break
            if ok:
                recs.append(rec)
        if recs:
            frames[k] = pd.DataFrame(recs)
    return frames


def peak_stats(df, z, r500_mpc, sn_min):
    """Substructure summary for one cluster."""
    d = df[df["SN_peak"] >= sn_min].sort_values("SN_peak", ascending=False)
    if len(d) == 0:
        return None
    kpc_per_deg = COSMO.angular_diameter_distance(z).to("kpc").value \
        * np.pi / 180.0
    out = {"n_peak": len(d), "sn_max": d["SN_peak"].iloc[0],
           "ellip_main": d["sn_ellip"].iloc[0],
           "fwhm_main": d["sn_fwhm"].iloc[0]}
    if len(d) >= 2:
        ra0, dec0 = d["ra"].iloc[0], d["dec"].iloc[0]
        ra1, dec1 = d["ra"].iloc[1], d["dec"].iloc[1]
        dra = (ra1 - ra0) * np.cos(np.radians(dec0))
        sep_deg = np.hypot(dra, dec1 - dec0)
        out["sep_kpc"] = sep_deg * kpc_per_deg
        out["sep_r500"] = out["sep_kpc"] / (r500_mpc * 1000.0)
        out["sn_ratio"] = d["SN_peak"].iloc[1] / d["SN_peak"].iloc[0]
    else:
        out["sep_kpc"] = out["sep_r500"] = out["sn_ratio"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="cnn_preds/b2_imgmassn_s*_preds.npz")
    ap.add_argument("--masses", default="lovoccs_wl_masses.csv")
    ap.add_argument("--sn-min", type=float, nargs="+", default=[3.0, 4.0, 5.0])
    args = ap.parse_args()

    cat = glob.glob("/oscar/data/idellant/Clusters/gen3_processing/"
                    "Peak catalogue/*.xlsx")[0]
    frames = parse_catalogue(cat)
    print(f"peak catalogue: {len(frames)} clusters, "
          f"{sum(len(v) for v in frames.values())} peaks total")

    m = pd.read_csv(args.masses).set_index("key")
    preds = sorted(glob.glob(args.preds))
    if not preds:
        raise SystemExit(f"no predictions matching {args.preds}")
    ens, names = None, None
    for p in preds:
        d = np.load(p, allow_pickle=True)
        v = d["obs_pred"]
        names = [str(s) for s in d["obs_name"]]
        ens = v if ens is None else ens + v
    ens = ens / len(preds)
    print(f"predictions: {len(preds)} runs matching {args.preds}, "
          f"{len(names)} targets\n")

    for sn_min in args.sn_min:
        rows = []
        for n, pred in zip(names, ens):
            k = KEY(n)
            if k not in frames or k not in m.index:
                continue
            st = peak_stats(frames[k], float(m.loc[k, "z"]),
                            float(m.loc[k, "r500c_mpc"]), sn_min)
            if st is None:
                continue
            st.update(target=n, pred=pred, log_m500=float(m.loc[k, "log_m500c"]))
            rows.append(st)
        t = pd.DataFrame(rows)
        if len(t) < 5:
            print(f"S/N > {sn_min}: only {len(t)} matched clusters, skipping")
            continue

        print(f"======== shear peaks at S/N > {sn_min}  (n={len(t)})")
        print(f"{'target':<20}{'pred TSC':>10}{'n_peak':>8}{'sn_max':>8}"
              f"{'ellip':>7}{'sep/r500':>10}{'log M500':>10}")
        for _, r in t.sort_values("pred").iterrows():
            sep = f"{r.sep_r500:.2f}" if np.isfinite(r.sep_r500) else "--"
            print(f"{r.target:<20}{r.pred:>10.2f}{int(r.n_peak):>8}"
                  f"{r.sn_max:>8.1f}{r.ellip_main:>7.2f}{sep:>10}"
                  f"{r.log_m500:>10.2f}")

        print(f"\n  {'statistic':<24}{'Spearman vs pred':>18}{'p':>9}"
              f"{'expected':>12}")
        tests = [("number of peaks", "n_peak", "negative"),
                 ("peak separation / r500", "sep_r500", "negative"),
                 ("main-peak ellipticity", "ellip_main", "negative"),
                 ("secondary / main S/N", "sn_ratio", "negative"),
                 ("peak S/N (control)", "sn_max", "none"),
                 ("halo mass (control)", "log_m500", "none")]
        for label, col, exp in tests:
            v = t[col].to_numpy(dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 5 or np.ptp(v[ok]) == 0:
                print(f"  {label:<24}{'--':>18}{'':>9}{exp:>12}")
                continue
            r, p = stats.spearmanr(v[ok], t["pred"].to_numpy()[ok])
            print(f"  {label:<24}{r:>18.3f}{p:>9.3f}{exp:>12}"
                  f"   (n={ok.sum()})")
        print()


if __name__ == "__main__":
    main()
