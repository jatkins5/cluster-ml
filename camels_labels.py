"""Derive TSC labels for CAMELS zooms from FoF halo-mass history.

CAMELS has no merger trees, so a major merger is flagged as a sudden
fractional jump in M200 between consecutive snapshots (Chuiyang's
suggestion; ~10% for rough parity with the TNG-Cluster merger catalog).
TSC = time between the last flagged jump and snapshot 90 (z~0).

Care taken:
- only snapshot pairs that are truly consecutive (SnapNum diff == 1)
  and that both pass the contamination cut are eligible, since gaps and
  fallback (contaminated) entries produce spurious jumps;
- growth is expressed per Gyr as well as raw fractional, because
  snapshot spacing in cosmic time is far from uniform;
- each zoom has its own cosmology (CAMELS varies Omega_m, sigma_8), so
  ages use that zoom's Omega0/HubbleParam read from its group catalog.

Import `derive_tsc` from the dataset builder; run as a script for the
threshold-sweep diagnostic.
"""
import argparse
import glob
import os

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

BASE = os.path.expanduser("~/data/Chuiyang/Camels")
HIST = f"{BASE}/Halomass_history/GZ28_FOF_mass_history.hdf5"
GROUPCAT = f"{BASE}/Groupcat/GZ28"
TARGET_SNAP = 90


def zoom_cosmology(zoom, default_om0=0.3):
    """Omega0/h for one zoom, from its snap-090 group catalog header."""
    pat = f"{GROUPCAT}/{zoom}/groups_090/fof_subhalo_tab_090.0.hdf5"
    try:
        with h5py.File(pat, "r") as f:
            a = f["Header"].attrs
            return float(a["Omega0"]), float(a["HubbleParam"])
    except (OSError, KeyError):
        return default_om0, None


def derive_tsc(m200, snapnum, redshift, passes, om0, h,
               threshold=0.10, use_rate=False, max_gyr=None):
    """TSC in Gyr at TARGET_SNAP. Returns (tsc, n_jumps, ok)."""
    order = np.argsort(snapnum)
    m200, snapnum = m200[order], snapnum[order]
    redshift, passes = redshift[order], passes[order]

    cosmo = FlatLambdaCDM(H0=100.0 * (h if h else 0.7), Om0=om0)
    age = cosmo.age(np.clip(redshift, 0.0, None)).to("Gyr").value

    tgt = np.where(snapnum == TARGET_SNAP)[0]
    if len(tgt) == 0 or not passes[tgt[0]]:
        return np.nan, 0, False
    tgt = tgt[0]

    jump_ages = []
    for i in range(1, tgt + 1):
        if snapnum[i] - snapnum[i - 1] != 1:
            continue
        if not (passes[i] and passes[i - 1]):
            continue
        if m200[i - 1] <= 0:
            continue
        frac = (m200[i] - m200[i - 1]) / m200[i - 1]
        if use_rate:
            dt = age[i] - age[i - 1]
            if dt <= 0:
                continue
            frac = frac / dt        # fractional growth per Gyr
        if frac > threshold:
            jump_ages.append(age[i])

    if not jump_ages:
        # no detected merger: censored, treat as at least the full
        # tracked baseline
        tsc = age[tgt] - age[0]
        return (tsc if max_gyr is None else min(tsc, max_gyr)), 0, False
    tsc = age[tgt] - max(jump_ages)
    if max_gyr is not None:
        tsc = min(tsc, max_gyr)
    return tsc, len(jump_ages), True


def load_all(threshold=0.10, use_rate=False, verbose=False):
    out = {}
    with h5py.File(HIST, "r") as f:
        zooms = sorted(f.keys())
        for z in zooms:
            g = f[z]
            om0, h = zoom_cosmology(z)
            tsc, njump, ok = derive_tsc(
                g["M200_Msun"][:], g["SnapNum"][:], g["Redshift"][:],
                g["passes_contamination_cut"][:], om0, h,
                threshold=threshold, use_rate=use_rate)
            snap = g["SnapNum"][:]
            i90 = np.where(snap == TARGET_SNAP)[0]
            out[z] = dict(
                tsc=tsc, n_jumps=njump, detected=ok, om0=om0, h=h,
                m200=float(g["M200_Msun"][i90[0]]) if len(i90) else np.nan,
                clean=bool(g["passes_contamination_cut"][i90[0]])
                if len(i90) else False,
                n_snap=len(snap),
                n_clean=int(g["passes_contamination_cut"][:].sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30])
    args = ap.parse_args()

    base = load_all(threshold=0.10)
    zooms = sorted(base)
    print(f"zooms in history file: {len(zooms)}")
    radios = glob.glob(f"{BASE}/Radio/radio_GZ28_*_snap090.hdf5")
    have_radio = {os.path.basename(p).split("radio_")[1].split("_snap")[0]
                  for p in radios}
    print(f"zooms with radio maps: {len(have_radio)}  "
          f"overlap: {len(set(zooms) & have_radio)}")

    m200 = np.array([base[z]["m200"] for z in zooms])
    clean = np.array([base[z]["clean"] for z in zooms])
    nsnap = np.array([base[z]["n_snap"] for z in zooms])
    om0 = np.array([base[z]["om0"] for z in zooms])
    hh = np.array([base[z]["h"] if base[z]["h"] else np.nan for z in zooms])
    print(f"\nsnap-90 clean (passes contamination): {clean.sum()}/{len(zooms)}")
    print(f"snapshots tracked per zoom: min={nsnap.min()} "
          f"med={int(np.median(nsnap))} max={nsnap.max()}")
    print(f"M200 [Msun] log10: min={np.log10(np.nanmin(m200)):.2f} "
          f"med={np.log10(np.nanmedian(m200)):.2f} "
          f"max={np.log10(np.nanmax(m200)):.2f}")
    print("  (TNG-Cluster M500c log10 range: 14.0-15.3)")
    print(f"Omega0: {np.nanmin(om0):.3f}-{np.nanmax(om0):.3f}   "
          f"h: {np.nanmin(hh):.3f}-{np.nanmax(hh):.3f}")

    print("\nthreshold sweep (fractional M200 jump between consecutive snaps):")
    print(f"{'thr':>6} {'detected':>9} {'med TSC':>9} {'IQR':>15} "
          f"{'med n_jumps':>12}")
    for thr in args.thresholds:
        d = load_all(threshold=thr)
        tsc = np.array([d[z]["tsc"] for z in zooms])
        det = np.array([d[z]["detected"] for z in zooms])
        nj = np.array([d[z]["n_jumps"] for z in zooms])
        q1, q3 = np.nanpercentile(tsc[det], [25, 75]) if det.any() else (0, 0)
        print(f"{thr:>6.2f} {det.sum():>4}/{len(zooms):<4} "
              f"{np.nanmedian(tsc[det]) if det.any() else np.nan:>9.2f} "
              f"{f'[{q1:.2f}, {q3:.2f}]':>15} {int(np.median(nj)):>12}")

    print("\nsame, growth rate per Gyr (snapshot-spacing corrected):")
    for thr in args.thresholds:
        d = load_all(threshold=thr, use_rate=True)
        tsc = np.array([d[z]["tsc"] for z in zooms])
        det = np.array([d[z]["detected"] for z in zooms])
        nj = np.array([d[z]["n_jumps"] for z in zooms])
        q1, q3 = np.nanpercentile(tsc[det], [25, 75]) if det.any() else (0, 0)
        print(f"{thr:>6.2f} {det.sum():>4}/{len(zooms):<4} "
              f"{np.nanmedian(tsc[det]) if det.any() else np.nan:>9.2f} "
              f"{f'[{q1:.2f}, {q3:.2f}]':>15} {int(np.median(nj)):>12}")

    print("\nTNG-Cluster reference: TSC 0-7.73 Gyr, median ~1.5")


if __name__ == "__main__":
    main()
