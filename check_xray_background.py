"""Is the real Chandra background lower than the mocks', or is it my
normalisation?

check_xray_domain_gap.py found the clusters predicted beyond the label range
have outer-region brightness below 97-100% of matched mocks. Two candidate
causes, with different fixes:

  a) the soxs background (sky + particle) is simply higher than reality;
  b) process_chandra.py divides all counts by a Cycle-22-equivalent
     exposure (exposure map / A_eff), which is right for sky photons but
     wrong for the particle background: particles are not focused by the
     mirrors, so they scale with time, not effective area. Where the real
     A_eff exceeds the Cycle-22 one (earlier observations, less
     contamination) the particle background is divided down.

Measured directly from the event files: 0.1-2 keV counts per arcmin^2 per ks
of livetime, in a 6-9 arcmin annulus around each target (on chips, point
sources suppressed by taking the median over 1-arcmin cells), against the
same quantity for the soxs sky and particle backgrounds the mocks use.
"""
import glob
import os
import re

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from build_xray_realistic import BLOCK, T_MOCK_KS, GRID

key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()
BLOCK_ARCMIN2 = (BLOCK * 0.492 / 60.0) ** 2


def soxs_rate(path):
    """Median counts / ks / arcmin^2 over blocks the chips cover."""
    with h5py.File(path, "r") as f:
        c = f["counts"][:].astype(float)            # (K, 128, 128) at 2 Ms
    per = c / T_MOCK_KS / BLOCK_ARCMIN2
    yy, xx = np.mgrid[:GRID, :GRID]
    rr = np.hypot(yy - GRID / 2, xx - GRID / 2) * BLOCK * 0.492 / 60.0
    ring = (rr > 2) & (rr < 6)                        # safely on-chip
    return float(np.median(per[:, ring]))


def real_rate(evt, ra0, dec0):
    with fits.open(evt) as h:
        d, hd = h["EVENTS"].data, h["EVENTS"].header
        live = float(hd.get("LIVETIME", hd.get("EXPOSURE"))) / 1e3
        cols = [c.name.lower() for c in h["EVENTS"].columns]
        ix, iy = cols.index("x") + 1, cols.index("y") + 1
        w = WCS(naxis=2)
        w.wcs.ctype = [hd[f"TCTYP{ix}"], hd[f"TCTYP{iy}"]]
        w.wcs.crval = [hd[f"TCRVL{ix}"], hd[f"TCRVL{iy}"]]
        w.wcs.crpix = [hd[f"TCRPX{ix}"], hd[f"TCRPX{iy}"]]
        w.wcs.cdelt = [hd[f"TCDLT{ix}"], hd[f"TCDLT{iy}"]]
        e = d["energy"] / 1000.0
        sel = (e >= 0.1) & (e <= 2.0)
        ra, dec = w.wcs_pix2world(d["x"][sel], d["y"][sel], 1)
        obs_date = hd.get("DATE-OBS", "")
    dx = (ra - ra0) * np.cos(np.radians(dec0)) * 60.0
    dy = (dec - dec0) * 60.0
    r = np.hypot(dx, dy)
    ring = (r > 6) & (r < 9)
    # 1-arcmin cells in the ring; empty cells are off-chip and dropped, and
    # the median suppresses point sources.
    cx, cy = np.floor(dx[ring]).astype(int), np.floor(dy[ring]).astype(int)
    cells = pd.Series(1, index=pd.MultiIndex.from_arrays([cx, cy])).groupby(
        level=[0, 1]).sum()
    cells = cells[cells > 0]
    return float(np.median(cells.values)) / live, live, len(cells), obs_date


def main():
    sky = soxs_rate("xray_skybkg_2Ms_128.h5")
    part = soxs_rate("xray_partbkg_2Ms_128.h5")
    print(f"mock background, 0.1-2 keV, counts/ks/arcmin^2: sky {sky:.4f} + "
          f"particle {part:.4f} = {sky + part:.4f}")

    obs = pd.read_csv("chandra/observations.csv")
    obs = obs[obs["detector"].astype(str).str.strip() == "ACIS-I"]
    tl = pd.read_csv("LoVoCCS_target_list - lovoccs.csv")
    pos = {key(r["name"]): (float(r["ra(deg)"]), float(r["dec(deg)"]),
                            float(r["redshift"])) for _, r in tl.iterrows()}
    rows = []
    for _, o in obs.iterrows():
        k = key(o["target"])
        evt = glob.glob(os.path.join("chandra", k, str(int(o["obsid"])),
                                     "*evt2.fits*"))
        if not evt or k not in pos:
            continue
        ra0, dec0, z = pos[k]
        rate, live, ncell, date = real_rate(evt[0], ra0, dec0)
        rows.append(dict(target=o["target"], obsid=int(o["obsid"]), date=date[:10],
                         z=z, live_ks=live, cells=ncell, rate=rate,
                         real_over_mock=rate / (sky + part)))
    t = pd.DataFrame(rows).sort_values("date")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nmedian real / mock background: {t.real_over_mock.median():.2f} "
          f"(range {t.real_over_mock.min():.2f}-{t.real_over_mock.max():.2f})")
    print("note: a 6-9' ring still holds some cluster emission for the nearest "
          "targets, so those rates are upper limits on the background")


if __name__ == "__main__":
    main()
