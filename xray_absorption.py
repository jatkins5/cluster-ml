"""Per-target Galactic absorption, and what it does to 0.1-2 keV count rates.

The mocks were absorbed with one column for every cluster (pyxsim wabs,
nH = 0.01e22 cm^-2), but each real target sits behind its own Galactic HI
column, and the 0.1-2 keV band is exactly where absorption bites. X-rays are
absorbed by the gas, not by dust, so this -- not a dust model -- is the
relevant correction.

For each LOFAR target:
  1. nH from HEASARC's nH tool (HI4PI, weighted average within 0.1 deg);
  2. the 0.1-2 keV ACIS-I count rate of an APEC plasma (kT = 5 keV,
     Z = 0.3 solar, at the target's redshift) through the Cycle-22 ARF,
     absorbed by the target's nH and by the mock's nH;
  3. the ratio, which the rebuild applies as an extra thinning factor on
     the source photons of every mock placed at that target's redshift.
The RMF is ignored: redistribution barely moves a band-integrated ratio.
kT sensitivity is reported so the single-temperature choice can be judged.
"""
import glob
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from astropy.io import fits

NH_MOCK = 0.01          # 1e22 cm^-2, as in X-ray_chandra_Snap0_*.py
BAND = (0.1, 2.0)
key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()


def nh_heasarc(ra, dec):
    q = urllib.parse.urlencode({"Entry": f"{ra},{dec}", "NR": "GRB/SIMBAD+Sesame/NED",
                                "CoordSys": "Equatorial", "equinox": "2000",
                                "radius": "0.1", "usemap": "0"})
    url = f"https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3nh/w3nh.pl?{q}"
    with urllib.request.urlopen(url, timeout=60) as r:
        txt = r.read().decode("utf-8", "replace")
    m = re.findall(r"Weighted average nH \(cm\*\*-2\)\s*([0-9.Ee+-]+)", txt)
    if not m:
        m = re.findall(r"nH \(cm\*\*-2\)\s*([0-9.Ee+-]+)", txt)
    return float(m[-1]) / 1e22 if m else np.nan


def band_rate(nh, kT, z, arf_e, arf_a):
    from soxs import ApecGenerator
    agen = ApecGenerator(0.05, 12.0, 4000)
    spec = agen.get_spectrum(kT, 0.3, z, 1.0)
    spec.apply_foreground_absorption(nh, model="wabs")
    e = spec.emid.value
    f = spec.flux.value * spec.de.value           # photons/s/cm^2 per bin
    aeff = np.interp(e, arf_e, arf_a, left=0.0, right=0.0)
    sel = (e >= BAND[0]) & (e <= BAND[1])
    return float((f * aeff)[sel].sum())


def main():
    arf = os.path.expanduser("~/.cache/soxs/acisi_aimpt_cy22.arf")
    with fits.open(arf) as h:
        d = h["SPECRESP"].data
        arf_e = 0.5 * (d["ENERG_LO"] + d["ENERG_HI"])
        arf_a = d["SPECRESP"]

    tl = pd.read_csv("LoVoCCS_target_list - lovoccs.csv")
    tl["k"] = tl["name"].map(key)
    lofar = {key(re.sub(r"^lotss_|\.fits$", "", os.path.basename(p)))
             for p in glob.glob(os.path.expanduser(
                 "~/data/cluster-ml/lotss_images/lotss_*.fits"))}
    t = tl[tl["k"].isin(lofar)].copy()

    rows = []
    for _, r in t.iterrows():
        ra, dec = float(r["ra(deg)"]), float(r["dec(deg)"])
        z = float(r["redshift"])
        try:
            nh = nh_heasarc(ra, dec)
        except Exception as e:
            print(f"  {r['name']}: nH query failed ({e})")
            nh = np.nan
        row = dict(name=r["name"], key=r["k"], z=z, nh_1e22=nh)
        if np.isfinite(nh) and np.isfinite(z):
            for kT in (3.0, 5.0, 8.0):
                row[f"ratio_kT{kT:.0f}"] = (band_rate(nh, kT, z, arf_e, arf_a)
                                            / band_rate(NH_MOCK, kT, z, arf_e, arf_a))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("nh_1e22")
    pd.set_option("display.width", 160)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    good = out["ratio_kT5"].notna()
    print(f"\nnH range {out.nh_1e22.min():.3g}-{out.nh_1e22.max():.3g} e22 "
          f"(mock {NH_MOCK}); count-rate factor at kT=5: median "
          f"{out.loc[good, 'ratio_kT5'].median():.3f}, range "
          f"{out.loc[good, 'ratio_kT5'].min():.3f}-{out.loc[good, 'ratio_kT5'].max():.3f}")
    spread = (out[["ratio_kT3", "ratio_kT8"]].max(axis=1)
              - out[["ratio_kT3", "ratio_kT8"]].min(axis=1)).max()
    print(f"largest kT 3->8 keV difference in the factor: {spread:.3f}")
    out.to_csv("lofar_nh.csv", index=False)
    print("wrote lofar_nh.csv")


if __name__ == "__main__":
    main()
