"""Which archival Chandra and XMM observations exist for the LOFAR targets?

The target list's Chandra/XMM columns are almost entirely empty, so query
HEASARC's master tables directly. The answer sets the depth a realistic
X-ray forward model has to imitate: the existing mocks are 2 Ms, and no real
cluster observation is anywhere near that.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.heasarc import Heasarc

key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()
df = pd.read_csv("LoVoCCS_target_list - lovoccs.csv")
df["k"] = df["name"].map(key)
lofar = {key(re.sub(r"^lotss_|\.fits$", "", os.path.basename(p)))
         for p in glob.glob(os.path.expanduser(
             "~/data/cluster-ml/lotss_images/lotss_*.fits"))}
t = df[df["k"].isin(lofar)].copy()
t["ra"] = pd.to_numeric(t["ra(deg)"], errors="coerce")
t["dec"] = pd.to_numeric(t["dec(deg)"], errors="coerce")

h = Heasarc()
rows = []
for _, r in t.iterrows():
    c = SkyCoord(r.ra, r.dec, unit="deg")
    out = {"name": r["name"], "z": r["redshift"]}
    for table, tcol, label in [("chanmaster", "exposure", "chandra"),
                               ("xmmmaster", "duration", "xmm")]:
        try:
            res = h.query_region(c, catalog=table, radius=6 * u.arcmin)
            tab = res.to_pandas() if res is not None else pd.DataFrame()
        except Exception as e:
            tab = pd.DataFrame()
            out[f"{label}_err"] = str(e)[:60]
        if len(tab):
            ecol = next((cc for cc in tab.columns if cc.lower() == tcol), None)
            ks = (pd.to_numeric(tab[ecol], errors="coerce").fillna(0) / 1e3
                  if ecol else pd.Series([np.nan] * len(tab)))
            out[f"{label}_n"] = len(tab)
            out[f"{label}_ks_total"] = ks.sum()
            out[f"{label}_ks_max"] = ks.max()
        else:
            out[f"{label}_n"] = 0
            out[f"{label}_ks_total"] = 0.0
            out[f"{label}_ks_max"] = 0.0
    rows.append(out)

res = pd.DataFrame(rows).sort_values("chandra_ks_total", ascending=False)
pd.set_option("display.width", 200)
print(res.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
for lab in ("chandra", "xmm"):
    have = res[f"{lab}_n"] > 0
    print(f"\n{lab}: {have.sum()}/{len(res)} targets observed; total exposure "
          f"median {res.loc[have, f'{lab}_ks_total'].median():.0f} ks, "
          f"range {res.loc[have, f'{lab}_ks_total'].min():.0f}-"
          f"{res.loc[have, f'{lab}_ks_total'].max():.0f} ks")
res.to_csv("lofar_xray_archive.csv", index=False)
print("\nwrote lofar_xray_archive.csv")
