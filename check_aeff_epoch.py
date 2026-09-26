"""On-axis effective area at 1.2 keV per observation, relative to Cycle 22.

The exposure map's peak is A_eff(1.2 keV, epoch) x livetime, so
peak / livetime / A_cy22 is the factor by which process_chandra.py's
'Cycle-22-equivalent exposure' exceeds the real livetime -- and so the
factor by which it divides down the unfocused particle background."""
import glob
import os
import re

import numpy as np
import pandas as pd
from astropy.io import fits

from process_chandra import aeff_cy22

A = aeff_cy22()
key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()
obs = pd.read_csv("chandra/observations.csv")
obs = obs[obs["detector"].astype(str).str.strip() == "ACIS-I"]
rows = []
for _, o in obs.iterrows():
    d = os.path.join("chandra", key(o["target"]), str(int(o["obsid"])))
    em = glob.glob(os.path.join(d, "fi*.expmap"))
    ev = glob.glob(os.path.join(d, "*evt2.fits*"))
    if not em or not ev:
        continue
    with fits.open(em[0]) as h:
        peak = float(np.nanmax(h[0].data))
    with fits.open(ev[0]) as h:
        hd = h["EVENTS"].header
        live = float(hd.get("LIVETIME", hd.get("EXPOSURE")))
        date = hd.get("DATE-OBS", "")[:10]
    rows.append(dict(target=o["target"], obsid=int(o["obsid"]), date=date,
                     aeff_epoch=peak / live, ratio_to_cy22=peak / live / A))
t = pd.DataFrame(rows).sort_values("date")
print(f"A_eff(1.2 keV, cy22) = {A:.1f} cm^2")
print(t.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
