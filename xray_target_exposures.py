"""Per-target Chandra ACIS-I exposure quantities the X-ray mocks must match
(review finding F5): livetime (particle background scales with it), the
Cycle-22-equivalent exposure the real images are normalised by (focused
photons scale with it), and the number of 9.5-12 keV events the particle
estimate is built from (its Poisson error is the noise of the subtraction).

Livetime and hard counts are summed over a target's ACIS-I observations from
the event headers; T_eq is the median per-block value the processed images
used (process_chandra.py stored it as obs/exposure_ks).
"""
import os
import re

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits

from process_chandra import expmap_for

key = lambda x: re.sub(r"[\s_]+", "", str(x)).upper()
obs = pd.read_csv("chandra/observations.csv")
obs = obs[obs["detector"].astype(str).str.strip() == "ACIS-I"]
with h5py.File("xray_obs_acisi_pbc.h5") as f:
    teq = dict(zip([s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/key"][:]],
                   f["obs/exposure_ks"][:]))
    zz = dict(zip([s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/key"][:]],
                  f["obs/z"][:]))
rows = []
for tgt, grp in obs.groupby(obs["target"]):
    k = key(tgt)
    if k not in teq:
        continue
    live, nh, raw = 0.0, 0, 0.0
    for _, o in grp.iterrows():
        d = os.path.join("chandra", k, str(int(o["obsid"])))
        try:
            evt, _ = expmap_for(d, int(o["obsid"]))
        except Exception as e:
            print(f"  {tgt} {o['obsid']}: {str(e)[:120]}")
            continue
        with fits.open(evt) as h:
            hd, dd = h["EVENTS"].header, h["EVENTS"].data
            live += float(hd.get("LIVETIME", hd.get("EXPOSURE"))) / 1e3
            e = dd["energy"] / 1000.0
            nh += int(((e >= 9.5) & (e <= 12.0)).sum())
        raw += float(o["exposure"]) / 1e3
    rows.append(dict(key=k, target=tgt, z=zz[k], t_eq_ks=float(teq[k]),
                     t_live_ks=live, t_raw_ks=raw, n_hard=nh,
                     particle_frac_err=1 / np.sqrt(max(nh, 1))))
df = pd.DataFrame(rows).sort_values("z")
df["eq_over_live"] = df.t_eq_ks / df.t_live_ks
df.to_csv("xray_target_exposures.csv", index=False)
print(df.round(3).to_string(index=False))
