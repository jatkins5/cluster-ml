"""How many real cutouts are usable, and how many of those have a WL mass?

Checks C1 (the name-key fix) and C2 (the 40 arcmin re-downloads) together,
and reports the intersection that B2's mass-conditioned model can actually
train against.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward_model_lotss import load_obs, target_key

obs = load_obs(os.path.expanduser("~/data/cluster-ml/lotss_images/*.fits"),
               "LoVoCCS_target_list - lovoccs.csv", 1000.0,
               mask_rng=np.random.default_rng(1000), max_beams=10.0)
print(f"\nusable cutouts: {len(obs)}")

m = pd.read_csv("lovoccs_wl_masses.csv")
mass = {k: (lm, fe) for k, lm, fe in
        zip(m["key"], m["log_m500c"], m["frac_err"])}

print(f"\n{'target':<20}{'z':>8}{'rms mJy/b':>11}{'masked':>8}"
      f"{'log M500c':>11}{'err':>8}")
have = 0
for o in sorted(obs, key=lambda x: x["name"]):
    lm, fe = mass.get(o["key"], (np.nan, np.nan))
    have += np.isfinite(lm)
    print(f"{o['name']:<20}{o['z']:>8.4f}{o['rms'] * 1e3:>11.3f}"
          f"{o['n_masked']:>8}"
          f"{lm:>11.2f}" if np.isfinite(lm) else
          f"{o['name']:<20}{o['z']:>8.4f}{o['rms'] * 1e3:>11.3f}"
          f"{o['n_masked']:>8}{'--':>11}", end="")
    print(f"{fe:>8.0%}" if np.isfinite(fe) else f"{'':>8}")

print(f"\nusable AND with a weak-lensing mass: {have}/{len(obs)}")
missing = [o["name"] for o in obs if o["key"] not in mass]
print(f"usable but no mass ({len(missing)}): {', '.join(sorted(missing))}")
