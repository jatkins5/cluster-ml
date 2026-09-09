"""How much real LoTSS sky do we already have for injection backgrounds?

Bottrell et al. (2019) find that inserting simulated sources into real survey
fields matters far more than any amount of synthetic noise/PSF modelling. The
cheapest source of real backgrounds is the cutouts already on disk: we only use
the central 1 Mpc of each, so offset regions of the same file are unused real
sky at the correct pixel scale.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from astropy.io import fits

files = sorted(glob.glob(os.path.expanduser(
    "~/data/cluster-ml/lotss_images/*.fits")))
df = pd.read_csv("LoVoCCS_target_list - lovoccs.csv")
df["key"] = df["name"].astype(str).str.replace(" ", "")
zmap = dict(zip(df["key"], pd.to_numeric(df["redshift"], errors="coerce")))

from astropy.cosmology import FlatLambdaCDM
COSMO = FlatLambdaCDM(H0=67.74, Om0=0.3089)
ARCSEC = np.pi / 180.0 / 3600.0
OBS_PIX = 1.5

print(f"{len(files)} FITS files\n")
print(f"{'name':<18}{'shape':>14}{'z':>8}{'1Mpc [px]':>11}{'tiles':>7}")
tot = 0
for p in files:
    key = re.sub(r"^lotss_|\.fits$", "", os.path.basename(p))
    with fits.open(p) as h:
        shp = np.squeeze(h[0].data).shape
    z = zmap.get(key, np.nan)
    if not np.isfinite(z):
        print(f"{key:<18}{str(shp):>14}{'--':>8}")
        continue
    kpas = COSMO.kpc_proper_per_arcmin(z).value / 60.0
    box_px = 1000.0 / (OBS_PIX * kpas)
    # Non-overlapping 1 Mpc tiles that fit in the image
    n = int(min(shp) // box_px) ** 2
    tot += max(n - 1, 0)          # minus the central one, already in use
    print(f"{key:<18}{str(shp):>14}{z:>8.4f}{box_px:>11.0f}{n:>7}")
print(f"\nnon-central 1 Mpc tiles available as backgrounds: {tot}")
