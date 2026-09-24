"""What do the existing X-ray mocks actually contain, and what real data
would they have to match?

Chuiyang's pipeline (TNGCluster_Xray_Snap99/*/X-ray_chandra_Snap0_*.py,
read-only) runs pyxsim -> soxs for Chandra ACIS-I at a 2 Ms exposure, every
cluster at z=0.05, with foreground and point-source backgrounds switched off.
Before designing a realistic-depth forward model we need:

  1. the image units, so count rates can be turned back into counts -- the
     raw pixels should be quantised at 1/exposure, which gives the
     conversion empirically even if the header does not;
  2. total source counts per cluster at 2 Ms, and hence at realistic depths;
  3. the sky footprint: ACIS-I is ~16.9 arcmin on a side, the image ~40,
     so part of every image may be unexposed;
  4. which X-ray observations exist for the LOFAR targets.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = "TNGCluster_Xray_Snap99/snap99_z"
files = sorted(glob.glob(os.path.join(ROOT, "halo_*_img.fits")))
rng = np.random.default_rng(0)
pick = [files[i] for i in rng.choice(len(files), 6, replace=False)]

print("======== header of one image")
with fits.open(pick[0]) as h:
    hdr = h[0].header
    for k in ["BUNIT", "EXPOSURE", "TELESCOP", "INSTRUME", "CDELT1", "CDELT2",
              "CTYPE1", "NAXIS1", "EMIN", "EMAX"]:
        if k in hdr:
            print(f"  {k:<9} = {hdr[k]}")
    extra = [k for k in hdr if k not in ("COMMENT", "HISTORY")][:40]
    print(f"  (all keys: {', '.join(extra)})")

print("\n======== quantisation, counts and footprint")
print(f"{'halo':<12}{'min>0':>11}{'counts@2Ms':>13}{'exposed':>9}"
      f"{'exp. radius [arcmin]':>22}")
for p in pick:
    with fits.open(p) as h:
        d = h[0].data.astype(np.float64)
    pos = d[d > 0]
    q = pos.min()
    # If pixels are counts/exposure, every value is an integer multiple of q.
    mult = pos / q
    integer_like = np.mean(np.abs(mult - np.round(mult)) < 1e-3)
    counts = np.round(mult).sum()
    n = d.shape[0]
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(yy - n / 2, xx - n / 2) * 0.492 / 60.0     # arcmin
    finite = np.isfinite(d)
    # radius beyond which fewer than half the pixels in an annulus are finite
    # and non-negative (the unexposed region is 0 or NaN after expmap division)
    edges = np.arange(0, rr.max(), 0.5)
    frac = [np.mean(finite[(rr >= a) & (rr < a + .5)] &
                    (d[(rr >= a) & (rr < a + .5)] != 0)) for a in edges]
    hid = re.search(r"halo_(\d+)_img", p).group(1)
    print(f"{hid:<12}{q:>11.3g}{counts:>13.0f}{finite.mean():>9.1%}"
          f"{'':>8}nonzero fraction by radius: "
          + " ".join(f"{a:.0f}':{f:.2f}" for a, f in zip(edges[::6], frac[::6])))
    print(f"{'':<12}values integer multiples of min>0: {integer_like:.1%}")

print("\n======== real X-ray coverage of the LOFAR targets")
df = pd.read_csv("LoVoCCS_target_list - lovoccs.csv")
key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()
df["k"] = df["name"].map(key)
lofar = {key(re.sub(r"^lotss_|\.fits$", "", os.path.basename(p)))
         for p in glob.glob(os.path.expanduser(
             "~/data/cluster-ml/lotss_images/lotss_*.fits"))}
cols = [c for c in df.columns if c.strip().lower() in
        ("chandra", "erosita", "xmm")]
sub = df[df["k"].isin(lofar)][["name", "redshift"] + cols]
print(sub.to_string(index=False))
for c in cols:
    v = sub[c].astype(str).str.strip()
    have = ~v.isin(["", "nan", "0", "no", "No", "N", "-"])
    print(f"  {c}: {have.sum()}/{len(sub)} LOFAR targets have an entry")
