#!/usr/bin/env python3
"""Download real LoTSS fields to use as injection backgrounds.

Bottrell et al. (2019, MNRAS 490, 5390) find that inserting simulated sources
into *real* survey fields matters far more than any amount of synthetic
noise/PSF modelling: their CNNs reached ~65% on survey-realistic data when
trained on analytic noise+PSF ("SemiReal", which is exactly what our forward
model produces) versus 87.1% when trained on sources inserted into real SDSS
fields. Botteon et al. (2023, A&A 672, A41) do the radio-specific version,
injecting mock haloes into real LoTSS data to derive upper limits.

Positions are drawn as random offsets from the LoVoCCS targets rather than
from a coverage footprint we would have to guess at. An offset of 1-3 degrees
is >50 Mpc at these redshifts, so the field is independent sky, while staying
inside the same survey area that produced our observations.

The service wants `pos=RA DEC` space-separated; a comma silently returns an
HTML login page instead of FITS.
"""
import argparse
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from astropy.io import fits

# DR3, not DR2: the LoVoCCS targets sit mostly at low declination,
# where DR2 returns "No suitable mosaic found". A fresh DR3 cutout
# reproduces each local observation's rms to 1.00, confirming the
# cutouts on disk are DR3 (check_obs_provenance.py).
BASE = "https://lofar-surveys.org/dr3-cutout.fits"


def fetch(ra, dec, size_arcmin, path, timeout=180):
    # "RA DEC" space-separated; a comma is silently rejected with an HTML page.
    q = urllib.parse.urlencode({"pos": f"{ra:.5f} {dec:.5f}",
                                "size": f"{size_arcmin:g}"})
    with urllib.request.urlopen(f"{BASE}?{q}", timeout=timeout) as r:
        blob = r.read()
    # An out-of-coverage or malformed request returns the site's HTML.
    if not blob.startswith(b"SIMPLE"):
        return False
    with open(path, "wb") as f:
        f.write(blob)
    return True


def usable(path, min_finite=0.9):
    try:
        with fits.open(path) as h:
            d = np.squeeze(h[0].data).astype(np.float64)
            hdr = h[0].header
    except Exception:
        return None
    if d.ndim != 2 or min(d.shape) < 64:
        return None
    finite = np.isfinite(d).mean()
    if finite < min_finite:
        return None
    v = d[np.isfinite(d)]
    # Sigma-clipped rms, matching forward_model_lotss.sigma_clipped_rms
    for _ in range(5):
        s = v.std()
        if not np.isfinite(s) or s <= 0:
            return None
        v = v[np.abs(v - np.median(v)) < 3 * s]
    rms = v.std()
    if not np.isfinite(rms) or rms <= 0:
        return None
    return dict(shape=d.shape, finite=finite, rms=float(rms),
                cdelt=float(abs(hdr.get("CDELT2", hdr.get("CD2_2", 0)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-csv",
                    default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--out-dir", default="lotss_fields")
    ap.add_argument("--per-target", type=int, default=8)
    ap.add_argument("--size-arcmin", type=float, default=20.0)
    ap.add_argument("--min-offset-deg", type=float, default=1.0)
    ap.add_argument("--max-offset-deg", type=float, default=3.0)
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="pause between requests; this is a shared service")
    ap.add_argument("--restrict-to-local", action="store_true",
                    help="only offset from targets we already have cutouts "
                         "for, which guarantees survey coverage")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.targets_csv)
    df["key"] = df["name"].astype(str).str.replace(" ", "")
    ra = pd.to_numeric(df["ra(deg)"], errors="coerce")
    dec = pd.to_numeric(df["dec(deg)"], errors="coerce")
    ok = np.isfinite(ra) & np.isfinite(dec)
    targets = list(zip(df["key"][ok], ra[ok], dec[ok]))
    if args.restrict_to_local:
        import glob as _g
        have = {os.path.basename(f)[6:-5]
                for f in _g.glob(os.path.expanduser(
                    "~/data/cluster-ml/lotss_images/lotss_*.fits"))}
        targets = [t for t in targets if t[0] in have]
    print(f"{len(targets)} targets, {args.per_target} offsets each")

    rng = np.random.default_rng(args.seed)
    kept, tried = [], 0
    for key, r0, d0 in targets:
        for k in range(args.per_target):
            path = os.path.join(args.out_dir, f"bg_{key}_{k:02d}.fits")
            if os.path.exists(path):
                continue
            rad = rng.uniform(args.min_offset_deg, args.max_offset_deg)
            th = rng.uniform(0, 2 * np.pi)
            d1 = d0 + rad * np.sin(th)
            r1 = r0 + rad * np.cos(th) / max(np.cos(np.radians(d1)), 0.1)
            if not (-90 < d1 < 90):
                continue
            tried += 1
            try:
                got = fetch(r1 % 360.0, d1, args.size_arcmin, path)
            except Exception as e:
                print(f"  {key}_{k:02d}: {type(e).__name__}")
                got = False
            time.sleep(args.sleep)
            if not got:
                if os.path.exists(path):
                    os.remove(path)
                continue
            info = usable(path)
            if info is None:
                os.remove(path)
                continue
            kept.append((f"{key}_{k:02d}", r1 % 360.0, d1, info))
            print(f"  {key}_{k:02d}: {info['shape']} "
                  f"rms={info['rms']*1e3:.3f} mJy/bm "
                  f"finite={info['finite']:.2f}")

    print(f"\nkept {len(kept)}/{tried} attempted")
    if kept:
        rms = np.array([k[3]["rms"] for k in kept])
        print(f"rms: median {np.median(rms)*1e3:.3f} mJy/beam "
              f"[{rms.min()*1e3:.3f}, {rms.max()*1e3:.3f}]")
        pd.DataFrame([{"name": n, "ra": r, "dec": d, **i}
                      for n, r, d, i in kept]).to_csv(
            os.path.join(args.out_dir, "fields.csv"), index=False)


if __name__ == "__main__":
    main()
