"""Are our local LoTSS cutouts comparable to a standard DR3 extraction?

The forward model takes each target's measured rms and uses it to set the mock
noise, so if the local cutouts are systematically noisier or more incomplete
than the survey they claim to come from, every mock inherits that. Compares
each local file against a fresh DR3 cutout at the same position.
"""
import argparse
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from astropy.io import fits


def clipped_rms(d):
    v = d[np.isfinite(d)]
    if v.size < 100:
        return np.nan
    for _ in range(5):
        s = v.std()
        if not np.isfinite(s) or s <= 0:
            return np.nan
        v = v[np.abs(v - np.median(v)) < 3 * s]
    return v.std()


def fetch(ra, dec, size, path, release="dr3"):
    q = urllib.parse.urlencode({"pos": f"{ra:.5f} {dec:.5f}", "size": f"{size:g}"})
    url = f"https://lofar-surveys.org/{release}-cutout.fits?{q}"
    with urllib.request.urlopen(url, timeout=180) as r:
        blob = r.read()
    if not blob.startswith(b"SIMPLE"):
        return False
    open(path, "wb").write(blob)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob-dir", default=os.path.expanduser(
        "~/data/cluster-ml/lotss_images"))
    ap.add_argument("--targets-csv", default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--tmp-dir", default="lotss_fields/_provenance")
    ap.add_argument("--size-arcmin", type=float, default=20.0)
    args = ap.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    df = pd.read_csv(args.targets_csv)
    df["key"] = df["name"].astype(str).str.replace(" ", "")
    ra = dict(zip(df["key"], pd.to_numeric(df["ra(deg)"], errors="coerce")))
    dec = dict(zip(df["key"], pd.to_numeric(df["dec(deg)"], errors="coerce")))

    print(f"{'target':<14}{'local rms':>12}{'dr3 rms':>11}{'ratio':>8}"
          f"{'loc finite':>12}{'dr3 finite':>12}")
    rows = []
    for p in sorted(os.listdir(args.glob_dir)):
        if not p.endswith(".fits"):
            continue
        key = re.sub(r"^lotss_|\.fits$", "", p)
        if key not in ra or not np.isfinite(ra[key]):
            continue
        with fits.open(os.path.join(args.glob_dir, p)) as h:
            dl = np.squeeze(h[0].data).astype(np.float64)
        out = os.path.join(args.tmp_dir, f"{key}.fits")
        if not os.path.exists(out):
            try:
                if not fetch(ra[key], dec[key], args.size_arcmin, out):
                    print(f"{key:<14}   (no DR3 coverage)")
                    continue
            except Exception as e:
                print(f"{key:<14}   ({type(e).__name__})")
                continue
            time.sleep(2)
        with fits.open(out) as h:
            dd = np.squeeze(h[0].data).astype(np.float64)
        rl, rd = clipped_rms(dl), clipped_rms(dd)
        fl, fd = np.isfinite(dl).mean(), np.isfinite(dd).mean()
        rows.append((key, rl, rd, fl, fd))
        print(f"{key:<14}{rl*1e3:>12.3f}{rd*1e3:>11.3f}{rl/rd:>8.2f}"
              f"{fl:>12.2f}{fd:>12.2f}")

    a = pd.DataFrame(rows, columns=["key", "local", "dr3", "fl", "fd"])
    print(f"\nrms ratio local/DR3: median {np.median(a['local']/a['dr3']):.2f} "
          f"[{np.percentile(a['local']/a['dr3'],10):.2f}, "
          f"{np.percentile(a['local']/a['dr3'],90):.2f}]")
    print(f"local finite fraction: median {a['fl'].median():.2f}, "
          f"files below 0.9: {(a['fl'] < 0.9).sum()}/{len(a)}")
    print(f"DR3   finite fraction: median {a['fd'].median():.2f}")
    a.to_csv("obs_provenance.csv", index=False)


if __name__ == "__main__":
    main()
