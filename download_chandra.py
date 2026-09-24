#!/usr/bin/env python3
"""Download archival Chandra data for the LOFAR-covered LoVoCCS targets.

Observation list from HEASARC's chanmaster table (the same query as
query_xray_archive.py), restricted to observations that are actually public
-- chanmaster also lists scheduled and proprietary ones, so the earlier
exposure totals are an upper bound until this filter is applied.

Files come from the CXC's public FTP mirror over HTTPS:
    https://cxc.cfa.harvard.edu/cdaftp/byobsid/<last digit>/<obsid>/primary/
We take the level-2 event list (evt2), the field-of-view region (fov1) and
the bad-pixel file (bpix1): enough to bin a 0.1-2 keV counts image on the
same band as the mocks and to know which sky the chips covered. Exposure
maps need CIAO, which is not installed; that is a processing decision for
later, not a download one.

Detector matters: the mocks are ACIS-I, and many archival cluster
observations are ACIS-S (different chip layout, field and response), so it
is recorded per observation rather than assumed.

Data live outside the repo, in ~/data/cluster-ml/chandra/<target>/<obsid>/.
"""
import argparse
import glob
import os
import re
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.heasarc import Heasarc

BASE = "https://cxc.cfa.harvard.edu/cdaftp/byobsid"
WANT = ("evt2", "fov1", "bpix1", "asol1")   # asol: fluximage needs the aspect solution
key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()


def targets():
    df = pd.read_csv("LoVoCCS_target_list - lovoccs.csv")
    df["k"] = df["name"].map(key)
    lofar = {key(re.sub(r"^lotss_|\.fits$", "", os.path.basename(p)))
             for p in glob.glob(os.path.expanduser(
                 "~/data/cluster-ml/lotss_images/lotss_*.fits"))}
    t = df[df["k"].isin(lofar)].copy()
    t["ra"] = pd.to_numeric(t["ra(deg)"], errors="coerce")
    t["dec"] = pd.to_numeric(t["dec(deg)"], errors="coerce")
    return t


def observations(t, radius_arcmin):
    h = Heasarc()
    rows = []
    for _, r in t.iterrows():
        c = SkyCoord(r.ra, r.dec, unit="deg")
        try:
            res = h.query_region(c, catalog="chanmaster",
                                 radius=radius_arcmin * u.arcmin)
        except Exception:
            res = None
        if res is None or len(res) == 0:
            continue
        tab = res.to_pandas()
        tab.columns = [c.lower() for c in tab.columns]
        for _, o in tab.iterrows():
            rows.append(dict(target=r["name"], key=r["k"], z=r["redshift"],
                             **{k: o[k] for k in tab.columns}))
    return pd.DataFrame(rows)


def listing(obsid):
    url = f"{BASE}/{obsid % 10}/{obsid}/primary/"
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", "replace")
    return url, sorted(set(re.findall(r'href="([^"]+\.fits(?:\.gz)?)"', html)))


def fetch(url, path, tries=3):
    for k in range(tries):
        try:
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
            return True
        except Exception as e:
            print(f"      retry {k + 1}: {e}")
            time.sleep(5 * (k + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-arcmin", type=float, default=6.0)
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/data/cluster-ml/chandra"))
    ap.add_argument("--probe", action="store_true",
                    help="print the observation table and one directory "
                         "listing, download nothing")
    args = ap.parse_args()

    obs = observations(targets(), args.radius_arcmin)
    print(f"chanmaster rows: {len(obs)}; columns: {list(obs.columns)}")
    now = datetime.now(timezone.utc)

    def is_public(r):
        st = str(r.get("status", "")).strip().lower()
        pd_ = str(r.get("public_date", "")).strip()
        ok_status = st in ("archived", "") or "archiv" in st
        try:
            pub = pd.to_datetime(pd_, utc=True)
            ok_date = pub <= now
        except Exception:
            ok_date = True        # unknown: let the download decide
        return ok_status and ok_date

    obs["public"] = obs.apply(is_public, axis=1)
    obs["obsid"] = obs["obsid"].astype(int)
    cols = [c for c in ["target", "obsid", "detector", "exposure", "status",
                        "public_date", "public"] if c in obs.columns]
    print(obs[cols].to_string(index=False))
    pub = obs[obs["public"]]
    exp_col = "exposure" if "exposure" in obs.columns else None
    if exp_col:
        tot = pub.groupby("target")[exp_col].sum() / 1e3
        print(f"\npublic: {len(pub)} observations of {pub['target'].nunique()}"
              f" targets; total ks per target median {tot.median():.0f}, "
              f"range {tot.min():.0f}-{tot.max():.0f}")
    if "detector" in pub.columns:
        print("detectors:", pub["detector"].astype(str).str.strip()
              .value_counts().to_dict())

    if args.probe:
        if len(pub):
            url, files = listing(int(pub["obsid"].iloc[0]))
            print(f"\n{url}\n  " + "\n  ".join(files))
        return

    os.makedirs(args.out_dir, exist_ok=True)
    pub.to_csv(os.path.join(args.out_dir, "observations.csv"), index=False)
    got = 0
    for _, r in pub.iterrows():
        oid = int(r["obsid"])
        d = os.path.join(args.out_dir, key(r["target"]), str(oid))
        os.makedirs(d, exist_ok=True)
        try:
            url, files = listing(oid)
        except Exception as e:
            print(f"  {r['target']} {oid}: listing failed ({e})")
            continue
        sel = [f for f in files if any(w in f for w in WANT)]
        if not any("evt2" in f for f in sel):
            print(f"  {r['target']} {oid}: no evt2 in {files}")
            continue
        ok = True
        for f in sel:
            path = os.path.join(d, os.path.basename(f))
            if os.path.exists(path):
                continue
            ok &= fetch(url + os.path.basename(f), path)
        got += ok
        print(f"  {r['target']:<18} {oid:>6}  "
              f"{str(r.get('detector', '')).strip():<8}"
              f"{float(r.get('exposure', np.nan)) / 1e3:>7.1f} ks  "
              f"{'ok' if ok else 'INCOMPLETE'}  ({len(sel)} files)")
    print(f"\ndownloaded {got}/{len(pub)} observations into {args.out_dir}")


if __name__ == "__main__":
    main()
