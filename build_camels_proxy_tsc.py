"""Build the mass-ratio proxy TSC label for the CAMELS GZ28 zooms.

This is the label the CAMELS experiment was missing. The mass-jump label in
camels_labels.py fires on ordinary accretion because it only sees the main
halo's mass history; the definition validated against TNG truth in
validate_merger_proxy.py instead asks whether a companion above a mass ratio
was physically inside R200, which is what dates a collision.

Per zoom, for every snapshot, this records the closest companion above each
mass-ratio threshold in units of R200. The combine stage turns that into
TSC = t(snap 90) - t(last snapshot with a hit).

Subhaloes are ordered by group, so group 0's subhaloes live at the front of
chunk 0 and the later chunks usually need not be opened at all.
"""
import argparse
import glob
import os

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

BASE = os.path.expanduser("~/data/Chuiyang/Camels")
GROUPCAT = f"{BASE}/Groupcat/GZ28"
PARTS = "camels_proxy_parts"
TARGET_SNAP = 90
THRESHOLDS = (0.02, 0.05, 0.10, 0.20, 1.0 / 3.0)
THR = 0.02       # winner of the TNG grid search
RAD = 1.0        # R200


def zoom_list():
    return sorted((d for d in os.listdir(GROUPCAT) if d.startswith("GZ28_")),
                  key=lambda x: int(x.split("_")[1]))


def read_group0(zoom, snap):
    """Group 0's R200 and its subhalo masses/positions, from as few chunks
    as possible."""
    files = sorted(glob.glob(f"{GROUPCAT}/{zoom}/groups_{snap:03d}/"
                             f"fof_subhalo_tab_{snap:03d}.*.hdf5"),
                   key=lambda x: int(x.split(".")[-2]))
    if not files:
        return None
    r200 = a = om0 = h = None
    mass, pos = [], []
    for path in files:
        with h5py.File(path, "r") as f:
            if r200 is None:
                at = f["Header"].attrs
                a, om0, h = (float(at["Time"]), float(at["Omega0"]),
                             float(at["HubbleParam"]))
                if "Group" not in f or "Group_R_Crit200" not in f["Group"]:
                    return None
                if len(f["Group/Group_R_Crit200"]) < 1:
                    return None
                r200 = float(f["Group/Group_R_Crit200"][0])
            if "Subhalo" not in f or "SubhaloMass" not in f["Subhalo"]:
                break
            gr = f["Subhalo/SubhaloGrNr"][:]
            keep = gr == 0
            if not keep.any():
                break
            mass.append(f["Subhalo/SubhaloMass"][:][keep])
            pos.append(f["Subhalo/SubhaloPos"][:][keep])
            if not keep[-1]:        # group 0 ended inside this chunk
                break
    if not mass:
        return None
    return (np.concatenate(mass), np.concatenate(pos), r200, a, om0, h)


def min_sep_at(zoom, snap):
    """Closest companion above each threshold, in R200."""
    got = read_group0(zoom, snap)
    out = np.full(len(THRESHOLDS), np.inf)
    if got is None:
        return out, np.nan, np.nan, np.nan, np.nan
    mass, pos, r200, a, om0, h = got
    if len(mass) < 2 or not np.isfinite(r200) or r200 <= 0:
        return out, r200, a, om0, h
    order = np.argsort(-mass)
    mass, pos = mass[order], pos[order]
    ratios = mass[1:] / mass[0]
    sep = np.linalg.norm(pos[1:] - pos[0], axis=1) / r200
    for t, thr in enumerate(THRESHOLDS):
        q = ratios > thr
        if q.any():
            out[t] = sep[q].min()
    return out, r200, a, om0, h


def stage_zooms(args):
    os.makedirs(PARTS, exist_ok=True)
    zooms = zoom_list()[args.start:args.end]
    snaps = list(range(TARGET_SNAP + 1))
    out = f"{PARTS}/zooms_{args.start:04d}_{args.end:04d}.npz"
    if os.path.exists(out) and not args.force:
        print(f"already done -> {out}")
        return

    zid = np.array([int(z.split("_")[1]) for z in zooms])
    ms = np.full((len(zooms), len(snaps), len(THRESHOLDS)), np.inf)
    scale = np.full((len(zooms), len(snaps)), np.nan)
    om0 = np.full(len(zooms), np.nan)
    hub = np.full(len(zooms), np.nan)
    for i, z in enumerate(zooms):
        for k, s in enumerate(snaps):
            v, _, a, o, h = min_sep_at(z, s)
            ms[i, k] = v
            scale[i, k] = a
            if s == TARGET_SNAP:
                om0[i], hub[i] = o, h
        hits = np.isfinite(ms[i, :, 0]).sum()
        print(f"{z}: snapshots with a companion above {THRESHOLDS[0]}: "
              f"{hits}/{len(snaps)}", flush=True)
    np.savez(out, zoom_id=zid, min_sep=ms, scale=scale, om0=om0, hub=hub,
             snaps=np.array(snaps), thresholds=np.array(THRESHOLDS))
    print(f"saved -> {out}")


def stage_combine(args):
    parts = sorted(glob.glob(f"{PARTS}/zooms_*.npz"))
    if not parts:
        raise SystemExit("no parts; run --stage zooms first")
    d = [np.load(p) for p in parts]
    zid = np.concatenate([x["zoom_id"] for x in d])
    ms = np.concatenate([x["min_sep"] for x in d])
    scale = np.concatenate([x["scale"] for x in d])
    om0 = np.concatenate([x["om0"] for x in d])
    hub = np.concatenate([x["hub"] for x in d])
    snaps = d[0]["snaps"]
    order = np.argsort(zid)
    zid, ms, scale, om0, hub = (zid[order], ms[order], scale[order],
                                om0[order], hub[order])
    print(f"combined {len(zid)} zooms from {len(parts)} parts")

    t = list(THRESHOLDS).index(THR)
    hit = ms[:, :, t] < RAD
    n = len(zid)
    tsc = np.full(n, np.nan)
    span = np.full(n, np.nan)
    for i in range(n):
        ok = np.isfinite(scale[i])
        if not ok.any() or not np.isfinite(om0[i]) or not np.isfinite(hub[i]):
            continue
        cosmo = FlatLambdaCDM(H0=100.0 * hub[i], Om0=om0[i])
        z = 1.0 / np.clip(scale[i], 1e-6, None) - 1.0
        age = np.full(len(snaps), np.nan)
        age[ok] = cosmo.age(np.clip(z[ok], 0.0, None)).to("Gyr").value
        t0 = age[snaps == TARGET_SNAP]
        if not len(t0) or not np.isfinite(t0[0]):
            continue
        span[i] = t0[0] - np.nanmin(age)
        w = np.where(hit[i] & ok)[0]
        if len(w):
            tsc[i] = t0[0] - age[w.max()]

    uncens = np.isfinite(tsc)
    filled = np.where(uncens, tsc, span)
    print(f"definition: mass ratio > {THR} within {RAD} R200")
    print(f"uncensored {uncens.sum()}/{n}  "
          f"({100 * uncens.mean():.1f}%; TNG had 303/352 = 86.1%)")
    print(f"history span: median {np.nanmedian(span):.2f} Gyr "
          f"(TNG had 4.62 Gyr)")
    ok = np.isfinite(filled)
    print(f"TSC: median {np.nanmedian(filled[ok]):.2f}  "
          f"IQR [{np.nanpercentile(filled[ok], 25):.2f}, "
          f"{np.nanpercentile(filled[ok], 75):.2f}]  "
          f"max {np.nanmax(filled[ok]):.2f} Gyr")
    print(f"TSC <= 2 Gyr: {(filled[ok] <= 2).sum()}/{ok.sum()}   "
          f"(TNG label: 606/1056 projections)")

    with h5py.File("camels_proxy_tsc.hdf5", "w") as f:
        f.create_dataset("zoom_id", data=zid)
        f.create_dataset("tsc_proxy", data=filled)
        f.create_dataset("tsc_proxy_unc", data=np.where(uncens, tsc, np.nan))
        f.create_dataset("uncensored", data=uncens.astype(np.int8))
        f.create_dataset("span_gyr", data=span)
        f.attrs["thr"] = THR
        f.attrs["rad_r200"] = RAD
    print("saved -> camels_proxy_tsc.hdf5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["zooms", "combine"], default="zooms")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (stage_zooms if a.stage == "zooms" else stage_combine)(a)
