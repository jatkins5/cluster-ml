#!/usr/bin/env python3
"""IDEALISED optical catalogue features for the E1/E2 pilot (review rounds
4-7, CODEX_CLAUDE_REVIEW.md). Read-only on Chuiyang's group catalogues and
cutouts.

Membership is the simulation's FoF group (no projection contamination, no
photo-z scatter): an information-content test, NOT an observationally
matched catalogue, and never used for real-cluster claims.

Per cluster (z=0 parent halo), snapshot (99, 91) and projection (xy, yz, xz;
the radio/X-ray order):
  centre_x   idealised X-ray centroid: emission-measure-weighted (m * rho)
             projected centroid of hot (T > 3e5 K), non-star-forming gas
             within a projected r500 of the FoF GroupPos (true r500)
  members    FoF subhalos with stars and total absolute r-band magnitude
             M_r < -19 (SubhaloStellarPhotometrics: total light of all star
             particles, intrinsic, no dust; there is no aperture photometry)
  bcg        brightest member in r within 0.5 r500 of centre_x
  gap12      M_r(2nd) - M_r(1st) among members within 0.5 r500 of centre_x
  off_bcg_x  |BCG - centre_x| / r500
  off_bcg_gal|BCG - number-weighted centroid of members within r500| / r500
  asym       |centroid(members < 0.5 r500) - centroid(members < r500)| / r500
  richness   members with M_r < -20 within r500 of centre_x
Missing cases are NaN with a reason code (no members / fewer than two in
0.5 r500 / no hot gas), never manufactured values. A per-member table
(M_r, g - r, projected R / r500) is saved for the E2 colour features, which
are built inside training folds (train_optical_pilot.py).

Catalogue checks (raised, not assumed): chunks concatenated in file order
give global indices; GroupFirstSub/GroupNsubs must agree with SubhaloGrNr
for every target group; positions converted ckpc/h -> physical kpc with the
snapshot's a and h; periodic wrap with the box size.
"""
import glob
import os
import re

import h5py
import numpy as np
import pandas as pd
from astropy import constants as c

C = "/oscar/data/idellant/Chuiyang"
BANDS = ["U", "B", "V", "K", "g", "r", "i", "z"]
IR, IG = BANDS.index("r"), BANDS.index("g")
PROJ = [(0, 1), (1, 2), (0, 2)]              # xy, yz, xz
MR_MEMBER, MR_RICH = -19.0, -20.0


def read_groupcat(snap):
    d = f"{C}/groupcat_classification/groupcat_{snap:03d}"
    files = sorted(glob.glob(f"{d}/fof_subhalo_tab_{snap:03d}.*.hdf5"),
                   key=lambda p: int(re.search(r"\.(\d+)\.hdf5$", p).group(1)))
    G = {k: [] for k in ("GroupPos", "Group_R_Crit500", "GroupFirstSub", "GroupNsubs")}
    S = {k: [] for k in ("SubhaloGrNr", "SubhaloPos", "SubhaloStellarPhotometrics", "SubhaloMassType")}
    hdr = None
    for fn in files:
        with h5py.File(fn, "r") as f:
            if hdr is None:
                hdr = {k: float(f["Header"].attrs[k]) for k in ("BoxSize", "HubbleParam", "Time", "Redshift")}
            if f["Header"].attrs["Ngroups_ThisFile"] > 0:
                for k in G:
                    G[k].append(f[f"Group/{k}"][:])
            if f["Header"].attrs["Nsubgroups_ThisFile"] > 0:
                for k in S:
                    S[k].append(f[f"Subhalo/{k}"][:])
    G = {k: np.concatenate(v) for k, v in G.items()}
    S = {k: np.concatenate(v) for k, v in S.items()}
    conv = hdr["Time"] / hdr["HubbleParam"]
    G["GroupPos"] = G["GroupPos"] * conv
    G["Group_R_Crit500"] = G["Group_R_Crit500"] * conv
    S["SubhaloPos"] = S["SubhaloPos"] * conv
    return G, S, hdr["BoxSize"] * conv, hdr


def targets(snap):
    """(z=0 parent id, this snapshot's FOF id, subhalo id for the cutout)."""
    if snap == 99:
        out = []
        for p in sorted(glob.glob("Radio_Data/radio_FOF*.npz")):
            fof, sub = map(int, re.search(r"radio_FOF(\d+)_sub(\d+)", p).groups())
            out.append((fof, fof, sub))
        return out
    m = pd.read_csv(f"{C}/TNGCluster_Cutout/fof_halo_to_sub{snap}.csv")
    return [(int(r.FOF_Halo_ID_At99), int(getattr(r, f"FOF_Halo_ID_At{snap}")),
             int(getattr(r, f"Subhalo_ID_At{snap}"))) for r in m.itertuples()]


def gas_centroids(cutout, centre, r500, box, a, h):
    """Emission-measure-weighted projected centroid of hot gas, per projection."""
    with h5py.File(cutout, "r") as f:
        g = f["PartType0"]
        pos = g["Coordinates"][:] * a / h
        m = g["Masses"][:].astype(float)
        rho = g["Density"][:].astype(float)
        u = g["InternalEnergy"][:].astype(float)
        xe = g["ElectronAbundance"][:].astype(float)
        sfr = g["StarFormationRate"][:]
    mu = 4.0 / (1 + 3 * 0.76 + 4 * 0.76 * xe) * c.m_p.cgs.value
    T = (2 / 3) * u / c.k_B.cgs.value * 1e10 * mu
    d = pos - centre
    d -= box * np.round(d / box)
    hot = (T > 3e5) & (sfr == 0) & (np.abs(d).max(axis=1) < 2 * r500)
    w = (m * rho)[hot]
    out = np.full((3, 2), np.nan)
    for k, (i, j) in enumerate(PROJ):
        sel = np.hypot(d[hot, i], d[hot, j]) < r500
        if sel.sum() > 10 and w[sel].sum() > 0:
            out[k] = [np.average(d[hot, i][sel], weights=w[sel]),
                      np.average(d[hot, j][sel], weights=w[sel])]
    return out


def features_one(dpos, mr, gr, r500, xc):
    """dpos (n, 2) member offsets from GroupPos in the projection plane."""
    f = dict(gap12=np.nan, off_bcg_x=np.nan, off_bcg_gal=np.nan, asym=np.nan,
             richness=np.nan, reason="")
    if not np.isfinite(xc).all():
        f["reason"] = "no hot gas centroid"
        return f, None
    R = np.hypot(*(dpos - xc).T) / r500
    inner, inr = R < 0.5, R < 1.0
    f["richness"] = float(((mr < MR_RICH) & inr).sum())
    if inr.sum() == 0:
        f["reason"] = "no members within r500"
        return f, None
    if inner.sum() >= 1:
        ib = np.where(inner)[0][np.argmin(mr[inner])]
        f["off_bcg_x"] = np.hypot(*(dpos[ib] - xc)) / r500
        f["off_bcg_gal"] = np.hypot(*(dpos[ib] - dpos[inr].mean(axis=0))) / r500
    else:
        f["reason"] = "no member within 0.5 r500"
    if inner.sum() >= 2:
        s = np.sort(mr[inner])
        f["gap12"] = s[1] - s[0]
    elif not f["reason"]:
        f["reason"] = "fewer than two members within 0.5 r500"
    if inner.sum() >= 3 and inr.sum() >= 3:
        f["asym"] = np.hypot(*(dpos[inner].mean(axis=0) - dpos[inr].mean(axis=0))) / r500
    elif not f["reason"]:
        f["reason"] = "fewer than three members for asymmetry"
    members = np.column_stack([mr[inr], gr[inr], R[inr]])
    return f, members


def main():
    rows, mem = [], []
    for snap in (99, 91):
        G, S, box, hdr = read_groupcat(snap)
        a, h = hdr["Time"], hdr["HubbleParam"]
        grnr = S["SubhaloGrNr"]
        phot = S["SubhaloStellarPhotometrics"]
        mstar = S["SubhaloMassType"][:, 4]
        tg = targets(snap)
        print(f"snapshot {snap}: z = {hdr['Redshift']:.4f}, {len(G['GroupPos'])} groups, "
              f"{len(grnr)} subhalos, {len(tg)} targets", flush=True)
        bad = 0
        for n, (parent, fof, sub) in enumerate(tg):
            first, ns = int(G["GroupFirstSub"][fof]), int(G["GroupNsubs"][fof])
            idx = np.where(grnr == fof)[0]
            if ns != len(idx) or (ns and (idx.min() != first or idx.max() != first + ns - 1)):
                bad += 1
                raise SystemExit(f"snap {snap} FOF{fof}: GroupFirstSub/GroupNsubs disagree with SubhaloGrNr")
            centre = G["GroupPos"][fof]
            r500 = float(G["Group_R_Crit500"][fof])
            ok = (mstar[idx] > 0) & np.isfinite(phot[idx, IR]) & (phot[idx, IR] < MR_MEMBER)
            sid = idx[ok]
            d3 = S["SubhaloPos"][sid] - centre
            d3 -= box * np.round(d3 / box)
            mr = phot[sid, IR]
            gr = phot[sid, IG] - phot[sid, IR]
            cut = f"{C}/TNGCluster_Cutout/snap{snap}/cutout_sub{sub}_FOF{fof}.hdf5"
            xc = (gas_centroids(cut, centre, r500, box, a, h) if os.path.exists(cut)
                  else np.full((3, 2), np.nan))
            for k, (i, j) in enumerate(PROJ):
                f, members = features_one(d3[:, [i, j]], mr, gr, r500, xc[k])
                rows.append(dict(group_id=parent, snap=snap, fof=fof, proj=k, r500_kpc=r500,
                                 n_members_total=len(sid), **f))
                if members is not None:
                    for mm in members:
                        mem.append((parent, snap, k, *mm))
            if n % 50 == 0:
                print(f"  {n}/{len(tg)}", flush=True)
        print(f"  catalogue mapping checks: {len(tg)} groups consistent")
    df = pd.DataFrame(rows)
    df.to_csv("optical_features_idealised.csv", index=False)
    np.save("optical_members_idealised.npy", np.array(mem, dtype=float))
    print(f"wrote optical_features_idealised.csv ({len(df)} rows) and "
          f"optical_members_idealised.npy ({len(mem)} member rows)")
    for col in ("gap12", "off_bcg_x", "off_bcg_gal", "asym", "richness"):
        v = df[col]
        print(f"  {col:<12} missing {v.isna().mean():.1%}; median {v.median():.3f}")
    print("  missing reasons:", df["reason"].replace("", "none").value_counts().to_dict())


if __name__ == "__main__":
    main()
