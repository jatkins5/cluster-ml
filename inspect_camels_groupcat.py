"""Check whether the CAMELS group catalogs can support the mass-ratio proxy.

The proxy validated on TNG needs, per snapshot: companion subhalo masses,
companion positions, and Group_R_Crit200 to put the separation in scaled
units. `Halomass_history` has none of that. This verifies the fields exist
in the CAMELS SUBFIND output and demonstrates the ratio/separation census
on a few zooms, plus the snapshot->cosmic-time coverage that sets how much
of the history is censored.
"""
import glob
import os
import sys

import h5py
import numpy as np

BASE = os.path.expanduser("~/data/Chuiyang/Camels/Groupcat/GZ28")
NTOP = 3
NEED = ["Group/Group_R_Crit200", "Group/Group_M_Crit200", "Group/GroupPos",
        "Subhalo/SubhaloMass", "Subhalo/SubhaloPos", "Subhalo/SubhaloGrNr"]


def chunk_files(zoom, snap):
    d = f"{BASE}/{zoom}/groups_{snap:03d}"
    return sorted(glob.glob(f"{d}/fof_subhalo_tab_{snap:03d}.*.hdf5"),
                  key=lambda x: int(x.split(".")[-2]))


def dump_schema(zoom, snap):
    files = chunk_files(zoom, snap)
    print(f"=== schema: {zoom} snap {snap} ({len(files)} chunks) ===")
    with h5py.File(files[0], "r") as f:
        print("Header:")
        for k, v in f["Header"].attrs.items():
            print(f"    {k} = {v}")
        for grp in ("Group", "Subhalo"):
            if grp not in f:
                print(f"  [{grp}] ABSENT")
                continue
            print(f"  [{grp}]")
            for k in f[grp]:
                print(f"    {k:<28} {f[grp][k].shape} {f[grp][k].dtype}")
    missing = []
    with h5py.File(files[0], "r") as f:
        for path in NEED:
            if path not in f:
                missing.append(path)
    print("\nrequired fields: "
          + ("ALL PRESENT" if not missing else f"MISSING {missing}"))
    return not missing


def read_snapshot(zoom, snap):
    """Concatenate the chunked catalog for one zoom/snapshot."""
    gr, gm, gpos, sm, sp, sg = [], [], [], [], [], []
    a = h = None
    for c in chunk_files(zoom, snap):
        with h5py.File(c, "r") as f:
            if a is None:
                a = f["Header"].attrs["Time"]
                h = f["Header"].attrs["HubbleParam"]
            if "Group" in f and "GroupPos" in f["Group"]:
                gr.append(f["Group/Group_R_Crit200"][:])
                gm.append(f["Group/Group_M_Crit200"][:])
                gpos.append(f["Group/GroupPos"][:])
            if "Subhalo" in f and "SubhaloMass" in f["Subhalo"]:
                sm.append(f["Subhalo/SubhaloMass"][:])
                sp.append(f["Subhalo/SubhaloPos"][:])
                sg.append(f["Subhalo/SubhaloGrNr"][:].astype(np.int64))
    if not gr or not sm:
        return None
    return dict(r200=np.concatenate(gr), m200=np.concatenate(gm),
                gpos=np.concatenate(gpos), smass=np.concatenate(sm),
                spos=np.concatenate(sp), sgrnr=np.concatenate(sg),
                a=a, h=h)


def census(d, group=0):
    """Mass ratios and separations of the companions of `group`."""
    m = d["sgrnr"] == group
    if m.sum() < 2:
        return None
    mass, pos = d["smass"][m], d["spos"][m]
    order = np.argsort(-mass)
    mass, pos = mass[order], pos[order]
    cen = pos[0]
    r200 = d["r200"][group]
    mu = mass[1:] / mass[0]
    sep = np.linalg.norm(pos[1:] - cen, axis=1) / max(r200, 1e-9)
    return mu[:NTOP], sep[:NTOP], mass[0], r200, m.sum()


def main():
    zooms = sorted(os.listdir(BASE), key=lambda x: int(x.split("_")[1]))
    print(f"{len(zooms)} zooms under {BASE}\n")
    if not dump_schema(zooms[0], 90):
        sys.exit("required fields absent -- proxy not buildable as designed")

    snaps = sorted(int(p.split("_")[-1])
                   for p in os.listdir(f"{BASE}/{zooms[0]}")
                   if p.startswith("groups_"))
    print(f"\nsnapshots per zoom: {len(snaps)} ({snaps[0]}-{snaps[-1]})")

    print(f"\n=== companion census at snap 90, group 0, first 8 zooms ===")
    print(f"{'zoom':<10} {'nsub':>6} {'M_cen':>10} {'R200':>9} "
          f"{'mu1':>7} {'sep1':>7} {'mu2':>7} {'sep2':>7}")
    for z in zooms[:8]:
        d = read_snapshot(z, 90)
        if d is None:
            print(f"{z:<10} no catalog")
            continue
        c = census(d)
        if c is None:
            print(f"{z:<10} {'<2 subhalos':>6}")
            continue
        mu, sep, mcen, r200, nsub = c
        g = lambda v, i: (f"{v[i]:.3f}" if len(v) > i else "-")
        print(f"{z:<10} {nsub:>6} {mcen:>10.3f} {r200:>9.1f} "
              f"{g(mu, 0):>7} {g(sep, 0):>7} {g(mu, 1):>7} {g(sep, 1):>7}")

    print(f"\n=== history of zoom {zooms[0]}: snapshots with a hit ===")
    print(f"(hit = companion mass ratio > 0.02 inside 1.0 R200)")
    print(f"{'snap':>5} {'a':>7} {'z':>7} {'mu1':>8} {'sep1':>8} {'hit':>5}")
    for s in snaps[::10] + [snaps[-1]]:
        d = read_snapshot(zooms[0], s)
        if d is None:
            print(f"{s:>5}  no catalog")
            continue
        c = census(d)
        if c is None:
            print(f"{s:>5} {d['a']:>7.3f} {1/d['a']-1:>7.3f}  <2 subhalos")
            continue
        mu, sep, _, _, _ = c
        hit = bool(((mu > 0.02) & (sep < 1.0)).any())
        print(f"{s:>5} {d['a']:>7.3f} {1 / d['a'] - 1:>7.3f} "
              f"{mu[0]:>8.4f} {sep[0]:>8.3f} {str(hit):>5}")


if __name__ == "__main__":
    main()
