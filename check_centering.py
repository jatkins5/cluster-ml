"""Is the weight-centroid image centre really offset from the halo?

check_weight_range.py measured the offset against the median shock-cell
position, which is only a proxy for the halo centre. This uses the
authoritative source instead: Group/GroupPos and Group/Group_R_Crit500 from
the snap99 FOF catalogue, indexed by FOF halo ID.

Frame: Radio_generation.py stores pos = Coordinates * a / h0, i.e. physical
kpc, so GroupPos (ckpc/h) gets the same a/h0 conversion. Positions are
unwrapped against the periodic box before differencing.
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np


def load_groups(gc_dir):
    """Concatenate Group fields across chunks; global index == FOF halo ID."""
    files = sorted(glob.glob(os.path.join(gc_dir, "fof_subhalo_tab_*.hdf5")),
                   key=lambda p: int(re.search(r"\.(\d+)\.hdf5$", p).group(1)))
    pos, r500, box, hpar, a = [], [], None, None, None
    for fn in files:
        with h5py.File(fn, "r") as f:
            if box is None:
                box = float(f["Header"].attrs["BoxSize"])
                hpar = float(f["Header"].attrs["HubbleParam"])
                a = float(f["Header"].attrs["Time"])
            if "Group" not in f or f["Header"].attrs["Ngroups_ThisFile"] == 0:
                continue
            pos.append(f["Group/GroupPos"][:])
            r500.append(f["Group/Group_R_Crit500"][:])
    return (np.concatenate(pos), np.concatenate(r500), box, hpar, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groupcat", default="/oscar/data/idellant/Chuiyang/"
                                          "groupcat_classification/groupcat_099")
    ap.add_argument("--glob", default="Radio_Data/radio_FOF*.npz")
    ap.add_argument("--n", type=int, default=0, help="0 = all clusters")
    ap.add_argument("--dataset", default="dataset_512.h5")
    args = ap.parse_args()

    gpos, gr500, box, hpar, a = load_groups(args.groupcat)
    print(f"groups: {len(gpos)}  BoxSize={box:.4g} ckpc/h  h={hpar}  a={a}")
    conv = a / hpar                      # ckpc/h -> physical kpc
    box_kpc = box * conv

    files = sorted(glob.glob(args.glob))
    if args.n:
        files = files[:args.n]

    with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5", "r") as f:
        cat_r500 = {int(h): float(r) * 1000.0
                    for h, r in zip(f["haloID"][:], f["r500c"][:])}

    d_cent, d_med, r5 = [], [], []
    cen_kpc, cat_r5, extent, hids = [], [], [], []
    for fn in files:
        hid = int(re.search(r"FOF(\d+)", os.path.basename(fn)).group(1))
        if hid >= len(gpos):
            print(f"  skip FOF{hid}: beyond catalogue")
            continue
        d = np.load(fn)
        w = d["w"].astype(np.float64)
        pos = d["pos"].astype(np.float64)
        good = np.isfinite(w) & (w > 0) & np.all(np.isfinite(pos), axis=1)
        w, pos = w[good], pos[good]
        if w.size == 0 or w.sum() <= 0:
            continue
        r500_kpc = float(gr500[hid]) * conv
        if not np.isfinite(r500_kpc) or r500_kpc <= 0:
            continue
        halo = gpos[hid].astype(np.float64) * conv
        # Unwrap the periodic box before differencing.
        rel = pos - halo
        rel -= box_kpc * np.round(rel / box_kpc)

        cen = (rel * w[:, None]).sum(axis=0) / w.sum()
        d_cent.append(np.linalg.norm(cen) / r500_kpc)
        d_med.append(np.linalg.norm(np.median(rel, axis=0)) / r500_kpc)
        r5.append(r500_kpc)
        cen_kpc.append(np.linalg.norm(cen))
        # Cross-check the "global group index == FOF halo id" assumption: if it
        # were wrong we would be indexing random low-mass groups, and the two
        # independent r500 values would not agree.
        cat_r5.append(cat_r500.get(hid, np.nan))
        extent.append(np.abs(rel).max())
        hids.append(hid)

    dc, dm, r5 = np.array(d_cent), np.array(d_med), np.array(r5)
    ck, cr, ex = np.array(cen_kpc), np.array(cat_r5), np.array(extent)
    ok = np.isfinite(cr) & (cr > 0)
    ratio = r5[ok] / cr[ok]
    print(f"\nindexing check: groupcat r500 / catalog r500c -> "
          f"median {np.median(ratio):.3f}, "
          f"5-95pct [{np.percentile(ratio, 5):.3f}, "
          f"{np.percentile(ratio, 95):.3f}]")
    print(f"centroid offset [kpc]: median {np.median(ck):.0f}, "
          f"90th {np.percentile(ck, 90):.0f}")
    print(f"particle extent / r500: median {np.median(ex / r5):.2f}, "
          f"max {np.max(ex / r5):.2f}  (centroid must sit inside this)")
    print(f"\n{len(dc)} clusters, r500 median {np.median(r5):.0f} kpc\n")
    for name, v in [("weight centroid (what build_dataset uses)", dc),
                    ("median shock-cell position (old proxy)", dm)]:
        print(f"{name}")
        print(f"   offset from GroupPos [r500]: median {np.median(v):.2f}  "
              f"90th {np.percentile(v, 90):.2f}  max {v.max():.2f}")
        print(f"   fraction beyond 0.5 r500: {(v > 0.5).mean():.0%}, "
              f"beyond 1.0: {(v > 1.0).mean():.0%}, "
              f"beyond the 4 r500 half-width: {(v > 4.0).mean():.1%}\n")
    tsc_correlation(hids, dc, args.dataset)


def tsc_correlation(hids, dc, dataset):
    """Does the offset itself carry merger information?

    build_dataset.py centres each image ON this offset, so whatever signal it
    holds is removed from the images by construction. If it correlates with
    pseudo-TSC, re-centring on GroupPos is not just a bug fix -- it restores a
    feature the CNN has never been able to see.
    """
    from scipy import stats as sps
    with h5py.File(dataset, "r") as f:
        ids = f["meta/halo_id"][:]
        tsc = f["labels/pseudo_tsc"][:]
    tmap = dict(zip(ids.tolist(), tsc.tolist()))
    pairs = [(o, tmap[h]) for h, o in zip(hids, dc) if h in tmap]
    o = np.array([p[0] for p in pairs])
    t = np.array([p[1] for p in pairs])
    ok = np.isfinite(o) & np.isfinite(t)
    o, t = o[ok], t[ok]
    r = sps.spearmanr(o, t)
    print(f"\ncentroid offset vs pseudo-TSC ({len(o)} clusters): "
          f"Spearman rho={r.statistic:+.3f}, p={r.pvalue:.3g}")
    q = np.quantile(o, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"{'offset quartile [r500]':<28}{'median TSC [Gyr]':>18}")
    for i in range(4):
        m = (o >= q[i]) & (o <= q[i + 1])
        print(f"  {q[i]:.2f} - {q[i+1]:.2f}{'':<14}{np.median(t[m]):>18.2f}")


if __name__ == "__main__":
    main()
