"""Validate a tree-free merger proxy against TNG-Cluster's truth TSC.

CAMELS has no merger trees, so any TSC label there must be proxied from
per-snapshot group catalogs. Before rebuilding the CAMELS label we test
the candidate proxy on TNG-Cluster, where the merger-tree truth exists.

Stage 1 (--stage truth) deconstructs the training label: recompute TSC
from cluster_mergers.hdf5 under various Mass_ratio cuts and see which
reproduces TSC_eachhalo_snap99. This tells us what merger definition
our regression target actually encodes.

Stage 2 (--stage proxy) computes the subhalo-mass-ratio proxy -- at each
snapshot, mu = M_sub2/M_sub1 within the primary FoF group; the merger
time is the last snapshot with mu above threshold -- and correlates it
with the truth.
"""
import argparse
import os

import h5py
import numpy as np
from scipy import stats

TRUTH_EVENTS = "TSC_Cutimages/cluster_mergers.hdf5"
TRUTH_TSC = "TSC_Cutimages/TSC_eachhalo_snap99.hdf5"
TARGET_CAT = os.path.expanduser("~/data/Chuiyang/targethalo_cat_TNGCluster")
GROUPCAT = os.path.expanduser("~/data/Chuiyang/groupcat_classification")
SNAP0 = 99

RATIO_CUTS = [-np.inf, 0.0, 0.03, 0.05, 0.10, 0.20, 0.25, 1.0 / 3.0, 0.50]


def snap_time_table():
    """snap -> cosmic time [Gyr], read off the truth catalog's own
    (Snap_coll, T_coll) pairs so the proxy shares the truth's clock."""
    with h5py.File(TRUTH_EVENTS, "r") as f:
        snap = f["Snap_coll"][:]
        t = f["T_coll"][:]
    table = {}
    for s in np.unique(snap):
        vals = t[snap == s]
        table[int(s)] = float(np.median(vals))
    return table


def load_truth():
    with h5py.File(TRUTH_TSC, "r") as f:
        halo_id = f["halo_id"][:]
        tsc = f["tsc_gyr"][:]
    return halo_id, tsc


def stage_truth():
    tab = snap_time_table()
    snaps = np.array(sorted(tab))
    print(f"snap->time table from truth catalog: snaps {snaps.min()}-"
          f"{snaps.max()} (n={len(snaps)})")
    t99 = tab.get(SNAP0)
    print(f"t(snap 99) = {t99} Gyr" if t99 else "snap 99 absent from table")

    with h5py.File(TRUTH_EVENTS, "r") as f:
        ev_halo = f["HaloID"][:]
        ratio = f["Mass_ratio"][:]
        t_coll = f["T_coll"][:]
        tsc_full = f["TSC_full"][:]
        snap_coll = f["Snap_coll"][:]

    print(f"\nevents: {len(ev_halo)}  unique halos: {len(np.unique(ev_halo))}")
    print(f"Mass_ratio: min={ratio.min():.4f} med={np.median(ratio):.4f} "
          f"max={ratio.max():.4f}")
    for c in RATIO_CUTS[1:]:
        print(f"  events with ratio > {c:.3f}: {(ratio > c).sum()}")
    print(f"T_coll: min={t_coll.min():.2f} max={t_coll.max():.2f} Gyr")
    print(f"TSC_full: min={tsc_full.min():.2f} med={np.median(tsc_full):.2f} "
          f"max={tsc_full.max():.2f} Gyr  (assumed collision->coalescence "
          f"duration)")

    halo_id, tsc = load_truth()
    ref_t0 = np.nanmax(t_coll + tsc_full)
    print(f"\nreference z=0 time (max T_coll+TSC_full) = {ref_t0:.3f} Gyr")
    # HaloID in the event catalog is a row index into the target-halo list,
    # not a FoF ID: it runs 0..N-1 while halo_id holds true FoF IDs.
    print(f"finite tsc_gyr: {np.isfinite(tsc).sum()}/{len(tsc)}  "
          f"range [{np.nanmin(tsc):.3f}, {np.nanmax(tsc[np.isfinite(tsc)]):.3f}]")
    print(f"event HaloID range: {ev_halo.min()}-{ev_halo.max()}, "
          f"{len(np.unique(ev_halo))} unique (target-list indices)")

    t99 = tab[SNAP0]
    print(f"\n{'cut':>8} {'clock':>8} {'t0':>7} {'n':>5} {'med TSC':>9} "
          f"{'rho':>7} {'med|dt|':>9} {'frac<0.05':>10}")
    best = None
    for c in RATIO_CUTS:
        for clock_name, use_full in (("T_coll", False), ("full", True)):
            for t0_name, t0 in (("t99", t99), ("maxev", ref_t0)):
                proxy = np.full(len(halo_id), np.nan)
                for i in range(len(halo_id)):
                    m = (ev_halo == i) & (ratio > c)
                    if not m.any():
                        continue
                    tt = t_coll[m] + (tsc_full[m] if use_full else 0.0)
                    tt = tt[tt <= t0]
                    if len(tt):
                        proxy[i] = t0 - tt.max()
                ok = np.isfinite(proxy) & np.isfinite(tsc)
                if ok.sum() < 5:
                    print(f"{c:>8.3f} {clock_name:>8} {t0_name:>7} "
                          f"{ok.sum():>5}  (too few matches)")
                    continue
                rho = stats.spearmanr(proxy[ok], tsc[ok]).statistic
                d = np.abs(proxy[ok] - tsc[ok])
                frac = (d < 0.05).mean()
                print(f"{c:>8.3f} {clock_name:>8} {t0_name:>7} {ok.sum():>5} "
                      f"{np.median(proxy[ok]):>9.2f} {rho:>7.3f} "
                      f"{np.median(d):>9.3f} {frac:>10.3f}")
                if best is None or frac > best[0]:
                    best = (frac, c, f"{clock_name}/{t0_name}", rho)
    if best is None:
        print("\nno ratio cut reproduced the label -- check ID matching")
        return
    print(f"\nbest reproduction of the training label: ratio cut "
          f"{best[1]:.3f}, clock {best[2]} "
          f"(exact-match frac {best[0]:.3f}, rho {best[3]:.3f})")

    # characterise the events that actually set the label
    print("\n--- properties of the label-setting (most recent) event ---")
    last_ratio = np.full(len(halo_id), np.nan)
    for i in range(len(halo_id)):
        m = np.where(ev_halo == i)[0]
        if not len(m):
            continue
        last_ratio[i] = ratio[m[np.argmax(t_coll[m])]]
    lr = last_ratio[np.isfinite(last_ratio)]
    print(f"n={len(lr)}  mass ratio of the merger that sets TSC: "
          f"med={np.median(lr):.3f} "
          f"IQR=[{np.percentile(lr, 25):.3f}, {np.percentile(lr, 75):.3f}]")
    for c in (0.05, 0.10, 0.20, 1.0 / 3.0):
        print(f"  fraction with ratio > {c:.2f}: {(lr > c).mean():.3f}")


def _wrap(d, box):
    return d - box * np.round(d / box)


NTOP = 3


def read_catalog(snap):
    """Load the group/subhalo tables for one snapshot.

    Prefers the single-file target-halo catalogue; falls back to the raw
    chunked fof_subhalo_tab files, which cover snapshots the single-file
    version is missing. In the chunked form SubhaloGrNr is a global group
    index, so chunks concatenate directly.
    """
    single = (f"{TARGET_CAT}/targethalo_cat_{snap:03d}/"
              f"TargetHalo_MergerCat_{snap:03d}.hdf5")
    if os.path.exists(single):
        with h5py.File(single, "r") as f:
            return (f["Group/GroupPos"][:], f["Group/Group_M_Crit200"][:],
                    f["Group/Group_R_Crit200"][:], f["Group/GroupNsubs"][:],
                    f["Group/FOF_Halo_IDs"][:].astype(np.int64),
                    f["Subhalo/SubhaloMass"][:],
                    f["Subhalo/SubhaloGrNr"][:].astype(np.int64),
                    f["Subhalo/SubhaloPos"][:], f["Subhalo/SubhaloFlag"][:])

    d = f"{GROUPCAT}/groupcat_{snap:03d}"
    chunks = sorted(os.listdir(d),
                    key=lambda x: int(x.split(".")[-2]))
    gp, gm, gr, gn = [], [], [], []
    sm, sg, sp, sf = [], [], [], []
    for c in chunks:
        with h5py.File(os.path.join(d, c), "r") as f:
            if "Group" in f and "GroupPos" in f["Group"]:
                gp.append(f["Group/GroupPos"][:])
                gm.append(f["Group/Group_M_Crit200"][:])
                gr.append(f["Group/Group_R_Crit200"][:])
                gn.append(f["Group/GroupNsubs"][:])
            if "Subhalo" in f and "SubhaloMass" in f["Subhalo"]:
                sm.append(f["Subhalo/SubhaloMass"][:])
                sg.append(f["Subhalo/SubhaloGrNr"][:].astype(np.int64))
                sp.append(f["Subhalo/SubhaloPos"][:])
                sf.append(f["Subhalo/SubhaloFlag"][:]
                          if "SubhaloFlag" in f["Subhalo"]
                          else np.ones(len(f["Subhalo/SubhaloMass"]), np.int8))
    gpos = np.concatenate(gp)
    return (gpos, np.concatenate(gm), np.concatenate(gr), np.concatenate(gn),
            np.arange(len(gpos), dtype=np.int64), np.concatenate(sm),
            np.concatenate(sg), np.concatenate(sp), np.concatenate(sf))


def snapshot_features(snap, targets_pos, box, match_kpc):
    """Per-target companion census at one snapshot.

    The truth catalog's "collision" is a pericentre passage of a subcluster,
    so presence of a companion is not enough -- we record masses *and*
    separations (in units of R200) for the NTOP most massive companion
    subhaloes of the primary FoF group, plus the most massive neighbouring
    FoF group that has not yet been linked in.
    """
    (gpos, gm200, gr200, nsubs, gid,
     smass, sgrnr, spos, sflag) = read_catalog(snap)

    n = len(targets_pos)
    out = dict(
        mu=np.full((n, NTOP), np.nan),      # companion/primary subhalo mass
        sep=np.full((n, NTOP), np.nan),     # separation / R200
        m200=np.full(n, np.nan),
        r200=np.full(n, np.nan),
        nsub=np.full(n, np.nan),            # companions with mu>0.01 in R200
        nb_mu=np.full(n, np.nan),           # nearest massive unlinked FoF
        nb_sep=np.full(n, np.nan),
    )

    # SubhaloGrNr indexes global FoF groups, which need not equal the row
    # index of this catalog if it stores a subset
    row_of_gid = np.full(int(max(gid.max(), sgrnr.max())) + 2, -1, np.int64)
    row_of_gid[gid] = np.arange(len(gid))
    grow = row_of_gid[sgrnr]

    order = np.argsort(grow, kind="stable")
    grow_s = grow[order]
    bounds = np.searchsorted(grow_s, np.arange(len(gpos) + 1))

    for i, p in enumerate(targets_pos):
        d = np.linalg.norm(_wrap(gpos - p, box), axis=1)
        near = np.where(d < match_kpc)[0]
        if not len(near):
            continue
        g = near[np.argmax(gm200[near])]
        r200 = gr200[g] if gr200[g] > 0 else np.nan
        out["m200"][i] = gm200[g]
        out["r200"][i] = r200
        if nsubs[g] < 1:
            continue

        idx = order[bounds[g]:bounds[g + 1]]
        idx = idx[sflag[idx] > 0]
        if len(idx) < 1:
            continue
        ms = smass[idx]
        rank = np.argsort(ms)[::-1]
        prim = idx[rank[0]]
        mprim = ms[rank[0]]
        comp = idx[rank[1:]]
        if len(comp):
            mc = smass[comp]
            dc = np.linalg.norm(_wrap(spos[comp] - spos[prim], box), axis=1)
            ratios = mc / mprim
            out["nsub"][i] = np.count_nonzero((ratios > 0.01) & (dc < r200))
            for k in range(min(NTOP, len(comp))):
                out["mu"][i, k] = ratios[k]
                out["sep"][i, k] = dc[k] / r200

        # most massive *other* FoF group within match_kpc (pre-infall pair)
        other = near[near != g]
        if len(other):
            j = other[np.argmax(gm200[other])]
            out["nb_mu"][i] = gm200[j] / gm200[g] if gm200[g] > 0 else np.nan
            out["nb_sep"][i] = d[j] / r200
    return out


PARTS = "merger_proxy_parts"


def available_snaps(min_snap):
    avail = set()
    for d in os.listdir(TARGET_CAT):
        if d.startswith("targethalo_cat_"):
            s = int(d.split("_")[-1])
            if os.path.exists(f"{TARGET_CAT}/{d}/"
                              f"TargetHalo_MergerCat_{s:03d}.hdf5"):
                avail.add(s)
    for d in os.listdir(GROUPCAT):
        if d.startswith("groupcat_") and d.split("_")[-1].isdigit():
            if os.listdir(f"{GROUPCAT}/{d}"):
                avail.add(int(d.split("_")[-1]))
    return sorted(s for s in avail if s >= min_snap)


def snap_times(snaps):
    tab = snap_time_table()
    missing = [s for s in snaps if s not in tab]
    if missing:
        ss = np.array(sorted(tab))
        tt = np.array([tab[s] for s in ss])
        for s in missing:
            tab[s] = float(np.interp(s, ss, tt))
    return np.array([tab[s] for s in snaps])


def load_targets():
    """Positions and masses of the 352 target halos at z=0."""
    halo_id, tsc = load_truth()
    path99 = (f"{TARGET_CAT}/targethalo_cat_{SNAP0:03d}/"
              f"TargetHalo_MergerCat_{SNAP0:03d}.hdf5")
    with h5py.File(path99, "r") as f:
        fof = f["Group/FOF_Halo_IDs"][:].astype(np.int64)
        gpos99 = f["Group/GroupPos"][:]
        gm99 = f["Group/Group_M_Crit200"][:]
    pos_of = {int(h): gpos99[j] for j, h in enumerate(fof)}
    m_of = {int(h): gm99[j] for j, h in enumerate(fof)}

    keep = np.array([int(h) in pos_of for h in halo_id])
    halo_id, tsc = halo_id[keep], tsc[keep]
    targets_pos = np.array([pos_of[int(h)] for h in halo_id])
    m200_99 = np.array([m_of[int(h)] for h in halo_id])
    return halo_id, tsc, targets_pos, m200_99


def stage_one_snapshot(args):
    """Compute and cache the companion census for a single snapshot.

    Reading the chunked catalogues is slow enough that snapshots are run as
    independent array tasks rather than one serial pass.
    """
    os.makedirs(PARTS, exist_ok=True)
    halo_id, tsc, targets_pos, _ = load_targets()
    s = args.snap
    out = f"{PARTS}/snap_{s:03d}.npz"
    if os.path.exists(out) and not args.force:
        print(f"snap {s}: already done -> {out}")
        return
    f = snapshot_features(s, targets_pos, args.box, args.match_kpc)
    np.savez(out, halo_id=halo_id, **f)
    mu1 = f["mu"][:, 0]
    print(f"snap {s}: matched {np.isfinite(f['m200']).sum()}/{len(halo_id)}  "
          f"med mu1={np.nanmedian(mu1):.4f}  "
          f"med sep1={np.nanmedian(f['sep'][:, 0]):.2f} R200  "
          f"frac mu1>1/3: {np.nanmean(mu1 > 1/3):.3f}")
    print(f"saved -> {out}")


def stage_combine(args):
    halo_id, tsc, _, m200_99 = load_targets()
    snaps = [s for s in available_snaps(args.min_snap)
             if os.path.exists(f"{PARTS}/snap_{s:03d}.npz")]
    print(f"combining {len(snaps)} snapshots: {snaps[0]}-{snaps[-1]}")

    n, ns = len(halo_id), len(snaps)
    hist = dict(mu=np.full((n, ns, NTOP), np.nan),
                sep=np.full((n, ns, NTOP), np.nan),
                m200=np.full((n, ns), np.nan),
                r200=np.full((n, ns), np.nan),
                nsub=np.full((n, ns), np.nan),
                nb_mu=np.full((n, ns), np.nan),
                nb_sep=np.full((n, ns), np.nan))
    for k, s in enumerate(snaps):
        d = np.load(f"{PARTS}/snap_{s:03d}.npz")
        if not np.array_equal(d["halo_id"], halo_id):
            raise SystemExit(f"snap {s}: halo ordering differs")
        for key in hist:
            hist[key][:, k] = d[key]

    t_of_snap = snap_times(snaps)
    t0 = t_of_snap[np.array(snaps) == SNAP0][0]
    span = t0 - t_of_snap.min()
    print(f"catalog spans {span:.2f} Gyr (snap {snaps[0]} -> {SNAP0}); "
          f"the proxy is censored above this")

    np.savez("merger_proxy_validation.npz", halo_id=halo_id, tsc_truth=tsc,
             snaps=np.array(snaps), t_of_snap=t_of_snap, t0=t0, span=span,
             m200_99=m200_99, **hist)
    print("saved -> merger_proxy_validation.npz")
    search_definitions(hist, tsc, t_of_snap, t0, span)


def search_definitions(hist, tsc, t_of_snap, t0, span):
    """Grid-search proxy merger definitions against the truth TSC.

    A collision in the truth catalog is a pericentre passage, so the
    candidate rule is "a companion above mass ratio `thr` was inside
    `rad` x R200" -- the last snapshot satisfying it sets the proxy TSC.
    """
    mu = hist["mu"]
    sep = hist["sep"]
    n, ns, _ = mu.shape
    recent = tsc <= 2.0
    print(f"\n{'thr':>7} {'rad':>6} {'n uncens':>9} {'med proxy':>10} "
          f"{'rho':>7} {'rho(unc)':>9} {'AUC<2Gyr':>9}")
    best = None
    for thr in (0.02, 0.05, 0.10, 0.20, 1.0 / 3.0):
        for rad in (0.5, 1.0, 2.0, np.inf):
            hit = (mu > thr) & (sep < rad)          # (n, ns, NTOP)
            any_hit = hit.any(axis=2)               # (n, ns)
            proxy = np.full(n, np.nan)
            for i in range(n):
                w = np.where(any_hit[i])[0]
                if len(w):
                    proxy[i] = t0 - t_of_snap[w.max()]
            uncens = np.isfinite(proxy)
            filled = np.where(uncens, proxy, span)
            ok = np.isfinite(tsc)
            rho = stats.spearmanr(filled[ok], tsc[ok]).statistic
            rho_u = (stats.spearmanr(proxy[uncens & ok],
                                     tsc[uncens & ok]).statistic
                     if (uncens & ok).sum() > 10 else np.nan)
            r = recent & ok
            nr = (~recent) & ok
            auc = (stats.mannwhitneyu(-filled[r], -filled[nr]).statistic
                   / (r.sum() * nr.sum())) if r.sum() and nr.sum() else np.nan
            rlab = "inf" if not np.isfinite(rad) else f"{rad:.1f}"
            print(f"{thr:>7.3f} {rlab:>6} {uncens.sum():>9} "
                  f"{np.nanmedian(proxy):>10.2f} {rho:>7.3f} "
                  f"{rho_u:>9.3f} {auc:>9.3f}")
            if best is None or (np.isfinite(rho) and rho > best[0]):
                best = (rho, thr, rlab, auc)
    print(f"\nbest proxy definition: mass ratio > {best[1]:.3f} within "
          f"{best[2]} R200  (rho={best[0]:.3f} vs truth TSC, "
          f"AUC for TSC<2Gyr = {best[3]:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["truth", "snapshot", "combine",
                                        "list"], default="truth")
    ap.add_argument("--snap", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--box", type=float, default=1003800.0,
                    help="box side in ckpc/h for periodic wrapping")
    ap.add_argument("--min-snap", type=int, default=71,
                    help="earliest snapshot to include in the proxy history")
    ap.add_argument("--match-kpc", type=float, default=3000.0,
                    help="search radius when identifying the primary FoF "
                         "group at earlier snapshots")
    args = ap.parse_args()
    if args.stage == "truth":
        stage_truth()
    elif args.stage == "list":
        print(" ".join(str(s) for s in available_snaps(args.min_snap)))
    elif args.stage == "snapshot":
        stage_one_snapshot(args)
    else:
        stage_combine(args)
