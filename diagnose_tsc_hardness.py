#!/usr/bin/env python3
"""Phase 4 Tier 0 (review rounds 32-33): why is the capped catalogue TSC harder
to regress than pseudo-TSC? Diagnostics on SAVED labels and predictions only.

  diagnose_tsc_hardness.py --expect-manifest <sha256> --out tsc_hardness_diag.npz

Inputs: the four accepted label-experiment arms lbl_{x,r}_{pseudo,tsc_tcoll}
(validated with compare_labels.load/check/check_cross), the stored merger
curves (dataset_nh4_144_s9991_128.h5), Chuiyang's TSC_new catalogue
(cluster_mergers.hdf5, TNG-Cluster_Catalog.hdf5), and group-catalogue headers
for snapshot redshifts (snapshots 71-99 only; earlier ages unavailable).

A0  stored rule: pseudo recomputed from the stored curves (build_dataset.py
    rule; 'all' at 99, 'pre' at 91) must equal the saved truth within 1e-5
    Gyr (rtol 0); classes exact, with a final-grid crossing distinct from no
    crossing. Catalogue reconstruction of S(tau) = sum exp(-dt/tau) in two
    named variants -- T_coll (primary) and Snap_coll ages (sensitivity; full
    clock convention, past mask by snapshot) -- compared with the stored
    curves near the crossing; if unmatched it is a "catalogue-defined activity
    score", not a reproduction. The most recent past T_coll is verified
    against the saved raw catalogue truth before any event attribute is used.
A1  error anatomy on snapshot-99 single observations (9 per parent), four
    arms vs the capped T_coll truth, in frozen strata (a = 4 ln 2): TSC
    [0,0.5] (0.5,1.5] (1.5,a] (a,4] finite>4 missing; stored pseudo class;
    n_2Gyr {0,1,>=2}; setting-event mass ratio (same event row; tie rule:
    largest finite nonnegative ratio, then lowest row; all-invalid -> lowest
    row, invalid category). Bias = prediction - truth. Paired parent
    bootstrap (within stratum, 4000 draws, seed recorded) for the MSE
    contrast tsc_tcoll - pseudo; no interval below two parents.
A2  MSE and Var(truth) beside R2; the exact identity
    mean((p_hat - t)^2) = mean((p_hat - p)^2) + mean((p - t)^2)
                          + 2 mean((p_hat - p)(p - t))
    for the pseudo arm, per modality and TSC stratum; upper clipping
    t_a = min(t_capped, a) vs min(t_hat, a) for the catalogue arm, beside the
    original; affine-invariance check.
A3  held-out label-to-label reference (not a ceiling): isotonic maps in both
    directions fitted per outer fold on snapshot-99 FITTING parents (outer
    fold and its inner parents excluded), one row per parent, endpoint
    clipping counted; label-map transfer of the pseudo arm (seeds averaged
    first, then mapped) on the single-observation estimand.
Nomination (X-ray only; practical prioritisation, not a test): range arm if
the (a,4] + finite>4 + missing strata hold >= 50% of the catalogue arm's SSE
AND its combined MSE there exceeds the pseudo arm's; else loss arm if the
catalogue arm's bias is > +0.3 Gyr in [0,0.5] and < -0.3 Gyr in finite>4;
else none. A4 is deferred and absent.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess

import h5py
import numpy as np
from astropy.cosmology import Planck15
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

import compare_labels as L

A_SAT = 4.0 * np.log(2.0)
TOL_PSEUDO = 1e-5
TOL_TSC = 1e-6
Z_SNAP = {99: 0.0, 91: 0.0994018026302219}
TSC_STRATA = ["[0,0.5]", "(0.5,1.5]", "(1.5,a]", "(a,4]", "finite>4", "missing"]
N2_STRATA = ["n2=0", "n2=1", "n2>=2"]
RANGE_STRATA = ("(a,4]", "finite>4", "missing")
C = "/oscar/data/idellant/Chuiyang"
DEFAULTS = dict(curves="dataset_nh4_144_s9991_128.h5", mergers=f"{C}/TSC_new/cluster_mergers.hdf5",
                catalog=f"{C}/TSC_new/TNG-Cluster_Catalog.hdf5",
                groupcat=f"{C}/groupcat_classification")


def fail(msg):
    raise SystemExit(msg)


# ---------------------------------------------------------------- labels
def crossing_class(curve):
    """Stored-rule classification of one curve: ('floor'|'interior'|'final'|
    'none', first crossing index or -1). A crossing at the final grid point
    is 'final', distinct from no crossing although both give pseudo = 4."""
    c = np.asarray(curve, dtype=np.float64) >= 0.5
    if not c.any():
        return "none", -1
    i = int(np.argmax(c))
    if i == 0:
        return "floor", 0
    return ("final" if i == len(c) - 1 else "interior"), i


def pseudo_from_curve(curve, tau):
    """build_dataset.py's rule in float64."""
    curve, tau = np.asarray(curve, np.float64), np.asarray(tau, np.float64)
    cls, i = crossing_class(curve)
    if cls == "none":
        return float(tau[-1])
    if cls == "floor":
        return float(tau[0])
    t0, t1, s0, s1 = tau[i - 1], tau[i], curve[i - 1], curve[i]
    return float(t0 + (0.5 - s0) * (t1 - t0) / (s1 - s0))


def tsc_stratum(raw):
    if not np.isfinite(raw):
        return "missing"
    if raw <= 0.5:
        return "[0,0.5]"
    if raw <= 1.5:
        return "(0.5,1.5]"
    if raw <= A_SAT:
        return "(1.5,a]"
    if raw <= 4.0:
        return "(a,4]"
    return "finite>4"


def n2_stratum(n):
    return "n2=0" if n == 0 else "n2=1" if n == 1 else "n2>=2"


# ---------------------------------------------------------------- catalogue
def load_events(mergers, catalog):
    """Event arrays with validated row -> parent mapping (HaloID is a row
    index into the catalogue's haloID list). Parents without events are
    legitimate and simply have no rows."""
    with h5py.File(catalog, "r") as f:
        targets = f["haloID"][:]
    if len(np.unique(targets)) != len(targets):
        fail("duplicate parent ids in the TNG-Cluster catalogue")
    with h5py.File(mergers, "r") as f:
        ev = {k: f[k][:] for k in ("HaloID", "T_coll", "Snap_coll", "Mass_ratio")}
    hf = ev["HaloID"].astype(np.float64)
    if not np.isfinite(hf).all() or not np.array_equal(hf, np.round(hf)):
        fail("event HaloID must be finite integers")
    hi = hf.astype(np.int64)
    if hi.min() < 0 or hi.max() >= len(targets):
        fail(f"event HaloID outside [0, {len(targets) - 1}]")
    if not np.isfinite(ev["T_coll"]).all():
        fail("non-finite T_coll")
    ev["parent"] = targets[hi]
    ev["row"] = np.arange(len(hi))
    return ev


def setting_event(rows, tcoll, mratio):
    """Most recent past event by T_coll; tie rule (review R33): largest finite
    nonnegative Mass_ratio, then lowest row; all tied ratios invalid -> lowest
    row with the invalid category. Returns (row or -1, tied count, ratio or
    NaN, valid flag). All attributes come from the same event row."""
    if len(rows) == 0:
        return -1, 0, np.nan, False
    t = tcoll[rows]
    tied = rows[t == t.max()]
    r = mratio[tied]
    ok = np.isfinite(r) & (r >= 0)
    if ok.any():
        best = tied[ok][r[ok] == r[ok].max()]
        sel = int(best.min())
        return sel, len(tied), float(mratio[sel]), True
    return int(tied.min()), len(tied), float(mratio[int(tied.min())]), False


def verify_recent(t_now, tcoll_past, raw, tag):
    """Most recent past T_coll must reproduce the saved raw catalogue truth
    (NaN <-> no past event) before any event attribute is used."""
    rec = t_now - np.max(tcoll_past) if len(tcoll_past) else np.nan
    if np.isnan(raw) != np.isnan(rec) or (np.isfinite(raw) and abs(raw - rec) > TOL_TSC):
        fail(f"{tag}: catalogue most-recent T_coll {rec} != saved raw truth {raw}")
    return rec


def snapshot_table(groupcat):
    """Snapshot -> (redshift, Planck15 age) from group-catalogue headers;
    only snapshots with an actual header are listed."""
    tab = {}
    for d in sorted(glob.glob(os.path.join(groupcat, "groupcat_0??"))):
        sn = int(d[-3:])
        f0 = os.path.join(d, f"fof_subhalo_tab_{sn:03d}.0.hdf5")
        if os.path.exists(f0):
            with h5py.File(f0, "r") as f:
                z = float(f["Header"].attrs["Redshift"])
            tab[sn] = (z, float(Planck15.age(z).value), f0)
    return tab


def reconstruct(t_now, dt_past, tau):
    """S(tau) = sum exp(-dt/tau) over past events."""
    if len(dt_past) == 0:
        return np.zeros(len(tau))
    return np.exp(-np.asarray(dt_past)[None, :] / np.asarray(tau)[:, None]).sum(1)


# ---------------------------------------------------------------- scoring
def fitting_parents(parents_fold, inner_f, f):
    """Snapshot-99 parents fitted for outer fold f: outer fold != f and not in
    f's inner-selection set."""
    return sorted(p for p, fo in parents_fold.items() if fo != f and p not in inner_f)


def mse_identity(p_hat, p, t):
    """Exact accounting identity terms; returns (lhs, term1, term2, cross)."""
    p_hat, p, t = map(np.asarray, (p_hat, p, t))
    lhs = np.mean((p_hat - t) ** 2)
    a, b, c = np.mean((p_hat - p) ** 2), np.mean((p - t) ** 2), 2 * np.mean((p_hat - p) * (p - t))
    return lhs, a, b, c


def nominate(comp):
    """X-ray nomination rule (review R33). comp: dict stratum -> dict with
    sse_cat, mse_cat, mse_pse, n_obs, bias_cat; plus 'sse_cat_total'."""
    sse_r = sum(comp[s]["sse_cat"] for s in RANGE_STRATA if s in comp)
    n_r = sum(comp[s]["n_obs"] for s in RANGE_STRATA if s in comp)
    share = sse_r / comp["sse_cat_total"]
    mse_cat_r = sse_r / n_r if n_r else np.nan
    mse_pse_r = (sum(comp[s]["mse_pse"] * comp[s]["n_obs"] for s in RANGE_STRATA if s in comp) / n_r
                 if n_r else np.nan)
    if share >= 0.50 and mse_cat_r > mse_pse_r:
        return "range (one lower-cap X-ray arm, min(T, a))", share, mse_cat_r, mse_pse_r
    b_young = comp.get("[0,0.5]", {}).get("bias_cat", np.nan)
    b_old = comp.get("finite>4", {}).get("bias_cat", np.nan)
    if b_young > 0.3 and b_old < -0.3:
        return "loss (one MSE catalogue-TSC X-ray arm)", share, mse_cat_r, mse_pse_r
    return "none", share, mse_cat_r, mse_pse_r


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-manifest", required=True)
    ap.add_argument("--pred-dir", default="cnn_preds")
    ap.add_argument("--out", default="tsc_hardness_diag.npz")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--boot-seed", type=int, default=20261106)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v)
    a = ap.parse_args()
    if os.path.exists(a.out):
        fail(f"refusing to overwrite {a.out}")
    here = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    prov = dict(head=head, sources={f: sha(os.path.join(here, f)) for f in
                                    ("diagnose_tsc_hardness.py", "compare_labels.py", "compare_e0.py")},
                inputs={k: sha(getattr(a, k)) for k in ("curves", "mergers", "catalog")},
                thresholds=dict(a=A_SAT, tol_pseudo=TOL_PSEUDO, tol_tsc=TOL_TSC, range_share=0.5,
                                bias=0.3, boot=a.boot, boot_seed=a.boot_seed))

    # ---- predictions: accepted validators
    seeds = L.PRODUCTION_SEEDS
    runs, refs = {}, {}
    for m in L.MODALITIES:
        runs[m] = L.load(a.pred_dir, m, seeds)
        refs[m] = L.check(runs[m], m, a.expect_manifest, 60)
    L.check_cross(refs)
    ref = refs["x"]
    gid, snp = ref["group_id"].astype(int), ref["snap"].astype(int)
    rows = list(zip(gid.tolist(), snp.tolist()))
    print(f"validated: 4 arms x {len(seeds)} seeds, {len(rows)} label rows")

    # ---- A0 stored rule
    with h5py.File(a.curves, "r") as f:
        tau = f["labels/tau_gyr"][:].astype(np.float64)
        s_all, s_pre = f["labels/label_score_all"][:], f["labels/label_score_pre"][:]
        mk = list(zip(f["meta/group_id"][:].astype(int).tolist(), f["meta/snap"][:].astype(int).tolist()))
    if len(set(mk)) != len(mk):
        fail("duplicate (parent, snapshot) rows in the curve file")
    pos = dict(zip(mk, range(len(mk))))
    miss = [r for r in rows if r not in pos]
    if miss:
        fail(f"{len(miss)} label rows missing from the curve file")
    cls_st, idx_st, ps_rec = [], [], []
    curves = []
    for (g, s) in rows:
        cv = (s_all if s == 99 else s_pre)[pos[(g, s)]]
        curves.append(cv.astype(np.float64))
        c_, i_ = crossing_class(cv)
        cls_st.append(c_); idx_st.append(i_)
        ps_rec.append(pseudo_from_curve(cv, tau))
    ps_rec = np.array(ps_rec)
    dmax = np.max(np.abs(ps_rec - ref["truth_pseudo"]))
    if dmax > TOL_PSEUDO:
        fail(f"stored-curve pseudo does not reproduce the saved truth (max |diff| {dmax:.3g})")
    cls_st = np.array(cls_st)
    print(f"\n== A0 stored rule: pseudo reproduced, max |diff| {dmax:.2e} Gyr (tol {TOL_PSEUDO}); classes: "
          + ", ".join(f"{c} {int((cls_st == c).sum())}" for c in ("floor", "interior", "final", "none")))

    # ---- A0 catalogue
    ev = load_events(a.mergers, a.catalog)
    by_parent = {}
    for r_, p_ in zip(ev["row"], ev["parent"]):
        by_parent.setdefault(int(p_), []).append(int(r_))
    tab = snapshot_table(a.groupcat)
    prov["snapshot_table"] = {str(k): v[:2] for k, v in tab.items()}
    n2, srow, stied, sratio, svalid = [], [], [], [], []
    rec = {"tcoll": [], "snap": []}
    avail_snap = []
    memb_change = 0
    for k_, (g, s) in enumerate(rows):
        t_now = float(Planck15.age(Z_SNAP[s]).value)
        evr = np.array(by_parent.get(g, []), dtype=int)
        past = evr[ev["T_coll"][evr] <= t_now + 1e-9] if len(evr) else evr
        verify_recent(t_now, ev["T_coll"][past], ref["truth_tcoll_raw"][k_], f"(parent {g}, snap {s})")
        dt = t_now - ev["T_coll"][past]
        n2.append(int(((dt >= 0) & (dt <= 2.0)).sum()))
        se = setting_event(past, ev["T_coll"], ev["Mass_ratio"])
        srow.append(se[0]); stied.append(se[1]); sratio.append(se[2]); svalid.append(se[3])
        rec["tcoll"].append(reconstruct(t_now, dt, tau))
        # Snap_coll variant: full clock convention (past mask by snapshot, age of Snap_coll)
        pv = evr[ev["Snap_coll"][evr] <= s] if len(evr) else evr
        if all(int(x) in tab for x in ev["Snap_coll"][pv]):
            ages = np.array([tab[int(x)][1] for x in ev["Snap_coll"][pv]])
            rec["snap"].append(reconstruct(t_now, t_now - ages, tau)); avail_snap.append(True)
        else:
            rec["snap"].append(np.full(len(tau), np.nan)); avail_snap.append(False)
        memb_change += int(set(pv.tolist()) != set(past.tolist()))
    n2, srow, sratio, svalid = map(np.array, (n2, srow, sratio, svalid))
    avail_snap = np.array(avail_snap)
    print(f"== A0 catalogue: most-recent past T_coll matches the saved raw truth for all {len(rows)} rows "
          f"(tol {TOL_TSC}); parents with no past event: {int((srow < 0).sum())}; snapshot table covers "
          f"snapshots {min(tab)}-{max(tab)}")
    a0 = {}
    for var in ("tcoll", "snap"):
        S = np.array(rec[var])
        for sn in (99, 91):
            msk = (snp == sn) & (np.isfinite(S).all(1))
            cls_r = np.array([crossing_class(x)[0] for x in S])
            ps_r = np.array([pseudo_from_curve(x, tau) if np.isfinite(x).all() else np.nan for x in S])
            near = []
            for k_ in np.where(msk)[0]:
                j = idx_st[k_] if idx_st[k_] >= 0 else len(tau) - 1
                js = [x for x in (j - 1, j, j + 1) if 0 <= x < len(tau)]
                near.append(np.max(np.abs(S[k_, js] - curves[k_][js])))
            near = np.array(near)
            dps = np.abs(ps_r[msk] - ref["truth_pseudo"][msk])
            chg = (cls_r[msk] != cls_st[msk])
            n_tot = int((snp == sn).sum())
            print(f"   variant {var:<5} snap {sn}: available {int(msk.sum())}/{n_tot}; near-crossing |dS| median "
                  f"{np.median(near) if len(near) else np.nan:.3g}, 95% {np.percentile(near, 95) if len(near) else np.nan:.3g}; "
                  f"|d pseudo| median {np.median(dps) if len(dps) else np.nan:.3g}, max {np.max(dps) if len(dps) else np.nan:.3g}; "
                  f"class changes {int(chg.sum())}; class change or |d pseudo| > 0.05: {int((chg | (dps > 0.05)).sum())}")
            a0[f"{var}_{sn}"] = dict(avail=int(msk.sum()), changed=int((chg | (dps > 0.05)).sum()))
    print(f"   Snap_coll past-event membership differs from T_coll for {memb_change} rows")
    matched = all(v["changed"] == 0 for k, v in a0.items() if k.startswith("tcoll"))
    print(f"   -> catalogue reconstruction {'reproduces' if matched else 'does NOT reproduce'} the stored curves "
          f"within the reported tolerance; {'' if matched else 'treat it as a catalogue-defined activity score'}")

    # ---- snapshot-99 parent strata
    m99 = snp == 99
    par99 = gid[m99]
    t_raw99, t_cap99, p99 = ref["truth_tcoll_raw"][m99], ref["truth_tcoll_capped"][m99], ref["truth_pseudo"][m99]
    st_tsc = np.array([tsc_stratum(x) for x in t_raw99])
    st_n2 = np.array([n2_stratum(x) for x in n2[m99]])
    vr = sratio[m99][svalid[m99]]
    q1, q2 = np.quantile(vr, [1 / 3, 2 / 3])
    st_mr = np.array(["missing" if r < 0 else "invalid" if not v else
                      ("mr<=q1" if x <= q1 else "q1<mr<=q2" if x <= q2 else "mr>q2")
                      for r, v, x in zip(srow[m99], svalid[m99], sratio[m99])])
    print(f"\n== A1 strata (snapshot 99, {len(par99)} parents); mass-ratio edges q1={q1:.4g}, q2={q2:.4g}; "
          f"tied setting events: {int((np.array(stied)[m99] > 1).sum())}")
    R, P = L.R_EXPECT, L.P_EXPECT
    rep = lambda v: np.repeat(np.repeat(v, R)[:, None], P, 1).ravel()     # parent -> 9 observations
    obs_par = rep(par99)
    y = rep(t_cap99)
    ens = {}
    for m in L.MODALITIES:
        ms = runs[m]["pseudo"][42]["sample_snap"] == 99
        for lab in L.LABELS:
            ens[(m, lab)] = np.mean([d["oof_single"][ms].ravel() for d in runs[m][lab].values()], 0)
    rng = np.random.default_rng(a.boot_seed)
    comp_x = {}
    out = dict(parent99=par99, st_tsc=st_tsc, st_n2=st_n2, st_mr=st_mr, setting_row=srow, setting_tied=np.array(stied),
               setting_ratio=sratio, n2=n2, cls_stored=cls_st, pseudo_rec=ps_rec, avail_snapvar=avail_snap)
    for m in L.MODALITIES:
        e_c, e_p = ens[(m, "tsc_tcoll")], ens[(m, "pseudo")]
        sse_c_tot = np.sum((e_c - y) ** 2)
        print(f"  -- {m}: stratum | parents obs | bias cat/pse | MAE cat/pse | MSE cat/pse | SSE share cat/pse | "
              f"dMSE cat-pse [95%]")
        for fam, st in (("TSC", st_tsc), ("pseudo class", cls_st[m99]), ("n_2Gyr", st_n2), ("mass ratio", st_mr)):
            for lev in (TSC_STRATA if fam == "TSC" else ["floor", "interior", "final", "none"] if fam == "pseudo class"
                        else N2_STRATA if fam == "n_2Gyr" else ["mr<=q1", "q1<mr<=q2", "mr>q2", "invalid", "missing"]):
                ps_ = par99[st == lev]
                if len(ps_) == 0:
                    print(f"     {fam}:{lev:<12} empty")
                    continue
                o = np.isin(obs_par, ps_)
                bc, bp = np.mean(e_c[o] - y[o]), np.mean(e_p[o] - y[o])
                mc, mp = np.mean((e_c[o] - y[o]) ** 2), np.mean((e_p[o] - y[o]) ** 2)
                ivs = ""
                if len(ps_) >= 2:
                    idx = {p_: np.where(obs_par == p_)[0] for p_ in ps_}
                    bs = []
                    for _ in range(a.boot):
                        s_ = np.concatenate([idx[p_] for p_ in rng.choice(ps_, len(ps_))])
                        bs.append(np.mean((e_c[s_] - y[s_]) ** 2) - np.mean((e_p[s_] - y[s_]) ** 2))
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    ivs = f"[{lo:+.3f}, {hi:+.3f}]"
                print(f"     {fam}:{lev:<12} {len(ps_):>4} {int(o.sum()):>5} | {bc:+.3f}/{bp:+.3f} | "
                      f"{np.mean(np.abs(e_c[o] - y[o])):.3f}/{np.mean(np.abs(e_p[o] - y[o])):.3f} | {mc:.3f}/{mp:.3f} | "
                      f"{np.sum((e_c[o] - y[o]) ** 2) / sse_c_tot:.3f}/{np.sum((e_p[o] - y[o]) ** 2) / np.sum((e_p - y) ** 2):.3f} | "
                      f"{mc - mp:+.3f} {ivs}")
                if m == "x" and fam == "TSC":
                    comp_x[lev] = dict(sse_cat=float(np.sum((e_c[o] - y[o]) ** 2)), mse_cat=mc, mse_pse=mp,
                                       n_obs=int(o.sum()), bias_cat=bc)
        if m == "x":
            comp_x["sse_cat_total"] = float(sse_c_tot)

    # ---- A2
    print("\n== A2 metric accounting (snapshot-99 single observations)")
    yp = rep(p99)
    for m in L.MODALITIES:
        for lab in L.LABELS:
            e = ens[(m, lab)]
            print(f"   {m}/{lab:<9}: vs T_coll R2 {r2_score(y, e):.3f} MSE {np.mean((e - y) ** 2):.3f} Var {np.var(y):.3f} | "
                  f"vs pseudo R2 {r2_score(yp, e):.3f} MSE {np.mean((e - yp) ** 2):.3f} Var {np.var(yp):.3f}")
        lhs, t1, t2, t3 = mse_identity(ens[(m, "pseudo")], yp, y)
        if not np.isclose(lhs, t1 + t2 + t3, rtol=0, atol=1e-10):
            fail("accounting identity failed")
        print(f"   {m} pseudo arm: MSE vs T {lhs:.3f} = model-vs-pseudo {t1:.3f} + pseudo-vs-T {t2:.3f} + 2 cross {t3:+.3f}")
        for lev in TSC_STRATA:
            o = np.isin(obs_par, par99[st_tsc == lev])
            if o.any():
                l_, a_, b_, c_ = mse_identity(ens[(m, "pseudo")][o], yp[o], y[o])
                print(f"      {lev:<10} {l_:.3f} = {a_:.3f} + {b_:.3f} + {c_:+.3f}")
        e = ens[(m, "tsc_tcoll")]
        ya, ea = np.minimum(y, A_SAT), np.minimum(e, A_SAT)
        print(f"   {m} catalogue arm, upper-clipped at a: R2 {r2_score(ya, ea):.3f} MSE {np.mean((ea - ya) ** 2):.3f} "
              f"(coarsened endpoint; original R2 {r2_score(y, e):.3f})")
        al, be = 2.5, -1.3
        if not np.isclose(r2_score(al * y + be, al * e + be), r2_score(y, e), rtol=0, atol=1e-12):
            fail("affine-invariance check failed")
    print("   affine invariance: R2 unchanged under identical affine maps (checked)")

    # ---- A3
    print("\n== A3 held-out label-to-label reference (isotonic, per outer fold, snapshot-99 fitting parents)")
    folds = {}
    for m in L.MODALITIES:
        for lab in L.LABELS:
            for d in runs[m][lab].values():
                pf = {int(g_): int(f_) for g_, f_ in zip(d["sample_group"], d["fold_of"])}
                inner = {f_: set(d["sel_groups_flat"][d["sel_groups_fold"] == f_].astype(int).tolist())
                         for f_ in range(L.N_FOLD)}
                if folds and (pf != folds["pf"] or inner != folds["inner"]):
                    fail(f"{m}/{lab}: fold memberships differ across arms/modalities")
                folds = dict(pf=pf, inner=inner)
    pf99 = {int(p_): folds["pf"][int(p_)] for p_ in par99}
    res = {"p2t": np.full(len(par99), np.nan), "t2p": np.full(len(par99), np.nan)}
    knots, clips, fitsets = {}, {"p2t": 0, "t2p": 0}, {}
    transfer = {m: np.full(len(y), np.nan) for m in L.MODALITIES}
    for f in range(L.N_FOLD):
        fit = fitting_parents(pf99, folds["inner"][f], f)
        held = [p_ for p_, fo in pf99.items() if fo == f]
        if set(fit) & set(held) or set(fit) & folds["inner"][f]:
            fail(f"fold {f}: fitting parents overlap held-out or inner parents")
        fitsets[f] = (np.array(fit), np.array(held))
        fi = np.isin(par99, fit); hi = np.isin(par99, held)
        for key_, xa, ya_ in (("p2t", p99, t_cap99), ("t2p", t_cap99, p99)):
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(xa[fi], ya_[fi])
            res[key_][hi] = iso.predict(xa[hi])
            clips[key_] += int(((xa[hi] < xa[fi].min()) | (xa[hi] > xa[fi].max())).sum())
            knots[(key_, f)] = (iso.X_thresholds_, iso.y_thresholds_)
            if key_ == "p2t":
                for m in L.MODALITIES:
                    oh = np.isin(obs_par, held)
                    transfer[m][oh] = iso.predict(ens[(m, "pseudo")][oh])
    print(f"   true pseudo -> capped T: R2 {r2_score(t_cap99, res['p2t']):.3f} (one row per parent; "
          f"endpoint clips {clips['p2t']})")
    print(f"   capped T -> true pseudo: R2 {r2_score(p99, res['t2p']):.3f} (endpoint clips {clips['t2p']})")
    for m in L.MODALITIES:
        print(f"   label-map transfer, {m} pseudo arm (seeds averaged, then mapped) vs T_coll: R2 "
              f"{r2_score(y, transfer[m]):.3f} (unmapped {r2_score(y, ens[(m, 'pseudo')]):.3f}; "
              f"catalogue arm {r2_score(y, ens[(m, 'tsc_tcoll')]):.3f})")

    # ---- nomination
    cand, share, mc_r, mp_r = nominate(comp_x)
    fin_old = comp_x.get("(a,4]", {}).get("sse_cat", 0) + comp_x.get("finite>4", {}).get("sse_cat", 0)
    print(f"\n== nomination (X-ray, practical prioritisation; not a test): range-strata SSE share {share:.3f} "
          f"(finite-old {fin_old / comp_x['sse_cat_total']:.3f}, missing "
          f"{comp_x.get('missing', {}).get('sse_cat', 0) / comp_x['sse_cat_total']:.3f}); combined MSE cat {mc_r:.3f} "
          f"vs pseudo {mp_r:.3f}; bias cat [0,0.5] {comp_x.get('[0,0.5]', {}).get('bias_cat', np.nan):+.3f}, "
          f"finite>4 {comp_x.get('finite>4', {}).get('bias_cat', np.nan):+.3f} -> candidate: {cand}")
    out.update(res_p2t=res["p2t"], res_t2p=res["t2p"],
               transfer_x=transfer["x"], transfer_r=transfer["r"],
               knots=np.array(json.dumps({f"{k}_{f}": [list(map(float, v[0])), list(map(float, v[1]))]
                                          for (k, f), v in knots.items()})),
               fitsets=np.array(json.dumps({str(f): [v[0].tolist(), v[1].tolist()] for f, v in fitsets.items()})),
               provenance=np.array(json.dumps(prov)), candidate=np.array(cand))
    np.savez(a.out, **out)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
