#!/usr/bin/env python3
"""Phase 4 Tier 0 (review rounds 32-34): why is the capped catalogue TSC harder
to regress than pseudo-TSC? Diagnostics on SAVED labels and predictions only.

  diagnose_tsc_hardness.py --expect-manifest <sha256> --out tsc_hardness_diag.npz

Inputs: the four accepted label-experiment arms lbl_{x,r}_{pseudo,tsc_tcoll}
(validated with compare_labels.load/check/check_cross, plus an exact
cross-modality row/sample ORDER check), the stored merger curves, Chuiyang's
TSC_new catalogue (cluster_mergers.hdf5, TNG-Cluster_Catalog.hdf5), and
group-catalogue headers for snapshot redshifts (optional; an empty or partial
table only makes the snapshot-clock sensitivity unavailable).

A0  stored rule: curves and tau grid validated (shape, finite, increasing
    0.1..4.0); pseudo recomputed from the stored curves (build_dataset.py
    rule) must equal the saved truth within 1e-5 Gyr (rtol 0), all finite;
    classes floor/interior/final/none (a final-grid crossing can give pseudo
    < 4 or exactly 4, and stays distinct from no crossing). Catalogue
    reconstruction S(tau) = sum exp(-dt/tau) in two named variants -- T_coll
    (primary) and Snap_coll ages (sensitivity) -- reported as (i) a
    material-difference flag (class change or |d pseudo| > 0.05 Gyr) and (ii)
    a separate full-curve agreement check (max |dS| <= 1e-3 on every tau,
    full coverage). Only (ii) may be called a reproduction; otherwise the
    result is a "catalogue-defined activity score". Past events: T_coll <=
    t_now + 1e-9 with elapsed time max(t_now - T_coll, 0), the same convention
    for verification, reconstruction and recent counts. The most recent past
    T_coll is verified against the saved raw catalogue truth first; a
    requested parent absent from the catalogue is rejected, a catalogue member
    without events is retained.
A1  error anatomy on snapshot-99 single observations (9 per parent), four
    arms vs the capped T_coll truth, in frozen strata (a = 4 ln 2), plus the
    TSC-band x stored-pseudo-class cross-tabulation. Bias = prediction -
    truth. Within-stratum paired parent bootstrap of the MSE contrast.
A2  MSE and Var(truth) beside R2; the exact identity (asserted overall and
    per TSC stratum); upper clipping; affine invariance (float64 path).
A3  isotonic label maps fitted per outer fold on snapshot-99 fitting parents;
    transfer of the pseudo arm (seeds averaged, then mapped); clip counts for
    label and prediction inputs; R2, MSE and Var reported.
Nomination (X-ray only; practical prioritisation, not a test), as agreed.
A4 is deferred and absent. All inputs, predictions (paths, hashes, original
provenance), arguments, the snapshot table (paths and header values) and every
derived per-row quantity are saved with (parent, snapshot) keys.
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
TOL_CURVE = 1e-3
EPS_PAST = 1e-9
Z_SNAP = {99: 0.0, 91: 0.0994018026302219}
TSC_STRATA = ["[0,0.5]", "(0.5,1.5]", "(1.5,a]", "(a,4]", "finite>4", "missing"]
CLASSES = ["floor", "interior", "final", "none"]
N2_STRATA = ["n2=0", "n2=1", "n2>=2"]
MR_STRATA = ["mr<=q1", "q1<mr<=q2", "mr>q2", "invalid", "missing"]
RANGE_STRATA = ("(a,4]", "finite>4", "missing")
C = "/oscar/data/idellant/Chuiyang"
DEFAULTS = dict(curves="dataset_nh4_144_s9991_128.h5", mergers=f"{C}/TSC_new/cluster_mergers.hdf5",
                catalog=f"{C}/TSC_new/TNG-Cluster_Catalog.hdf5",
                groupcat=f"{C}/groupcat_classification")


def fail(msg):
    raise SystemExit(msg)


def as_int(a, what):
    """Integer ids without silent truncation."""
    af = np.asarray(a, dtype=np.float64)
    if af.ndim != 1 or not np.isfinite(af).all() or not np.array_equal(af, np.round(af)):
        fail(f"{what} must be 1-D finite integers")
    return af.astype(np.int64)


# ---------------------------------------------------------------- labels
def crossing_class(curve):
    """('floor'|'interior'|'final'|'none', first crossing index or -1). A
    first crossing at the final grid point is 'final': it interpolates to <= 4
    (exactly 4 when the final score is exactly 0.5) and stays distinct from no
    crossing, which also gives the numeric value 4."""
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


def validate_curves(tau, s_all, s_pre):
    tau = np.asarray(tau, np.float64)
    if tau.ndim != 1 or len(tau) < 2 or not np.isfinite(tau).all() or not (np.diff(tau) > 0).all() \
            or tau[0] <= 0 or not np.isclose(tau[0], 0.1) or not np.isclose(tau[-1], 4.0):
        fail("tau grid must be 1-D, finite, positive, strictly increasing from 0.1 to 4.0")
    for nm, s in (("label_score_all", s_all), ("label_score_pre", s_pre)):
        if s.ndim != 2 or s.shape[1] != len(tau):
            fail(f"{nm} must be 2-D with {len(tau)} columns, got {s.shape}")
    if s_all.shape != s_pre.shape:
        fail("label_score_all and label_score_pre shapes differ")
    return tau


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
    """Validated events and the catalogue membership. HaloID is a row index
    into the catalogue's haloID list; members without events are legitimate."""
    with h5py.File(catalog, "r") as f:
        targets = as_int(f["haloID"][:], "catalogue haloID")
    if len(np.unique(targets)) != len(targets):
        fail("duplicate parent ids in the TNG-Cluster catalogue")
    with h5py.File(mergers, "r") as f:
        ev = {k: f[k][:] for k in ("HaloID", "T_coll", "Snap_coll", "Mass_ratio")}
    n = len(ev["HaloID"])
    for k, v in ev.items():
        if v.ndim != 1 or len(v) != n:
            fail(f"event array {k} must be 1-D with {n} entries")
    hi = as_int(ev["HaloID"], "event HaloID")
    if hi.min() < 0 or hi.max() >= len(targets):
        fail(f"event HaloID outside [0, {len(targets) - 1}]")
    ev["Snap_coll"] = as_int(ev["Snap_coll"], "event Snap_coll")
    t = ev["T_coll"].astype(np.float64)
    if not np.isfinite(t).all() or (t < 0).any():
        fail("event T_coll must be finite and nonnegative")
    ev["T_coll"] = t
    ev["Mass_ratio"] = ev["Mass_ratio"].astype(np.float64)
    ev["parent"] = targets[hi]
    ev["row"] = np.arange(n)
    by_parent = {}
    for r_, p_ in zip(ev["row"], ev["parent"]):
        by_parent.setdefault(int(p_), []).append(int(r_))
    return ev, set(targets.tolist()), by_parent


def events_for(g, members, by_parent):
    """Event rows of a requested parent; reject a parent not in the catalogue."""
    if g not in members:
        fail(f"requested parent {g} is not a catalogue member")
    return np.array(by_parent.get(g, []), dtype=int)


def past_events(rows, tcoll, t_now):
    """Primary convention: admitted if T_coll <= t_now + 1e-9; elapsed time
    clipped at zero. Returns (rows, elapsed)."""
    if len(rows) == 0:
        return rows, np.zeros(0)
    m = tcoll[rows] <= t_now + EPS_PAST
    return rows[m], np.maximum(t_now - tcoll[rows[m]], 0.0)


def setting_event(rows, tcoll, mratio):
    """Most recent past event by T_coll; tie rule (review R33): largest finite
    nonnegative Mass_ratio, then lowest row; all tied ratios invalid -> lowest
    row with the invalid category. Returns (row or -1, tied count, ratio or
    NaN, valid flag), all from the same event row."""
    if len(rows) == 0:
        return -1, 0, np.nan, False
    t = tcoll[rows]
    tied = rows[t == t.max()]
    r = mratio[tied]
    ok = np.isfinite(r) & (r >= 0)
    if ok.any():
        sel = int(tied[ok][r[ok] == r[ok].max()].min())
        return sel, len(tied), float(mratio[sel]), True
    sel = int(tied.min())
    return sel, len(tied), float(mratio[sel]), False


def verify_recent(dt_past, raw, tag):
    """Most recent past elapsed time must reproduce the saved raw truth (NaN
    <-> no past event) before any event attribute is used."""
    rec = float(np.min(dt_past)) if len(dt_past) else np.nan
    if np.isnan(raw) != np.isnan(rec) or (np.isfinite(raw) and abs(raw - rec) > TOL_TSC):
        fail(f"{tag}: catalogue most-recent elapsed time {rec} != saved raw truth {raw}")
    return rec


def snapshot_table(groupcat):
    """Snapshot -> dict(z, age, path) from group-catalogue headers present on
    disk; validated; possibly empty (the snapshot clock is then unavailable)."""
    tab = {}
    for d in sorted(glob.glob(os.path.join(groupcat, "groupcat_0??"))):
        sn = int(d[-3:])
        f0 = os.path.join(d, f"fof_subhalo_tab_{sn:03d}.0.hdf5")
        if os.path.exists(f0):
            with h5py.File(f0, "r") as f:
                z = float(f["Header"].attrs["Redshift"])
            if not np.isfinite(z) or z < 0:
                fail(f"{f0}: invalid header redshift {z}")
            tab[sn] = dict(z=z, age=float(Planck15.age(z).value), path=f0)
    for sn, z in Z_SNAP.items():
        if sn in tab and abs(tab[sn]["z"] - z) > 1e-6:
            fail(f"snapshot {sn} header redshift {tab[sn]['z']} != reference epoch {z}")
    zs = [tab[k]["z"] for k in sorted(tab)]
    if any(b > a for a, b in zip(zs, zs[1:])):
        fail("header redshifts must not increase with snapshot number")
    return tab


def reconstruct(dt_past, tau):
    if len(dt_past) == 0:
        return np.zeros(len(tau))
    return np.exp(-np.asarray(dt_past)[None, :] / np.asarray(tau)[:, None]).sum(1)


# ---------------------------------------------------------------- scoring
def same_order(refs):
    """Review R34-3: the diagnostic flattens predictions in each modality's
    native order, so rows and samples must be identical and identically
    ordered across modalities."""
    (m0, a), (m1, b) = list(refs.items())
    for k in ("group_id", "snap", "sample_group", "sample_snap", "sample_realization", "fold_of"):
        if not np.array_equal(a[k], b[k]):
            fail(f"{m0} vs {m1}: {k} order differs across modalities")


def fitting_parents(parents_fold, inner_f, f):
    return sorted(p for p, fo in parents_fold.items() if fo != f and p not in inner_f)


def mse_identity(p_hat, p, t):
    p_hat, p, t = (np.asarray(v, dtype=np.float64) for v in (p_hat, p, t))
    lhs = np.mean((p_hat - t) ** 2)
    a, b, c = np.mean((p_hat - p) ** 2), np.mean((p - t) ** 2), 2 * np.mean((p_hat - p) * (p - t))
    if not np.isclose(lhs, a + b + c, rtol=0, atol=1e-10):
        fail("accounting identity failed")
    return lhs, a, b, c


def affine_check(y, e):
    y, e = np.asarray(y, np.float64), np.asarray(e, np.float64)
    al, be = 2.5, -1.3
    d = abs(r2_score(al * y + be, al * e + be) - r2_score(y, e))
    if d > 1e-10:
        fail(f"affine-invariance check failed ({d:.3g})")
    return d


def ensemble(runs_m_lab, mask):
    """float64 seed-mean of single-view predictions on the masked samples."""
    return np.mean([d["oof_single"][mask].astype(np.float64).ravel() for d in runs_m_lab.values()], 0)


def nominate(comp):
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


def run(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-manifest", required=True)
    ap.add_argument("--pred-dir", default="cnn_preds")
    ap.add_argument("--out", default="tsc_hardness_diag.npz")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--boot-seed", type=int, default=20261106)
    ap.add_argument("--seeds", default=",".join(map(str, L.PRODUCTION_SEEDS)))
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        fail(f"refusing to overwrite {a.out}")
    seeds = [int(s) for s in a.seeds.split(",")]
    here = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    prov = dict(head=head, args=vars(a),
                sources={f: sha(os.path.join(here, f)) for f in
                         ("diagnose_tsc_hardness.py", "compare_labels.py", "compare_e0.py")},
                inputs={k: sha(getattr(a, k)) for k in ("curves", "mergers", "catalog")},
                thresholds=dict(a=A_SAT, tol_pseudo=TOL_PSEUDO, tol_tsc=TOL_TSC, tol_curve=TOL_CURVE,
                                eps_past=EPS_PAST, range_share=0.5, bias=0.3, boot=a.boot,
                                boot_seed=a.boot_seed))

    # ---- predictions: accepted validators + exact cross-modality order
    runs, refs = {}, {}
    pred_records = {}
    for m in L.MODALITIES:
        runs[m] = L.load(a.pred_dir, m, seeds)
        refs[m] = L.check(runs[m], m, a.expect_manifest, 60)
        for lab in L.LABELS:
            for s in seeds:
                p_ = os.path.join(a.pred_dir, L.pattern(m, lab).format(s))
                pred_records[p_] = dict(sha256=sha(p_), provenance=str(runs[m][lab][s]["provenance"]))
    L.check_cross(refs)
    same_order(refs)
    prov["predictions"] = pred_records
    ref = refs["x"]
    gid = as_int(ref["group_id"], "artifact group_id")
    snp = as_int(ref["snap"], "artifact snap")
    rows = list(zip(gid.tolist(), snp.tolist()))
    print(f"validated: 4 arms x {len(seeds)} seeds, {len(rows)} label rows; cross-modality order identical")

    # ---- A0 stored rule
    with h5py.File(a.curves, "r") as f:
        tau = f["labels/tau_gyr"][:]
        s_all, s_pre = f["labels/label_score_all"][:], f["labels/label_score_pre"][:]
        mk = list(zip(as_int(f["meta/group_id"][:], "curve group_id").tolist(),
                      as_int(f["meta/snap"][:], "curve snap").tolist()))
    tau = validate_curves(tau, s_all, s_pre)
    if len(mk) != s_all.shape[0]:
        fail("curve-file meta and curve rows differ in length")
    if len(set(mk)) != len(mk):
        fail("duplicate (parent, snapshot) rows in the curve file")
    pos = dict(zip(mk, range(len(mk))))
    if any(r not in pos for r in rows):
        fail("label rows missing from the curve file")
    curves = np.array([(s_all if s == 99 else s_pre)[pos[(g, s)]] for g, s in rows], dtype=np.float64)
    if not np.isfinite(curves).all():
        fail("non-finite stored curve values")
    cls_st = np.array([crossing_class(cv)[0] for cv in curves])
    idx_st = np.array([crossing_class(cv)[1] for cv in curves])
    ps_rec = np.array([pseudo_from_curve(cv, tau) for cv in curves])
    if not np.isfinite(ps_rec).all():
        fail("non-finite reconstructed stored-curve pseudo")
    dps_st = np.abs(ps_rec - ref["truth_pseudo"].astype(np.float64))
    if not (dps_st <= TOL_PSEUDO).all():
        fail(f"stored-curve pseudo does not reproduce the saved truth (max |diff| {dps_st.max():.3g})")
    print(f"\n== A0 stored rule: pseudo reproduced (max |diff| {dps_st.max():.2e} Gyr <= {TOL_PSEUDO}); classes: "
          + ", ".join(f"{c} {int((cls_st == c).sum())}" for c in CLASSES))

    # ---- A0 catalogue
    ev, members, by_parent = load_events(a.mergers, a.catalog)
    tab = snapshot_table(a.groupcat)
    prov["snapshot_table"] = {str(k): v for k, v in tab.items()}
    print(f"   snapshot table: {('snapshots %d-%d (%d headers)' % (min(tab), max(tab), len(tab))) if tab else 'EMPTY (snapshot clock unavailable)'}")
    n = len(rows)
    n2 = np.zeros(n, int); srow = np.full(n, -1); stied = np.zeros(n, int)
    sratio = np.full(n, np.nan); svalid = np.zeros(n, bool)
    S_t = np.zeros((n, len(tau))); S_s = np.full((n, len(tau)), np.nan)
    avail_s = np.zeros(n, bool); memb_chg = np.zeros(n, bool)
    for k_, (g, s) in enumerate(rows):
        t_now = float(Planck15.age(Z_SNAP[s]).value)
        evr = events_for(g, members, by_parent)
        past, dt = past_events(evr, ev["T_coll"], t_now)
        verify_recent(dt, float(ref["truth_tcoll_raw"][k_]), f"(parent {g}, snap {s})")
        n2[k_] = int((dt <= 2.0).sum())
        srow[k_], stied[k_], sratio[k_], svalid[k_] = setting_event(past, ev["T_coll"], ev["Mass_ratio"])
        S_t[k_] = reconstruct(dt, tau)
        pv = evr[ev["Snap_coll"][evr] <= s] if len(evr) else evr
        memb_chg[k_] = set(pv.tolist()) != set(past.tolist())
        if tab and all(int(x) in tab for x in ev["Snap_coll"][pv]):   # empty table: unavailable everywhere
            ages = np.array([tab[int(x)]["age"] for x in ev["Snap_coll"][pv]])
            S_s[k_] = reconstruct(np.maximum(t_now - ages, 0.0), tau)
            avail_s[k_] = True
    print(f"   most-recent past elapsed time matches the saved raw truth for all {n} rows (tol {TOL_TSC}); "
          f"catalogue members with no past event: {int((srow < 0).sum())}")
    a0 = {}
    for var, S, av in (("tcoll", S_t, np.ones(n, bool)), ("snap", S_s, avail_s)):
        cls_r = np.array([crossing_class(x)[0] if a_ else "" for x, a_ in zip(S, av)])
        ps_r = np.array([pseudo_from_curve(x, tau) if a_ else np.nan for x, a_ in zip(S, av)])
        d_ps = np.abs(ps_r - ref["truth_pseudo"])
        near = np.full(n, np.nan); full = np.full(n, np.nan)
        for k_ in np.where(av)[0]:
            j = idx_st[k_] if idx_st[k_] >= 0 else len(tau) - 1
            js = [x for x in (j - 1, j, j + 1) if 0 <= x < len(tau)]
            near[k_] = np.max(np.abs(S[k_, js] - curves[k_, js]))
            full[k_] = np.max(np.abs(S[k_] - curves[k_]))
        chg = av & ((cls_r != cls_st) | (d_ps > 0.05))
        a0[var] = dict(cls=cls_r, pseudo=ps_r, d_pseudo=d_ps, near_dS=near, full_dS=full, material=chg, avail=av)
        for sn in (99, 91):
            msk = (snp == sn)
            am = msk & av
            n_av = int(am.sum())
            reproduce = n_av == int(msk.sum()) and n_av > 0 and np.nanmax(full[am]) <= TOL_CURVE
            a0[var][f"reproduces_{sn}"] = bool(reproduce)
            print(f"   variant {var:<5} snap {sn}: available {n_av}/{int(msk.sum())}; near-crossing |dS| median "
                  f"{np.nanmedian(near[am]) if n_av else np.nan:.3g}; full-curve max |dS| median "
                  f"{np.nanmedian(full[am]) if n_av else np.nan:.3g}, max {np.nanmax(full[am]) if n_av else np.nan:.3g}; "
                  f"|d pseudo| median {np.nanmedian(d_ps[am]) if n_av else np.nan:.3g}; class changes "
                  f"{int((am & (cls_r != cls_st)).sum())}; material differences {int(chg[msk].sum())}; "
                  f"{'REPRODUCES (full-curve, full coverage)' if reproduce else 'catalogue-defined activity score (not a reproduction)'}")
    for sn in (99, 91):
        print(f"   Snap_coll past-event membership differs from T_coll for {int(memb_chg[snp == sn].sum())} rows at snap {sn}")

    # ---- snapshot-99 strata
    m99 = snp == 99
    par99 = gid[m99]
    t_raw99 = ref["truth_tcoll_raw"][m99].astype(np.float64)
    t_cap99 = ref["truth_tcoll_capped"][m99].astype(np.float64)
    p99 = ref["truth_pseudo"][m99].astype(np.float64)
    st_tsc = np.array([tsc_stratum(x) for x in t_raw99])
    st_n2 = np.array([n2_stratum(x) for x in n2[m99]])
    vr = sratio[m99][svalid[m99]]
    q1, q2 = (np.quantile(vr, [1 / 3, 2 / 3]) if len(vr) else (np.nan, np.nan))
    st_mr = np.array(["missing" if r < 0 else "invalid" if not v else
                      ("mr<=q1" if x <= q1 else "q1<mr<=q2" if x <= q2 else "mr>q2")
                      for r, v, x in zip(srow[m99], svalid[m99], sratio[m99])])
    cls99 = cls_st[m99]
    print(f"\n== A1 strata (snapshot 99, {len(par99)} parents); mass-ratio edges q1={q1:.4g}, q2={q2:.4g}; "
          f"setting events with ties: {int((stied[m99] > 1).sum())}")
    print("   TSC band x stored pseudo class (parent counts): " + "  ".join(f"{c}" for c in CLASSES))
    xtab = np.zeros((len(TSC_STRATA), len(CLASSES)), int)
    for i_, lev in enumerate(TSC_STRATA):
        for j_, c in enumerate(CLASSES):
            xtab[i_, j_] = int(((st_tsc == lev) & (cls99 == c)).sum())
        print(f"     {lev:<10} " + "  ".join(f"{v:>5}" for v in xtab[i_]))
    R, P = L.R_EXPECT, L.P_EXPECT
    rep = lambda v: np.repeat(np.repeat(np.asarray(v), R)[:, None], P, 1).ravel()
    obs_par = rep(par99)
    y = rep(t_cap99)
    yp = rep(p99)
    ms = refs["x"]["sample_snap"] == 99
    ens = {(m, lab): ensemble(runs[m][lab], ms) for m in L.MODALITIES for lab in L.LABELS}
    rng = np.random.default_rng(a.boot_seed)
    comp_x = {}
    strata_rows = []
    for m in L.MODALITIES:
        e_c, e_p = ens[(m, "tsc_tcoll")], ens[(m, "pseudo")]
        sse_c_tot, sse_p_tot = np.sum((e_c - y) ** 2), np.sum((e_p - y) ** 2)
        print(f"  -- {m}: stratum | parents obs | bias cat/pse | MAE cat/pse | MSE cat/pse | SSE share cat/pse | "
              f"dMSE cat-pse [95%]")
        for fam, st, levels in (("TSC", st_tsc, TSC_STRATA), ("pseudo class", cls99, CLASSES),
                                ("n_2Gyr", st_n2, N2_STRATA), ("mass ratio", st_mr, MR_STRATA)):
            for lev in levels:
                ps_ = par99[st == lev]
                if len(ps_) == 0:
                    print(f"     {fam}:{lev:<12} empty")
                    continue
                o = np.isin(obs_par, ps_)
                bc, bp = np.mean(e_c[o] - y[o]), np.mean(e_p[o] - y[o])
                mc, mp = np.mean((e_c[o] - y[o]) ** 2), np.mean((e_p[o] - y[o]) ** 2)
                lo = hi = np.nan
                if len(ps_) >= 2:
                    idx = {p_: np.where(obs_par == p_)[0] for p_ in ps_}
                    bs = []
                    for _ in range(a.boot):
                        s_ = np.concatenate([idx[p_] for p_ in rng.choice(ps_, len(ps_))])
                        bs.append(np.mean((e_c[s_] - y[s_]) ** 2) - np.mean((e_p[s_] - y[s_]) ** 2))
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                sc, sp = np.sum((e_c[o] - y[o]) ** 2) / sse_c_tot, np.sum((e_p[o] - y[o]) ** 2) / sse_p_tot
                print(f"     {fam}:{lev:<12} {len(ps_):>4} {int(o.sum()):>5} | {bc:+.3f}/{bp:+.3f} | "
                      f"{np.mean(np.abs(e_c[o] - y[o])):.3f}/{np.mean(np.abs(e_p[o] - y[o])):.3f} | {mc:.3f}/{mp:.3f} | "
                      f"{sc:.3f}/{sp:.3f} | {mc - mp:+.3f} " + (f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else "(n<2)"))
                strata_rows.append(dict(modality=m, family=fam, level=lev, parents=int(len(ps_)), obs=int(o.sum()),
                                        bias_cat=bc, bias_pse=bp, mse_cat=mc, mse_pse=mp, share_cat=sc, share_pse=sp,
                                        dmse_lo=lo, dmse_hi=hi))
                if m == "x" and fam == "TSC":
                    comp_x[lev] = dict(sse_cat=float(np.sum((e_c[o] - y[o]) ** 2)), mse_cat=mc, mse_pse=mp,
                                       n_obs=int(o.sum()), bias_cat=bc)
        if m == "x":
            comp_x["sse_cat_total"] = float(sse_c_tot)

    # ---- A2
    print("\n== A2 metric accounting (snapshot-99 single observations, float64)")
    a2 = []
    for m in L.MODALITIES:
        for lab in L.LABELS:
            e = ens[(m, lab)]
            print(f"   {m}/{lab:<9}: vs T_coll R2 {r2_score(y, e):.3f} MSE {np.mean((e - y) ** 2):.3f} Var {np.var(y):.3f} | "
                  f"vs pseudo R2 {r2_score(yp, e):.3f} MSE {np.mean((e - yp) ** 2):.3f} Var {np.var(yp):.3f}")
            affine_check(y, e)
        lhs, t1, t2, t3 = mse_identity(ens[(m, "pseudo")], yp, y)
        print(f"   {m} pseudo arm: MSE vs T {lhs:.3f} = model-vs-pseudo {t1:.3f} + pseudo-vs-T {t2:.3f} + 2 cross {t3:+.3f}")
        a2.append(dict(modality=m, stratum="all", lhs=lhs, model_vs_pseudo=t1, pseudo_vs_T=t2, cross=t3))
        for lev in TSC_STRATA:
            o = np.isin(obs_par, par99[st_tsc == lev])
            if o.any():
                l_, a_, b_, c_ = mse_identity(ens[(m, "pseudo")][o], yp[o], y[o])
                print(f"      {lev:<10} {l_:.3f} = {a_:.3f} + {b_:.3f} + {c_:+.3f}")
                a2.append(dict(modality=m, stratum=lev, lhs=l_, model_vs_pseudo=a_, pseudo_vs_T=b_, cross=c_))
        e = ens[(m, "tsc_tcoll")]
        ya, ea = np.minimum(y, A_SAT), np.minimum(e, A_SAT)
        print(f"   {m} catalogue arm, upper-clipped at a: R2 {r2_score(ya, ea):.3f} MSE {np.mean((ea - ya) ** 2):.3f} "
              f"Var {np.var(ya):.3f} (coarsened endpoint; original R2 {r2_score(y, e):.3f} MSE {np.mean((e - y) ** 2):.3f})")
    print("   affine invariance: checked on every arm (float64)")

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
    knots, fitsets = {}, {}
    clips = {"p2t": 0, "t2p": 0}
    clip_tr = {m: {} for m in L.MODALITIES}
    transfer = {m: np.full(len(y), np.nan) for m in L.MODALITIES}
    for f in range(L.N_FOLD):
        fit = fitting_parents(pf99, folds["inner"][f], f)
        held = sorted(p_ for p_, fo in pf99.items() if fo == f)
        if set(fit) & set(held) or set(fit) & folds["inner"][f] or not fit or not held:
            fail(f"fold {f}: fitting parents overlap held-out/inner parents or a set is empty")
        fitsets[f] = (fit, held)
        fi, hi = np.isin(par99, fit), np.isin(par99, held)
        for key_, xa, ya_ in (("p2t", p99, t_cap99), ("t2p", t_cap99, p99)):
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(xa[fi], ya_[fi])
            res[key_][hi] = iso.predict(xa[hi])
            clips[key_] += int(((xa[hi] < xa[fi].min()) | (xa[hi] > xa[fi].max())).sum())
            knots[f"{key_}_{f}"] = [iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()]
            if key_ == "p2t":
                oh = np.isin(obs_par, held)
                for m in L.MODALITIES:
                    xin = ens[(m, "pseudo")][oh]
                    transfer[m][oh] = iso.predict(xin)
                    clip_tr[m][f] = int(((xin < xa[fi].min()) | (xin > xa[fi].max())).sum())
    print(f"   true pseudo -> capped T: R2 {r2_score(t_cap99, res['p2t']):.3f} MSE "
          f"{np.mean((res['p2t'] - t_cap99) ** 2):.3f} Var {np.var(t_cap99):.3f} (one row per parent; clips {clips['p2t']})")
    print(f"   capped T -> true pseudo: R2 {r2_score(p99, res['t2p']):.3f} MSE "
          f"{np.mean((res['t2p'] - p99) ** 2):.3f} Var {np.var(p99):.3f} (clips {clips['t2p']})")
    for m in L.MODALITIES:
        tm = transfer[m]
        print(f"   label-map transfer, {m} pseudo arm (seeds averaged, then mapped) vs T_coll: R2 {r2_score(y, tm):.3f} "
              f"MSE {np.mean((tm - y) ** 2):.3f} Var {np.var(y):.3f} (prediction-input clips by fold {clip_tr[m]}; "
              f"unmapped R2 {r2_score(y, ens[(m, 'pseudo')]):.3f}; catalogue arm R2 {r2_score(y, ens[(m, 'tsc_tcoll')]):.3f})")

    # ---- nomination
    cand, share, mc_r, mp_r = nominate(comp_x)
    fin_old = (comp_x.get("(a,4]", {}).get("sse_cat", 0) + comp_x.get("finite>4", {}).get("sse_cat", 0)) / comp_x["sse_cat_total"]
    miss_sh = comp_x.get("missing", {}).get("sse_cat", 0) / comp_x["sse_cat_total"]
    by = comp_x.get("[0,0.5]", {}).get("bias_cat", np.nan)
    bo = comp_x.get("finite>4", {}).get("bias_cat", np.nan)
    print(f"\n== nomination (X-ray, practical prioritisation; not a test): range-strata SSE share {share:.3f} "
          f"(finite-old {fin_old:.3f}, missing {miss_sh:.3f}); combined MSE cat {mc_r:.3f} vs pseudo {mp_r:.3f}; "
          f"bias cat [0,0.5] {by:+.3f}, finite>4 {bo:+.3f} -> candidate: {cand}")

    out = dict(row_group=gid, row_snap=snp, parent99=par99,
               cls_stored=cls_st, idx_stored=idx_st, pseudo_stored_rec=ps_rec, d_pseudo_stored=dps_st,
               n2=n2, setting_row=srow, setting_tied=stied, setting_ratio=sratio, setting_valid=svalid,
               membership_change=memb_chg, avail_snapclock=avail_s,
               S_tcoll=S_t, S_snap=S_s, tau=tau,
               st_tsc=st_tsc, st_n2=st_n2, st_mr=st_mr, mr_edges=np.array([q1, q2]), xtab=xtab,
               xtab_rows=np.array(TSC_STRATA), xtab_cols=np.array(CLASSES),
               strata=np.array(json.dumps(strata_rows)), a2=np.array(json.dumps(a2)),
               a3_p2t=res["p2t"], a3_t2p=res["t2p"], transfer_x=transfer["x"], transfer_r=transfer["r"],
               a3_clips=np.array(json.dumps(dict(label=clips, transfer=clip_tr))),
               knots=np.array(json.dumps(knots)), fitsets=np.array(json.dumps({str(k): v for k, v in fitsets.items()})),
               nomination=np.array(json.dumps(dict(candidate=cand, share=share, finite_old_share=fin_old,
                                                   missing_share=miss_sh, mse_cat=mc_r, mse_pse=mp_r,
                                                   bias_young=by, bias_old=bo))),
               provenance=np.array(json.dumps(prov, default=str)))
    for var, d in a0.items():
        for k, v in d.items():
            out[f"a0_{var}_{k}"] = np.asarray(v)
    np.savez(a.out, **out)
    print(f"\nsaved {a.out}")
    return out


if __name__ == "__main__":
    run()
