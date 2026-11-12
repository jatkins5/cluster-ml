#!/usr/bin/env python3
"""Phase 5 (review rounds 42-43): is the learned image statistic associated with
catalogue-defined EARLIER merger activity, at fixed time since the latest
recorded collision? Diagnostics on SAVED predictions only; exploratory and
conditional; no causal claim.

  diagnose_history.py --expect-manifest <sha256> --out history_diag.npz

Response: one row per snapshot-99 parent = mean over the 5 seeds, then over
the parent's 9 single observations. Primary arm: Huber catalogue-trained
X-ray (lbl_x_tsc_tcoll); radio and pseudo arms secondary/contextual.

Features (catalogue-defined, T_coll clock, Tier-0 admission/zero-clipping,
latest elapsed time verified against the saved raw truth first):
  latest group = ALL admitted rows at the maximum admitted T_coll
  prev         = admitted rows with T_coll strictly below that maximum
  S_prev(tau) = sum_prev exp(-dt/tau); M_prev(tau) = sum_prev mu exp(-dt/tau)
  (tau0 = 1 Gyr; 0.5, 2 labelled sensitivities). M_prev is UNAVAILABLE if any
  prev ratio is invalid (negative/non-finite); no prev events -> genuine 0.
  M_all likewise unavailable if ANY included event's ratio is invalid;
  latest strength = summed latest-group ratio, unavailable if any invalid.
Primary cohort: finite 0 <= raw T_last <= 4 Gyr, intersected with M_prev(1)
availability.

B1  OLS: yhat = 1 + 4 reference-coded outer-fold indicators + g(T_last) +
    beta * M_prev. Primary g: piecewise linear, columns
    [T, (T-0.25)+, (T-0.5)+, (T-1)+, (T-2)+, (T-3)+] (with the intercept: 7
    nuisance-time columns). Alternate g (one, fixed): natural cubic spline,
    boundary knots 0 and 4, interior 0.25, 0.75, 1.5, 3.0 (ESL 5.2.1
    truncated-power form; 6 columns with the intercept). Adjusted
    sensitivity: + true log M500 (sim_logm), latest strength, latest tie
    count, with the unadjusted model refitted on the SAME parents. 4000
    attempted parent-bootstrap draws stratified by original outer fold, every
    term refitted, no retries; any invalid draw -> interval unavailable.
    Support: residual SD of M_prev after the nuisance design < 5% of its raw
    SD -> "insufficient support under the prespecified rule"; rank deficiency
    is a distinct failure. One-SD effect = beta x fixed original-cohort SD.
B2  within each original outer fold: sorted parent ids, seeded balanced
    two-way split (seed 20261112 + f), fit on one half / evaluate on the
    other, both directions; time-only (7 columns) vs + M_prev (8 columns),
    OLS on the primary basis. Each half n >= 20, both fits full rank, support
    rule on the calibration half. Primary result only with valid held-out
    predictions for the ENTIRE primary cohort across all five folds. Held-out
    R2 of each and the paired parent bootstrap of the difference (stratified
    by fold). Secondary: 1-D isotonic rankings (T_last, pseudo increasing;
    S_all, M_all decreasing) on their own common cohort, same splits.
Gate for DISCUSSING B3 (practical; pass / fail / unavailable per part): (i)
primary beta < 0 with a valid interval excluding 0; (ii) alternate-basis beta
the same sign with a valid interval excluding 0; (iii) B2 full-coverage
held-out dR2 > 0 with a valid interval excluding 0. Not a significance test;
failure is not evidence of absence.
"""
import argparse
import json
import os
import subprocess

import numpy as np
from sklearn.isotonic import IsotonicRegression

import compare_labels as L
import diagnose_tsc_hardness as D

TAU0, TAU_SENS = 1.0, (0.5, 2.0)
KNOTS_PL = (0.25, 0.5, 1.0, 2.0, 3.0)
# PROPOSED AMENDMENT (round 44, awaiting acceptance): B2 fits use a reduced
# basis. With KNOTS_PL, 3 of 10 real fit halves (~30 parents) contain no
# T_last in (3, 4], so (T-3)+ is an all-zero column and the primary is
# unavailable by construction (checked from T_last only).
B2_KNOTS = (0.5, 1.0, 2.0)
NCS_KNOTS = (0.0, 0.25, 0.75, 1.5, 3.0, 4.0)        # boundary 0 and 4, interior as listed
T_MAX = 4.0
BOOT, BOOT_SEED, SPLIT_SEED = 4000, 20261112, 20261112
MIN_HALF, SUPPORT_FRAC = 20, 0.05
RANK_TOL = 1e-10


def fail(msg):
    raise SystemExit(msg)


# ---------------------------------------------------------------- features
def event_features(rows, tcoll, mratio, t_now):
    """Strict-previous history features for one parent-snapshot."""
    past, dt = D.past_events(rows, tcoll, t_now)
    out = dict(T_last=np.nan, n_latest=0, n_prev=0, prev_invalid=0, latest_invalid=0,
               latest_rows=[], prev_rows=[], latest_strength=np.nan)
    for tau in (TAU0,) + TAU_SENS:
        out[f"S_prev_{tau}"] = 0.0
        out[f"M_prev_{tau}"] = 0.0
        out[f"M_prev_incomplete_{tau}"] = 0.0
        out[f"S_all_{tau}"] = 0.0
        out[f"M_all_{tau}"] = 0.0
    if len(past) == 0:
        return out, dt
    tmax = tcoll[past].max()
    latest = past[tcoll[past] == tmax]
    prev = past[tcoll[past] < tmax]
    dtp = np.maximum(t_now - tcoll[prev], 0.0)
    dtl = np.maximum(t_now - tcoll[latest], 0.0)
    ok = lambda r: np.isfinite(mratio[r]) & (mratio[r] >= 0)
    out.update(T_last=float(np.min(dt)), n_latest=len(latest), n_prev=len(prev),
               prev_invalid=int((~ok(prev)).sum()), latest_invalid=int((~ok(latest)).sum()),
               latest_rows=latest.tolist(), prev_rows=prev.tolist())
    out["latest_strength"] = float(mratio[latest].sum()) if ok(latest).all() else np.nan
    for tau in (TAU0,) + TAU_SENS:
        wp, wl = np.exp(-dtp / tau), np.exp(-dtl / tau)
        out[f"S_prev_{tau}"] = float(wp.sum())
        out[f"S_all_{tau}"] = float(wp.sum() + wl.sum())
        vp = ok(prev)
        out[f"M_prev_incomplete_{tau}"] = float((mratio[prev][vp] * wp[vp]).sum())
        out[f"M_prev_{tau}"] = float((mratio[prev] * wp).sum()) if vp.all() else np.nan
        out[f"M_all_{tau}"] = (float((mratio[prev] * wp).sum() + (mratio[latest] * wl).sum())
                               if vp.all() and ok(latest).all() else np.nan)
    return out, dt


# ---------------------------------------------------------------- designs
def pl_basis(T, knots=KNOTS_PL):
    T = np.asarray(T, np.float64)
    cols = [np.ones_like(T), T] + [np.maximum(T - k, 0.0) for k in knots]
    names = ["1", "T"] + [f"(T-{k})+" for k in knots]
    return np.column_stack(cols), names


def ncs_basis(T, knots=NCS_KNOTS):
    """Natural cubic spline (ESL eq. 5.4-5.5): 1, T, and K-2 terms; K = 6."""
    T, xi = np.asarray(T, np.float64), np.asarray(knots, np.float64)
    K = len(xi)
    d = lambda k: (np.maximum(T - xi[k], 0) ** 3 - np.maximum(T - xi[K - 1], 0) ** 3) / (xi[K - 1] - xi[k])
    cols = [np.ones_like(T), T] + [d(k) - d(K - 2) for k in range(K - 2)]
    names = ["1", "T"] + [f"ncs{k}" for k in range(K - 2)]
    return np.column_stack(cols), names


def fold_indicators(fold):
    fold = np.asarray(fold, int)
    return np.column_stack([(fold == f).astype(float) for f in range(1, L.N_FOLD)]), \
        [f"fold{f}" for f in range(1, L.N_FOLD)]


def ols(X, y):
    """(coef, rank_ok, condition number of the unit-norm column-scaled X)."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        return None, False, np.inf
    norms = np.linalg.norm(X, axis=0)
    if (norms == 0).any():
        return None, False, np.inf
    Xs = X / norms
    rank = np.linalg.matrix_rank(Xs, tol=RANK_TOL * max(Xs.shape))
    cond = float(np.linalg.cond(Xs))
    if rank < X.shape[1]:
        return None, False, cond
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return coef, True, cond


def support(X_nuis, f):
    """Residual SD of feature f after the nuisance design / raw SD."""
    f = np.asarray(f, np.float64)
    sd = float(np.std(f))
    if not np.isfinite(sd) or sd == 0:
        return np.nan, "zero or non-finite raw SD"
    c, ok, _ = ols(X_nuis, f)
    if not ok:
        return np.nan, "nuisance design rank-deficient"
    r = float(np.std(f - X_nuis @ c) / sd)
    return r, ("ok" if r >= SUPPORT_FRAC else "insufficient support under the prespecified rule")


def b1_fit(y, T, fold, F, basis="pl", extra=None, boot=BOOT, seed=BOOT_SEED):
    """Fit the B1 model and its stratified parent bootstrap. Returns a dict
    with coefficient, interval (or None) and every status."""
    Xb, nb = (pl_basis(T) if basis == "pl" else ncs_basis(T))
    Xf, nf = fold_indicators(fold)
    nuis = [Xb, Xf] + ([extra[0]] if extra is not None else [])
    names = nb + nf + (extra[1] if extra is not None else []) + ["F"]
    Xn = np.column_stack(nuis)
    X = np.column_stack([Xn, F])
    res = dict(n=len(y), names=names, beta=np.nan, sd_ref=float(np.std(F)), lo=np.nan, hi=np.nan,
               status="ok", support=np.nan, cond=np.nan, partial_r2=np.nan, draws_valid=0, draws=boot)
    sup, sstat = support(Xn, F)
    res["support"] = sup
    coef, ok, cond = ols(X, y)
    res["cond"] = cond
    if not ok:
        res["status"] = "rank-deficient"
        return res
    if sstat != "ok":
        res["status"] = sstat
    res["beta"] = float(coef[-1])
    cn, okn, _ = ols(Xn, y)
    if okn:
        sse_r = np.sum((y - Xn @ cn) ** 2)
        res["partial_r2"] = float((sse_r - np.sum((y - X @ coef) ** 2)) / sse_r)
    rng = np.random.default_rng(seed)
    idx_f = [np.where(fold == f)[0] for f in range(L.N_FOLD)]
    bs, bad = [], []
    for b in range(boot):
        s = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_f if len(ix)])
        cb, okb, _ = ols(X[s], y[s])
        if not okb or not np.isfinite(cb[-1]):
            bad.append(b)
            continue
        bs.append(cb[-1])
    res["draws_valid"] = len(bs)
    res["boot_betas"] = np.array(bs)
    res["failed_draws"] = np.array(bad, dtype=int)
    if boot == 0:
        res["interval_status"] = "not requested"
    elif bad:
        res["interval_status"] = f"unavailable: {len(bad)} of {boot} draws invalid"
    else:
        res["lo"], res["hi"] = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
        res["interval_status"] = "ok"
    return res


def gate_part(res, sign=None):
    """pass / fail / unavailable for a B1 gate component."""
    if res["status"] != "ok" or res.get("interval_status") != "ok":
        return "unavailable"
    neg = res["beta"] < 0 and res["hi"] < 0
    pos = res["beta"] > 0 and res["lo"] > 0
    if sign is None:
        return "pass" if neg else "fail"
    return "pass" if (neg if sign < 0 else pos) else "fail"


# ---------------------------------------------------------------- B2
def make_splits(parents, parent_fold):
    """Per original fold: sorted parent ids, seeded balanced halves."""
    splits = {}
    for f in range(L.N_FOLD):
        ps = np.array(sorted(p for p in parents if parent_fold[p] == f))
        perm = np.random.default_rng(SPLIT_SEED + f).permutation(len(ps))
        h = len(ps) // 2
        splits[f] = (ps[perm[:h]], ps[perm[h:]])
    return splits


def check_splits(splits, parent_fold):
    """Calibration/evaluation halves must be disjoint and from ONE original fold."""
    for f, (a, b) in splits.items():
        if set(a.tolist()) & set(b.tolist()):
            fail(f"fold {f}: calibration and evaluation halves overlap")
        for p in list(a) + list(b):
            if parent_fold[int(p)] != f:
                fail(f"fold {f}: parent {p} belongs to original fold {parent_fold[int(p)]} (cross-fold calibration)")


def b2_run(par, y, T, F, parent_fold, splits, boot=BOOT, seed=BOOT_SEED, min_half=MIN_HALF, knots=B2_KNOTS):
    """Held-out predictions for time-only and +F models, per fold and
    direction; primary only if every parent is covered."""
    pos = {int(p): i for i, p in enumerate(par)}
    pred0, pred1 = np.full(len(par), np.nan), np.full(len(par), np.nan)
    status = {}
    for f, (a, b) in splits.items():
        for name, fit, ev in (("A->B", a, b), ("B->A", b, a)):
            fi = np.array([pos[int(p)] for p in fit if int(p) in pos], int)
            ei = np.array([pos[int(p)] for p in ev if int(p) in pos], int)
            key = f"fold{f}:{name}"
            if len(fi) < min_half or len(ei) < min_half:
                status[key] = f"unavailable: n fit {len(fi)} / eval {len(ei)} < {min_half}"
                continue
            X0f, _ = pl_basis(T[fi], knots)
            X1f = np.column_stack([X0f, F[fi]])
            c0, ok0, _ = ols(X0f, y[fi])
            c1, ok1, _ = ols(X1f, y[fi])
            sup, sst = support(X0f, F[fi])
            if not (ok0 and ok1):
                status[key] = "unavailable: rank-deficient fit"
                continue
            if sst != "ok":
                status[key] = f"unavailable: {sst}"
                continue
            X0e, _ = pl_basis(T[ei], knots)
            pred0[ei] = X0e @ c0
            pred1[ei] = np.column_stack([X0e, F[ei]]) @ c1
            status[key] = "ok"
    full = bool(np.isfinite(pred0).all() and np.isfinite(pred1).all())
    res = dict(status=status, full_coverage=full, pred0=pred0, pred1=pred1,
               covered=int((np.isfinite(pred0) & np.isfinite(pred1)).sum()), n=len(par))
    m = np.isfinite(pred0) & np.isfinite(pred1)
    r2 = lambda yy, pp: (1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)
                         if np.var(yy) > 0 else np.nan)
    res["r2_0"], res["r2_1"] = (r2(y[m], pred0[m]), r2(y[m], pred1[m])) if m.any() else (np.nan, np.nan)
    res["dr2"] = res["r2_1"] - res["r2_0"]
    res["lo"] = res["hi"] = np.nan
    if not full:
        res["interval_status"] = f"unavailable: primary needs full coverage ({res['covered']}/{len(par)})"
        return res
    folds = np.array([parent_fold[int(p)] for p in par])
    rng = np.random.default_rng(seed)
    idx = [np.where(folds == f)[0] for f in range(L.N_FOLD)]
    bs, bad = [], 0
    for _ in range(boot):
        s = np.concatenate([rng.choice(ix, len(ix)) for ix in idx if len(ix)])
        d = r2(y[s], pred1[s]) - r2(y[s], pred0[s])
        if not np.isfinite(d):
            bad += 1
            continue
        bs.append(d)
    if bad:
        res["interval_status"] = f"unavailable: {bad} of {boot} draws undefined"
    else:
        res["lo"], res["hi"] = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
        res["interval_status"] = "ok"
    return res


def b2_isotonic(par, y, summaries, parent_fold, splits, min_half=MIN_HALF):
    """Secondary 1-D isotonic rankings on the same splits (common cohort)."""
    pos = {int(p): i for i, p in enumerate(par)}
    out = {}
    for name, (x, increasing) in summaries.items():
        pred = np.full(len(par), np.nan)
        clips, stat = 0, {}
        for f, (a, b) in splits.items():
            for d_, fit, ev in (("A->B", a, b), ("B->A", b, a)):
                fi = np.array([pos[int(p)] for p in fit if int(p) in pos], int)
                ei = np.array([pos[int(p)] for p in ev if int(p) in pos], int)
                if len(fi) < min_half or len(ei) < min_half:
                    stat[f"fold{f}:{d_}"] = "unavailable: n < min"
                    continue
                iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(x[fi], y[fi])
                pred[ei] = iso.predict(x[ei])
                clips += int(((x[ei] < x[fi].min()) | (x[ei] > x[fi].max())).sum())
                stat[f"fold{f}:{d_}"] = "ok"
        m = np.isfinite(pred)
        r2 = (1 - np.sum((y[m] - pred[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)) if m.sum() > 1 else np.nan
        out[name] = dict(r2=float(r2), covered=int(m.sum()), n=len(par), clips=clips, status=stat)
    return out


# ---------------------------------------------------------------- main
def run(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-manifest", required=True)
    ap.add_argument("--pred-dir", default="cnn_preds")
    ap.add_argument("--out", default="history_diag.npz")
    ap.add_argument("--seeds", default=",".join(map(str, L.PRODUCTION_SEEDS)))
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--min-half", type=int, default=MIN_HALF)
    for k in ("mergers", "catalog"):
        ap.add_argument(f"--{k}", default=D.DEFAULTS[k])
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        fail(f"refusing to overwrite {a.out}")
    seeds = [int(s) for s in a.seeds.split(",")]
    here = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    prov = dict(head=head, args=vars(a),
                sources={f: D.sha(os.path.join(here, f)) for f in
                         ("diagnose_history.py", "diagnose_tsc_hardness.py", "compare_labels.py", "compare_e0.py")},
                inputs={k: D.sha(getattr(a, k)) for k in ("mergers", "catalog")},
                constants=dict(tau0=TAU0, tau_sens=TAU_SENS, knots_pl=KNOTS_PL, b2_knots=B2_KNOTS, ncs_knots=NCS_KNOTS,
                               boot=a.boot, boot_seed=BOOT_SEED, split_seed=SPLIT_SEED, min_half=a.min_half,
                               support_frac=SUPPORT_FRAC, rank_tol=RANK_TOL))

    runs, refs, pred_records = {}, {}, {}
    for m in L.MODALITIES:
        runs[m] = L.load(a.pred_dir, m, seeds)
        refs[m] = L.check(runs[m], m, a.expect_manifest, 60)
        for lab in L.LABELS:
            for s in seeds:
                p_ = os.path.join(a.pred_dir, L.pattern(m, lab).format(s))
                pred_records[p_] = dict(sha256=D.sha(p_), provenance=str(runs[m][lab][s]["provenance"]))
    L.check_cross(refs)
    D.same_order(refs)
    prov["predictions"] = pred_records
    ref = refs["x"]
    gid, snp = D.as_int(ref["group_id"], "group_id"), D.as_int(ref["snap"], "snap")
    m99 = snp == 99
    par = gid[m99]
    raw = ref["truth_tcoll_raw"][m99].astype(np.float64)
    cap = ref["truth_tcoll_capped"][m99].astype(np.float64)
    pse = ref["truth_pseudo"][m99].astype(np.float64)
    if "sim_logm" not in ref or ref["sim_logm"].shape != gid.shape or not np.isfinite(ref["sim_logm"]).all():
        fail("sim_logm missing, mis-shaped or non-finite")
    for m in L.MODALITIES:
        for lab in L.LABELS:
            for d in runs[m][lab].values():
                if not np.array_equal(d["sim_logm"], ref["sim_logm"]):
                    fail(f"{m}/{lab}: sim_logm differs across artifacts")
    logm = ref["sim_logm"][m99].astype(np.float64)
    pf = {int(g_): int(f_) for g_, f_ in zip(ref["sample_group"], ref["fold_of"])}
    fold = np.array([pf[int(p)] for p in par])

    # responses: 5-seed ensemble, then the parent's 9 single observations
    ms = ref["sample_snap"] == 99
    R, P = L.R_EXPECT, L.P_EXPECT
    resp = {}
    for m in L.MODALITIES:
        for lab in L.LABELS:
            e = D.ensemble(runs[m][lab], ms).reshape(len(par), R * P)
            resp[(m, lab)] = e.mean(axis=1)

    # history features
    ev, members, by_parent = D.load_events(a.mergers, a.catalog)
    t_now = float(D.Planck15.age(D.Z_SNAP[99]).value)
    feats = []
    for k_, g in enumerate(par):
        evr = D.events_for(int(g), members, by_parent)
        fo, dt = event_features(evr, ev["T_coll"], ev["Mass_ratio"], t_now)
        D.verify_recent(dt, float(raw[k_]), f"parent {g}")
        feats.append(fo)
    F = lambda key: np.array([f[key] for f in feats], dtype=np.float64)
    T_last = F("T_last")
    mprev = F(f"M_prev_{TAU0}")
    finite_le4 = np.isfinite(T_last) & (T_last <= T_MAX)
    primary = finite_le4 & np.isfinite(mprev)
    adj = primary & np.isfinite(F("latest_strength"))
    iso_c = primary & np.isfinite(F(f"M_all_{TAU0}"))
    reason = np.array(["no recorded event" if not np.isfinite(t) else "T_last > 4" if t > T_MAX
                       else "invalid prev ratio" if not np.isfinite(mv) else "" for t, mv in zip(T_last, mprev)])
    print(f"cohort: {len(par)} snapshot-99 parents; finite 0<=T_last<=4: {int(finite_le4.sum())}; primary "
          f"(M_prev available): {int(primary.sum())}; excluded: "
          + ", ".join(f"{r_} {int((reason == r_).sum())}" for r_ in ("no recorded event", "T_last > 4",
                                                                      "invalid prev ratio"))
          + f"; adjustment-eligible {int(adj.sum())}; isotonic common cohort {int(iso_c.sum())}")
    print(f"   prev events: parents with any prev {int((F('n_prev') > 0)[primary].sum())}; latest tie groups > 1: "
          f"{int((F('n_latest') > 1)[primary].sum())}; M_prev(1) raw SD {np.std(mprev[primary]):.4g}")

    # ---- B1
    out_b1 = {}
    pm = primary
    y_x = resp[("x", "tsc_tcoll")]
    def show(lab, r):
        print(f"   {lab:<44} n {r['n']:>3} beta {r['beta']:+.4f} (one-SD {r['beta'] * r['sd_ref']:+.4f}) "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {r.get('interval_status', '')}; status {r['status']}; "
              f"support {r['support']:.3f}; cond {r['cond']:.3g}; partial R2 {r['partial_r2']:.4f}")
    print("\n== B1 (parent-level OLS, original-fold intercepts; raw beta per unit of M_prev)")
    out_b1["primary"] = b1_fit(y_x[pm], T_last[pm], fold[pm], mprev[pm], "pl", boot=a.boot)
    show("PRIMARY x/tsc_tcoll, piecewise linear, M_prev", out_b1["primary"])
    out_b1["alt"] = b1_fit(y_x[pm], T_last[pm], fold[pm], mprev[pm], "ncs", boot=a.boot)
    show("x/tsc_tcoll, natural cubic spline, M_prev", out_b1["alt"])
    ex = (np.column_stack([logm, F("latest_strength"), F("n_latest")]), ["logM500", "latest_strength", "n_latest"])
    out_b1["adj_unadjusted_same"] = b1_fit(y_x[adj], T_last[adj], fold[adj], mprev[adj], "pl", boot=a.boot)
    show("x/tsc_tcoll, unadjusted on adjustment cohort", out_b1["adj_unadjusted_same"])
    out_b1["adjusted"] = b1_fit(y_x[adj], T_last[adj], fold[adj], mprev[adj], "pl",
                                extra=(ex[0][adj], ex[1]), boot=a.boot)
    show("x/tsc_tcoll, + logM500, latest strength, ties", out_b1["adjusted"])
    for tau in TAU_SENS:
        mt = F(f"M_prev_{tau}")
        out_b1[f"tau{tau}"] = b1_fit(y_x[pm], T_last[pm], fold[pm], mt[pm], "pl", boot=a.boot)
        show(f"x/tsc_tcoll, M_prev tau={tau} (sensitivity)", out_b1[f"tau{tau}"])
    sp = F(f"S_prev_{TAU0}")
    out_b1["S_prev"] = b1_fit(y_x[pm], T_last[pm], fold[pm], sp[pm], "pl", boot=a.boot)
    show("x/tsc_tcoll, S_prev (secondary)", out_b1["S_prev"])
    for lab_, key in (("r/tsc_tcoll (secondary)", ("r", "tsc_tcoll")), ("x/pseudo (context)", ("x", "pseudo")),
                      ("r/pseudo (context)", ("r", "pseudo"))):
        out_b1[lab_] = b1_fit(resp[key][pm], T_last[pm], fold[pm], mprev[pm], "pl", boot=a.boot)
        show(lab_ + ", M_prev", out_b1[lab_])
    # checks: capped truth -> beta ~ 0; residual response -> same beta
    chk = b1_fit(cap[pm], T_last[pm], fold[pm], mprev[pm], "pl", boot=0)
    if not abs(chk["beta"]) < 1e-8 * max(1.0, np.abs(cap[pm]).max()):
        fail(f"algebra check failed: capped-truth beta {chk['beta']}")
    chk2 = b1_fit(y_x[pm] - cap[pm], T_last[pm], fold[pm], mprev[pm], "pl", boot=0)
    if not np.isclose(chk2["beta"], out_b1["primary"]["beta"], rtol=0, atol=1e-8):
        fail("residual-response identity check failed")
    lab_ctx = b1_fit(pse[pm], T_last[pm], fold[pm], mprev[pm], "pl", boot=0)
    print(f"   checks: capped truth beta {chk['beta']:.2e} (algebra); residual response beta equals primary; "
          f"stored pseudo as response (label context) beta {lab_ctx['beta']:+.4f}")

    # ---- B2
    print(f"\n== B2 (held-out within each original fold; piecewise-linear time basis, knots {B2_KNOTS} [proposed])")
    splits = make_splits(par[pm].tolist(), pf)
    check_splits(splits, pf)
    b2 = b2_run(par[pm], y_x[pm], T_last[pm], mprev[pm], pf, splits, boot=a.boot, min_half=a.min_half)
    for k_, v in b2["status"].items():
        if v != "ok":
            print(f"   {k_}: {v}")
    print(f"   coverage {b2['covered']}/{b2['n']} ({'full' if b2['full_coverage'] else 'PARTIAL: descriptive only'}); "
          f"held-out R2 time-only {b2['r2_0']:.4f}, + M_prev {b2['r2_1']:.4f}; dR2 {b2['dr2']:+.4f} "
          f"[{b2['lo']:+.4f}, {b2['hi']:+.4f}] {b2['interval_status']}")
    ic = iso_c
    splits_iso = {f: (np.array([p for p in a_ if p in set(par[ic].tolist())]),
                      np.array([p for p in b_ if p in set(par[ic].tolist())])) for f, (a_, b_) in splits.items()}
    iso = b2_isotonic(par[ic], y_x[ic], {"T_last": (T_last[ic], True), "pseudo": (pse[ic], True),
                                         "S_all": (F(f"S_all_{TAU0}")[ic], False),
                                         "M_all": (F(f"M_all_{TAU0}")[ic], False)}, pf, splits_iso, a.min_half)
    for k_, v in iso.items():
        print(f"   isotonic {k_:<7} (secondary; learned-output tracking) held-out R2 {v['r2']:.4f} "
              f"coverage {v['covered']}/{v['n']} clips {v['clips']}")

    # ---- gate
    g1 = gate_part(out_b1["primary"])
    g2 = gate_part(out_b1["alt"], sign=-1 if out_b1["primary"]["beta"] < 0 else +1) \
        if np.isfinite(out_b1["primary"]["beta"]) else "unavailable"
    if g2 == "pass" and not out_b1["primary"]["beta"] < 0:
        g2 = "fail"
    if not b2["full_coverage"] or b2["interval_status"] != "ok":
        g3 = "unavailable"
    else:
        g3 = "pass" if (b2["dr2"] > 0 and b2["lo"] > 0) else "fail"
    overall = "discuss B3" if (g1, g2, g3) == ("pass", "pass", "pass") else "stop (no association resolved / unavailable)"
    adj_note = ("adjusted beta {:+.4f} [{:+.4f}, {:+.4f}] vs unadjusted-same-cohort {:+.4f}".format(
        out_b1["adjusted"]["beta"], out_b1["adjusted"]["lo"], out_b1["adjusted"]["hi"],
        out_b1["adj_unadjusted_same"]["beta"]))
    print(f"\n== gate (practical; not a significance test): (i) {g1}; (ii) {g2}; (iii) {g3} -> {overall}; "
          f"sensitivity: {adj_note}")

    def pack(r):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
    out = dict(parent=par, fold=fold, T_last=T_last, primary=primary, finite_le4=finite_le4, adj_mask=adj,
               iso_mask=iso_c, exclusion=reason, logm=logm,
               feats=np.array(json.dumps([{k: v for k, v in f.items()} for f in feats], default=float)),
               responses=np.array(json.dumps({f"{k[0]}/{k[1]}": v.tolist() for k, v in resp.items()})),
               b1=np.array(json.dumps({k: pack(v) for k, v in out_b1.items()}, default=float)),
               b2=np.array(json.dumps(pack({k: v for k, v in b2.items()}), default=float)),
               b2_iso=np.array(json.dumps(iso, default=float)),
               splits=np.array(json.dumps({str(f): [a_.tolist(), b_.tolist()] for f, (a_, b_) in splits.items()})),
               gate=np.array(json.dumps(dict(i=g1, ii=g2, iii=g3, overall=overall))),
               provenance=np.array(json.dumps(prov, default=str)))
    np.savez(a.out, **out)
    print(f"\nsaved {a.out}")
    return out


if __name__ == "__main__":
    run()
