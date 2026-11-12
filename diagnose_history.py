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
    other, both directions; time-only vs + M_prev, OLS on a piecewise-linear
    basis with knots 0.5, 1, 2 Gyr (5 time-only columns, 6 with M_prev).
    AMENDED in review round 44: the agreed basis used the B1 knots, but after
    examining the T_last distribution only (before inspecting any response
    or association), 3 of 10 fit halves had no parent in (3, 4] and an
    all-zero (T-3)+ column. Each half n >= 20, both fits full rank, support
    rule on the calibration half, no automatic basis adaptation. Primary
    result only if all five folds and all ten directions pass AND every
    primary parent has exactly one valid held-out prediction. Held-out
    R2 of each and the paired parent bootstrap of the difference (stratified
    by fold). Secondary: 1-D isotonic rankings (T_last, pseudo increasing;
    S_all, M_all decreasing) on their own common cohort, same splits.
Modes: production (exactly seeds 42-46, 4000 draws, min half 20) or smoke
(reduced settings allowed, labelled in log and artifact, never nominates B3).
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
# B2 basis amendment, accepted in review round 44 (age distribution only, before
# any response/association): with KNOTS_PL, 3 of 10 real fit halves had no
# T_last in (3, 4], so (T-3)+ was an all-zero column.
B2_KNOTS = (0.5, 1.0, 2.0)
B2_AMENDMENT = ("B2 knots amended from the B1 knots to (0.5, 1, 2) in review round 44 after examining "
                "the T_last distribution only, before inspecting any response or association")
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
    """(coef, rank_ok, cond, rank) on unit-norm column-scaled X; coef is None
    unless the design is finite, non-empty and of full column rank."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or X.shape[0] != len(y):
        return None, False, np.inf, 0
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        return None, False, np.inf, 0
    norms = np.linalg.norm(X, axis=0)
    Xs = X / np.where(norms > 0, norms, 1.0)
    rank = int(np.linalg.matrix_rank(Xs, tol=RANK_TOL * max(Xs.shape)))
    cond = float(np.linalg.cond(Xs))
    if rank < X.shape[1] or (norms == 0).any():
        return None, False, cond, rank
    return np.linalg.lstsq(X, y, rcond=None)[0], True, cond, rank


def support(X_nuis, f):
    """(residual SD of f after the nuisance design / raw SD, status, residual vector)."""
    f = np.asarray(f, np.float64)
    if len(f) == 0:
        return np.nan, "empty", None
    sd = float(np.std(f))
    if not np.isfinite(sd) or sd == 0:
        return np.nan, "zero or non-finite raw SD", None
    c, ok, _, _ = ols(X_nuis, f)
    if not ok:
        return np.nan, "nuisance design rank-deficient", None
    resid = f - X_nuis @ c
    r = float(np.std(resid) / sd)
    return r, ("ok" if r >= SUPPORT_FRAC else "insufficient support under the prespecified rule"), resid


def r2_score(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    if len(y) < 2 or not np.var(y) > 0:
        return np.nan
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def b1_fit(y, T, fold, F, basis="pl", extra=None, boot=BOOT, seed=BOOT_SEED, ids=None):
    """B1 model and its fold-stratified parent bootstrap. The same keys are
    returned whether or not the fit is identifiable."""
    y, T, F = (np.asarray(v, np.float64) for v in (y, T, F))
    fold = np.asarray(fold, int)
    res = dict(n=len(y), ids=None if ids is None else [int(i) for i in ids], basis=basis, names=[],
               n_columns=0, coef=None, beta=np.nan, sd_ref=np.nan, lo=np.nan, hi=np.nan, status="ok",
               support=np.nan, support_status="", residual_feature=None, rank=0, rank_tol=RANK_TOL,
               scaling="unit-norm columns", col_norms=None, cond=np.nan, partial_r2=np.nan,
               partial_r2_status="", draws=boot, draws_valid=0, boot_betas=[], failed_draws=[],
               interval_status="")
    if len(y) == 0:
        res.update(status="unavailable: empty cohort", interval_status="unavailable: empty cohort",
                   partial_r2_status="undefined: empty cohort")
        return res
    Xb, nb = (pl_basis(T) if basis == "pl" else ncs_basis(T))
    Xf, nf = fold_indicators(fold)
    Xn = np.column_stack([Xb, Xf] + ([extra[0]] if extra is not None else []))
    X = np.column_stack([Xn, F])
    res.update(names=nb + nf + (list(extra[1]) if extra is not None else []) + ["F"], n_columns=X.shape[1],
               sd_ref=float(np.std(F)), col_norms=np.linalg.norm(X, axis=0).tolist())
    sup, sstat, resid = support(Xn, F)
    res.update(support=sup, support_status=sstat, residual_feature=resid)
    coef, ok, cond, rank = ols(X, y)
    res.update(cond=cond, rank=rank)
    if not ok:
        res.update(status="rank-deficient", interval_status="unavailable: fit not identifiable (rank-deficient)",
                   partial_r2_status="undefined: fit not identifiable")
        return res
    if sstat != "ok":
        res["status"] = sstat
    res["coef"] = coef.tolist()
    res["beta"] = float(coef[-1])
    cn, okn, _, _ = ols(Xn, y)
    if not np.var(y) > 0:
        res["partial_r2_status"] = "undefined: zero response variance"
    elif not okn:
        res["partial_r2_status"] = "undefined: nuisance fit not identifiable"
    else:
        sse_r = float(np.sum((y - Xn @ cn) ** 2))
        if not sse_r > 0:
            res["partial_r2_status"] = "undefined: zero nuisance residual sum of squares"
        else:
            res["partial_r2"] = (sse_r - float(np.sum((y - X @ coef) ** 2))) / sse_r
            res["partial_r2_status"] = "ok"
    rng = np.random.default_rng(seed)
    idx_f = [np.where(fold == f)[0] for f in range(L.N_FOLD)]
    bs, bad = [], []
    for b in range(boot):
        s = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_f if len(ix)])
        cb, okb, _, _ = ols(X[s], y[s])
        if not okb or not np.isfinite(cb[-1]):
            bad.append(b)
            continue
        bs.append(float(cb[-1]))
    res.update(draws_valid=len(bs), boot_betas=bs, failed_draws=bad)
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
        ps = np.array(sorted(int(p) for p in parents if parent_fold[int(p)] == f), dtype=int)
        perm = np.random.default_rng(SPLIT_SEED + f).permutation(len(ps))
        h = len(ps) // 2
        splits[f] = (ps[perm[:h]], ps[perm[h:]])
    return splits


def check_splits(splits, parent_fold, expected=None):
    """Halves: one original fold each, no duplicates, disjoint; with
    `expected`, together they cover exactly those parents once."""
    if sorted(splits) != list(range(L.N_FOLD)):
        fail(f"splits must be keyed by original folds 0-{L.N_FOLD - 1}, got {sorted(splits)}")
    seen = []
    for f, (a, b) in splits.items():
        for nm, h in (("A", a), ("B", b)):
            if len(set(np.asarray(h).tolist())) != len(h):
                fail(f"fold {f}: duplicate parent ids in half {nm}")
        if set(np.asarray(a).tolist()) & set(np.asarray(b).tolist()):
            fail(f"fold {f}: calibration and evaluation halves overlap")
        for p in list(a) + list(b):
            if parent_fold[int(p)] != f:
                fail(f"fold {f}: parent {p} belongs to original fold {parent_fold[int(p)]} (cross-fold calibration)")
        seen += [int(p) for p in list(a) + list(b)]
    if expected is not None and sorted(seen) != sorted(int(p) for p in expected):
        fail("split halves do not cover the analysis parents exactly once")


def b2_run(par, y, T, F, parent_fold, splits, boot=BOOT, seed=BOOT_SEED, min_half=MIN_HALF, knots=B2_KNOTS):
    """Held-out time-only and +F predictions per fold and direction. The
    primary requires all ten directions valid and exactly one held-out
    prediction for every parent."""
    par = np.asarray(par, int)
    if len(set(par.tolist())) != len(par):
        fail("B2: duplicate parents in the analysis cohort")
    check_splits(splits, parent_fold, expected=par)
    y, T, F = (np.asarray(v, np.float64) for v in (y, T, F))
    pos = {int(p): i for i, p in enumerate(par)}
    pred0, pred1 = np.full(len(par), np.nan), np.full(len(par), np.nan)
    npred = np.zeros(len(par), int)
    edges = [-np.inf] + list(knots) + [np.inf]
    dirs = []
    for f in range(L.N_FOLD):
        a, b = splits[f]
        for name, fit, ev in (("A->B", a, b), ("B->A", b, a)):
            fi = np.array([pos[int(p)] for p in fit], int)
            ei = np.array([pos[int(p)] for p in ev], int)
            rec = dict(fold=f, direction=name, fit_ids=[int(p) for p in fit], eval_ids=[int(p) for p in ev],
                       n_fit=len(fi), n_eval=len(ei), segment_counts_fit=np.histogram(T[fi], edges)[0].tolist(),
                       coef0=None, coef1=None, rank0=0, rank1=0, cond0=np.nan, cond1=np.nan,
                       support=np.nan, support_status="", status="")
            if len(fi) < min_half or len(ei) < min_half:
                rec["status"] = f"unavailable: n fit {len(fi)} / eval {len(ei)} < {min_half}"
                dirs.append(rec)
                continue
            X0f, _ = pl_basis(T[fi], knots)
            X1f = np.column_stack([X0f, F[fi]])
            c0, ok0, rec["cond0"], rec["rank0"] = ols(X0f, y[fi])
            c1, ok1, rec["cond1"], rec["rank1"] = ols(X1f, y[fi])
            rec["support"], rec["support_status"], _ = support(X0f, F[fi])
            if not (ok0 and ok1):
                rec["status"] = "unavailable: rank-deficient fit"
            elif rec["support_status"] != "ok":
                rec["status"] = f"unavailable: {rec['support_status']}"
            else:
                rec["coef0"], rec["coef1"] = c0.tolist(), c1.tolist()
                X0e, _ = pl_basis(T[ei], knots)
                pred0[ei] = X0e @ c0
                pred1[ei] = np.column_stack([X0e, F[ei]]) @ c1
                npred[ei] += 1
                rec["status"] = "ok"
            dirs.append(rec)
    all_dirs = all(r_["status"] == "ok" for r_ in dirs) and len(dirs) == 2 * L.N_FOLD
    m = np.isfinite(pred0) & np.isfinite(pred1)
    full = bool(len(par) > 0 and all_dirs and (npred == 1).all() and m.all())
    res = dict(parents=par.tolist(), directions=dirs, all_directions_ok=all_dirs, full_coverage=full,
               pred0=pred0, pred1=pred1, n_predictions=npred, covered=int(m.sum()), n=len(par),
               knots=list(knots), r2_0=np.nan, r2_1=np.nan, dr2=np.nan, lo=np.nan, hi=np.nan,
               draws=boot, draws_valid=0, boot_dr2=[], failed_draws=[], interval_status="")
    if m.any():
        res["r2_0"], res["r2_1"] = r2_score(y[m], pred0[m]), r2_score(y[m], pred1[m])
        res["dr2"] = res["r2_1"] - res["r2_0"]
    if not full:
        why = ("empty cohort" if len(par) == 0 else
               "not all ten fold directions valid" if not all_dirs else "incomplete or repeated coverage")
        res["interval_status"] = f"unavailable: primary needs full coverage ({why}; {res['covered']}/{len(par)})"
        return res
    if not np.isfinite(res["dr2"]):
        res["interval_status"] = "unavailable: held-out R2 undefined (zero response variance)"
        return res
    folds = np.array([parent_fold[int(p)] for p in par])
    rng = np.random.default_rng(seed)
    idx = [np.where(folds == f)[0] for f in range(L.N_FOLD)]
    bs, bad = [], []
    for b in range(boot):
        s = np.concatenate([rng.choice(ix, len(ix)) for ix in idx if len(ix)])
        d = r2_score(y[s], pred1[s]) - r2_score(y[s], pred0[s])
        if not np.isfinite(d):
            bad.append(b)
            continue
        bs.append(float(d))
    res.update(draws_valid=len(bs), boot_dr2=bs, failed_draws=bad)
    if boot == 0:
        res["interval_status"] = "not requested"
    elif bad:
        res["interval_status"] = f"unavailable: {len(bad)} of {boot} draws undefined"
    else:
        res["lo"], res["hi"] = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
        res["interval_status"] = "ok"
    return res


def b2_isotonic(par, y, summaries, parent_fold, splits, min_half=MIN_HALF):
    """Secondary 1-D isotonic rankings on the same splits (their own common
    cohort). Partial coverage is labelled descriptive."""
    par = np.asarray(par, int)
    check_splits(splits, parent_fold, expected=par)
    y = np.asarray(y, np.float64)
    pos = {int(p): i for i, p in enumerate(par)}
    out = {}
    for name, (x, increasing) in summaries.items():
        x = np.asarray(x, np.float64)
        pred = np.full(len(par), np.nan)
        clips, dirs = 0, []
        for f in range(L.N_FOLD):
            a, b = splits[f]
            for d_, fit, ev in (("A->B", a, b), ("B->A", b, a)):
                fi = np.array([pos[int(p)] for p in fit], int)
                ei = np.array([pos[int(p)] for p in ev], int)
                rec = dict(fold=f, direction=d_, fit_ids=[int(p) for p in fit], eval_ids=[int(p) for p in ev],
                           x_thresholds=None, y_thresholds=None, clips=0, status="")
                if len(fi) < min_half or len(ei) < min_half:
                    rec["status"] = f"unavailable: n fit {len(fi)} / eval {len(ei)} < {min_half}"
                elif not (np.isfinite(x[fi]).all() and np.isfinite(y[fi]).all() and np.isfinite(x[ei]).all()):
                    rec["status"] = "unavailable: non-finite summary or response"
                else:
                    iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(x[fi], y[fi])
                    pred[ei] = iso.predict(x[ei])
                    rec["clips"] = int(((x[ei] < x[fi].min()) | (x[ei] > x[fi].max())).sum())
                    rec.update(x_thresholds=iso.X_thresholds_.tolist(), y_thresholds=iso.y_thresholds_.tolist(),
                               status="ok")
                    clips += rec["clips"]
                dirs.append(rec)
        m = np.isfinite(pred)
        r2 = r2_score(y[m], pred[m]) if m.any() else np.nan
        cov = ("empty cohort" if len(par) == 0 else
               "full" if m.all() and all(r_["status"] == "ok" for r_ in dirs) else "partial: descriptive only")
        out[name] = dict(increasing=increasing, parents=par.tolist(), pred=pred, r2=r2,
                         r2_status="ok" if np.isfinite(r2) else "undefined (empty or zero variance)",
                         covered=int(m.sum()), n=len(par), coverage=cov, clips=clips, directions=dirs)
    return out


def _unav(status):
    return status if status.startswith("unavailable") else f"unavailable: {status}"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    return o


# ---------------------------------------------------------------- main
def run(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-manifest", required=True)
    ap.add_argument("--mode", choices=("production", "smoke"), default="production")
    ap.add_argument("--pred-dir", default="cnn_preds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, L.PRODUCTION_SEEDS)))
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--min-half", type=int, default=MIN_HALF)
    for k in ("mergers", "catalog"):
        ap.add_argument(f"--{k}", default=D.DEFAULTS[k])
    a = ap.parse_args(argv)
    if not a.out.endswith(".npz"):
        fail(f"--out must end in .npz (numpy would otherwise append it): {a.out}")
    if os.path.exists(a.out):
        fail(f"refusing to overwrite {a.out}")
    try:
        seeds = L.parse_seeds(a.seeds, a.mode)
    except ValueError:
        fail(f"seed list must be comma-separated integers: {a.seeds!r}")
    if a.boot < 0 or a.min_half < 2:
        fail(f"--boot must be >= 0 and --min-half >= 2 (got {a.boot}, {a.min_half})")
    if a.mode == "production" and (a.boot != BOOT or a.min_half != MIN_HALF):
        fail(f"production requires --boot {BOOT} and --min-half {MIN_HALF}; use --mode smoke for reduced settings")
    smoke = a.mode == "smoke"
    tagp = "SMOKE (not scientific)" if smoke else "PRIMARY"
    if smoke:
        print(f"*** SMOKE MODE: seeds {seeds}, {a.boot} draws, min half {a.min_half}; "
              "component calculations only, no B3 nomination ***")
    here = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    prov = dict(head=head, mode=a.mode, args=vars(a),
                sources={f: D.sha(os.path.join(here, f)) for f in
                         ("diagnose_history.py", "diagnose_tsc_hardness.py", "compare_labels.py", "compare_e0.py")},
                inputs={k: D.sha(getattr(a, k)) for k in ("mergers", "catalog")},
                constants=dict(tau0=TAU0, tau_sens=TAU_SENS, knots_pl=KNOTS_PL, b2_knots=B2_KNOTS,
                               b2_amendment=B2_AMENDMENT, ncs_knots=NCS_KNOTS, boot=a.boot, boot_seed=BOOT_SEED,
                               split_seed=SPLIT_SEED, min_half=a.min_half, support_frac=SUPPORT_FRAC,
                               rank_tol=RANK_TOL))

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
    fold = np.array([pf[int(p)] for p in par], dtype=int)

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
    sd_m = float(np.std(mprev[primary])) if primary.any() else np.nan
    print(f"cohort: {len(par)} snapshot-99 parents; finite 0<=T_last<=4: {int(finite_le4.sum())}; primary "
          f"(M_prev available): {int(primary.sum())}; excluded: "
          + ", ".join(f"{r_} {int((reason == r_).sum())}" for r_ in ("no recorded event", "T_last > 4",
                                                                      "invalid prev ratio"))
          + f"; adjustment-eligible {int(adj.sum())}; isotonic common cohort {int(iso_c.sum())}")
    print(f"   prev events: parents with any prev {int((F('n_prev') > 0)[primary].sum())}; latest tie groups > 1: "
          f"{int((F('n_latest') > 1)[primary].sum())}; M_prev(1) raw SD {sd_m:.4g}")

    # ---- B1
    out_b1 = {}
    pm = primary
    y_x = resp[("x", "tsc_tcoll")]
    def show(lab, r):
        print(f"   {lab:<44} n {r['n']:>3} beta {r['beta']:+.4f} (one-SD {r['beta'] * r['sd_ref']:+.4f}) "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {r['interval_status']}; status {r['status']}; "
              f"support {r['support']:.3f}; rank {r['rank']}/{r['n_columns']}; cond {r['cond']:.3g}; "
              f"partial R2 {r['partial_r2']:.4f}")
    def fit(yv, mask, feat, basis="pl", extra=None, boot=a.boot):
        return b1_fit(yv[mask], T_last[mask], fold[mask], feat[mask], basis,
                      extra=None if extra is None else (extra[0][mask], extra[1]), boot=boot, ids=par[mask])
    print("\n== B1 (parent-level OLS, original-fold intercepts; raw beta per unit of M_prev)")
    out_b1["primary"] = fit(y_x, pm, mprev)
    show(f"{tagp} x/tsc_tcoll, piecewise linear, M_prev", out_b1["primary"])
    out_b1["alt"] = fit(y_x, pm, mprev, "ncs")
    show("x/tsc_tcoll, natural cubic spline, M_prev", out_b1["alt"])
    ex = (np.column_stack([logm, F("latest_strength"), F("n_latest")]), ["logM500", "latest_strength", "n_latest"])
    out_b1["adj_unadjusted_same"] = fit(y_x, adj, mprev)
    show("x/tsc_tcoll, unadjusted on adjustment cohort", out_b1["adj_unadjusted_same"])
    out_b1["adjusted"] = fit(y_x, adj, mprev, extra=ex)
    show("x/tsc_tcoll, + logM500, latest strength, ties", out_b1["adjusted"])
    for tau in TAU_SENS:
        out_b1[f"tau{tau}"] = fit(y_x, pm, F(f"M_prev_{tau}"))
        show(f"x/tsc_tcoll, M_prev tau={tau} (sensitivity)", out_b1[f"tau{tau}"])
    out_b1["S_prev"] = fit(y_x, pm, F(f"S_prev_{TAU0}"))
    show("x/tsc_tcoll, S_prev (secondary)", out_b1["S_prev"])
    for lab_, key in (("r/tsc_tcoll (secondary)", ("r", "tsc_tcoll")), ("x/pseudo (context)", ("x", "pseudo")),
                      ("r/pseudo (context)", ("r", "pseudo"))):
        out_b1[lab_] = fit(resp[key], pm, mprev)
        show(lab_ + ", M_prev", out_b1[lab_])

    # checks: capped truth -> beta ~ 0; residual response -> same beta (only when identifiable)
    checks = {}
    chk = fit(cap, pm, mprev, boot=0)
    tol_a = 1e-8 * max(1.0, float(np.abs(cap[pm]).max()) if pm.any() else 1.0)
    if not np.isfinite(chk["beta"]):
        checks["algebra"] = dict(status=_unav(chk["status"]), beta=chk["beta"], tol=tol_a, fit=chk)
    elif not abs(chk["beta"]) < tol_a:
        fail(f"algebra check failed: capped-truth beta {chk['beta']}")
    else:
        checks["algebra"] = dict(status="ok", beta=chk["beta"], tol=tol_a, fit=chk)
    chk2 = fit(y_x - cap, pm, mprev, boot=0)
    if not (np.isfinite(chk2["beta"]) and np.isfinite(out_b1["primary"]["beta"])):
        checks["residual_identity"] = dict(status=_unav(chk2["status"]), beta=chk2["beta"],
                                           primary_beta=out_b1["primary"]["beta"], fit=chk2)
    elif not np.isclose(chk2["beta"], out_b1["primary"]["beta"], rtol=0, atol=1e-8):
        fail("residual-response identity check failed")
    else:
        checks["residual_identity"] = dict(status="ok", beta=chk2["beta"], primary_beta=out_b1["primary"]["beta"],
                                           fit=chk2)
    checks["label_context"] = fit(pse, pm, mprev, boot=0)
    print(f"   checks: algebra {checks['algebra']['status']} (capped truth beta {chk['beta']:.2e}); residual identity "
          f"{checks['residual_identity']['status']}; stored pseudo as response (label context) beta "
          f"{checks['label_context']['beta']:+.4f} ({checks['label_context']['status']})")

    # ---- B2
    print(f"\n== B2 (held-out within each original fold; piecewise-linear time basis, knots {B2_KNOTS}, "
          f"amended in round 44)")
    splits = make_splits(par[pm].tolist(), pf)
    b2 = b2_run(par[pm], y_x[pm], T_last[pm], mprev[pm], pf, splits, boot=a.boot, min_half=a.min_half)
    for r_ in b2["directions"]:
        print(f"   fold{r_['fold']} {r_['direction']}: n {r_['n_fit']}/{r_['n_eval']}; fit-half segment counts "
              f"{r_['segment_counts_fit']}; rank {r_['rank0']}/{r_['rank1']}; cond {r_['cond0']:.3g}/{r_['cond1']:.3g}; "
              f"support {r_['support']:.3f}; {r_['status']}")
    print(f"   {tagp}: coverage {b2['covered']}/{b2['n']} "
          f"({'full' if b2['full_coverage'] else 'NOT full: descriptive only'}); held-out R2 time-only "
          f"{b2['r2_0']:.4f}, + M_prev {b2['r2_1']:.4f}; dR2 {b2['dr2']:+.4f} [{b2['lo']:+.4f}, {b2['hi']:+.4f}] "
          f"{b2['interval_status']}")
    ic_set = set(par[iso_c].tolist())
    splits_iso = {f: (np.array([p for p in a_ if p in ic_set], int), np.array([p for p in b_ if p in ic_set], int))
                  for f, (a_, b_) in splits.items()}
    iso = b2_isotonic(par[iso_c], y_x[iso_c], {"T_last": (T_last[iso_c], True), "pseudo": (pse[iso_c], True),
                                               "S_all": (F(f"S_all_{TAU0}")[iso_c], False),
                                               "M_all": (F(f"M_all_{TAU0}")[iso_c], False)}, pf, splits_iso,
                      a.min_half)
    for k_, v in iso.items():
        print(f"   isotonic {k_:<7} (secondary; learned-output tracking) held-out R2 {v['r2']:.4f} "
              f"coverage {v['covered']}/{v['n']} ({v['coverage']}) clips {v['clips']}")

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
    if smoke:
        overall = "smoke run: components exercised only; no B3 nomination"
    else:
        overall = ("discuss B3" if (g1, g2, g3) == ("pass", "pass", "pass")
                   else "stop (no association resolved / unavailable)")
    adj_note = ("adjusted beta {:+.4f} [{:+.4f}, {:+.4f}] ({}) vs unadjusted-same-cohort {:+.4f}".format(
        out_b1["adjusted"]["beta"], out_b1["adjusted"]["lo"], out_b1["adjusted"]["hi"],
        out_b1["adjusted"]["status"], out_b1["adj_unadjusted_same"]["beta"]))
    print(f"\n== gate (practical; not a significance test){' [SMOKE]' if smoke else ''}: (i) {g1}; (ii) {g2}; "
          f"(iii) {g3} -> {overall}; sensitivity: {adj_note}")

    J = lambda o: np.array(json.dumps(jsonable(o)))
    out = dict(mode=np.array(a.mode), parent=par, fold=fold, T_last=T_last, primary=primary, finite_le4=finite_le4,
               adj_mask=adj, iso_mask=iso_c, exclusion=reason, logm=logm,
               feats=J(feats), responses=J({f"{k[0]}/{k[1]}": v for k, v in resp.items()}),
               b1=J(out_b1), checks=J(checks), b2=J(b2), b2_iso=J(iso),
               splits=J({str(f): [a_, b_] for f, (a_, b_) in splits.items()}),
               splits_iso=J({str(f): [a_, b_] for f, (a_, b_) in splits_iso.items()}),
               gate=J(dict(i=g1, ii=g2, iii=g3, overall=overall, mode=a.mode)),
               provenance=np.array(json.dumps(jsonable(prov), default=str)))
    np.savez(a.out, **out)
    print(f"\nsaved {a.out}")
    return out


if __name__ == "__main__":
    run()
