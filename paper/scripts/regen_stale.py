#!/usr/bin/env python3
"""Regenerate the manuscript items marked \\stale{} (review rounds 48-50).

Run from the repository root, under SLURM:
    ./venv/bin/python paper/scripts/regen_stale.py --run-dir paper/staged/<new dir> \
        --tasks w1,w4,w5,w6,w7,w8,w9            (job C)
    ./venv/bin/python paper/scripts/regen_stale.py --run-dir paper/staged/<new dir> \
        --tasks w2,w3 --phase-npz <job A output>  (job D, after jobs A and B)

Every task fingerprints its inputs BEFORE reading them, validates them against
the declared protocol, and writes fig_<name>.pdf (where a figure exists),
<task>.json (numbers, estimands, checks, input fingerprints, exact command) and
<task>_arrays.npz (identity-aligned arrays behind every number and plot) into
the run directory. manifest.json is written last: each task's status, whether
any helper or input changed during the run, and `promotable` = every task ok,
no source changed and no uncommitted helper. The run directory must not exist
beforehand; nothing outside it is written.

Tasks
  w1  clean CNN, scalar/mass baselines (outer CV on each seed's reconstructed
      CNN partition), within-original-fold calibration experiment and
      held-out tracking diagnostics, mass terciles, partial rank correlation
  w2  Minkowski-functional XGBoost at 144 MHz and (paired) 1.4 GHz
  w3  phase diagram / gas-phase fractions (job A) and the 144 MHz vs 1.4 GHz
      comparison (jobs A and B)
  w4  aperture retention (1 and 3 Mpc boxes)
  w5  radio gallery and realism curves (check_radio_realism definition)
  w6  predicted vs true pseudo-TSC and age-binned residuals
  w7  power-mass relation (log of arithmetic-mean linear power)
  w8  LoVoCCS radio vs X-ray observed predictions
  w9  projected outermost-peak diagnostic (bounded redetection)
"""
import argparse
import glob
import inspect
import json
import os
import re
import subprocess
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.getcwd())
import prov_utils as PU

COL, TEXT = 3.32, 6.97
C_SIM, C_OBS = "#3b6ea8", "#c2455d"
SEEDS = [42, 43, 44, 45, 46]
N_FOLD = 5
ALPHAS = np.logspace(-3, 3, 13)
CAL_SEED, BOOT_SEED, BOOT = 20261113, 20261113, 4000
MIN_CAL = 15
TSC_CAP = 4.0
RESID_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
TOL_CROP_DEX, TOL_REAL_DEX = 1e-3, 1e-5
TOL_LABEL, TOL_LOGM, TOL_RATIO = 1e-6, 1e-5, 1e-6
TOL_CELL_RATIO = 1e-9
# declared map protocol (validated explicitly, never by equality of two files)
PROTO = dict(snapshot=99, redshift=0.0, center_mode="grouppos", extent_r500=4.0, spread_cells=False,
             n_clusters=352, n_projections=3, projections=["xy", "yz", "xz"])
PHASE_CONFIG = dict(snapshot=99, z=0.0, model="synch", nu_lo_ghz=0.144, nu_hi_ghz=1.4, nh_cut=1e-4,
                    nh_cut_rule="kept: n_H <= nh_cut", t_cold_kev=0.1, w_scale_cells=206.0,
                    shock="Machnumber > 1.3 & EnergyDissipation > 0", nh_edges=[-7.0, 0.0, 71],
                    t_edges=[-3.5, 1.5, 51], normalisation="per parent by its whole valid population")
PHASE_PER_PARENT = ("halo_id", "n_cells", "n_invalid", "n_valid", "n_zero_w", "w_total", "n_out_of_range",
                    "w_out_of_range", "n_kept", "w_kept", "frac_above", "frac_cold", "top_all", "top_cut",
                    "mach_all", "mach_cut", "ratio_all", "ratio_cut", "check_cut_n", "check_cut_wrel",
                    "check_all_n", "cut_status")
DEPS = {"w2": ("B",), "w3": ("A", "B")}
B_PRESCRIPTION = dict(model="synch", nu_ghz=1.4, w_scale=16.48, max_nh=1e-4, snapshot=99, z=0.0,
                      spread_cells=False)

IN = dict(ds128="dataset_syn_nh4_144_128.h5", ds512="dataset_syn_nh4_144_512.h5",
          shape128="dataset_syn_nh4_144_128_shape.h5", ds1400="dataset_syn_nh4_1400_128.h5",
          inj3="injected_syn_phys_b3000.h5", inj1="injected_syn_phys_b1000.h5",
          catalog="Radio_Data/TNG-Cluster_Catalog.hdf5", relic_cat="relic_catalog_radio_syn_nh4_144.h5",
          tsc_legacy="TSC_Cutimages/TSC_eachhalo_snap99.hdf5", ds_legacy="dataset.h5",
          wl="lovoccs_wl_masses.csv", ratio_table="Radio_Data_new/radio_spectral_ratio_table.npz",
          cells144="Radio_Cells_syn_nh4_144", cells1400="Radio_Cells_syn_nh4_1400")
PRED = dict(clean="cnn_preds/pooled_syn_inner_s{}.npz", shape="cnn_preds/pooled_shape_syn_inner_s{}.npz",
            joint1="syn_phys_b1000_joint_s{}_preds.npz", radio3="syn_phys_b3000_sfimg_s{}_preds.npz",
            xray="fix_x_s{}_preds.npz")
HELPERS = ["paper/scripts/regen_stale.py", "prov_utils.py", "minkowski_functionals.py", "check_radio_realism.py",
           "detect_relics_radio.py", "train_cnn_pooled.py", "compare_syn.py", "forward_model_lotss.py",
           "extract_phase_syn.py", "manifest_cells_1400.py", "build_radio_cells.py", "build_dataset.py",
           "build_injected_dataset.py"]


class Fail(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


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
    if isinstance(o, bytes):
        return o.decode()
    return o


def attr(v):
    v = v.tolist() if hasattr(v, "tolist") else v
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, list):
        return [x.decode() if isinstance(x, bytes) else x for x in v]
    return v


# ======================================================================= pure helpers
def r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    if len(y) < 2 or not np.var(y) > 0:
        return np.nan
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def validate_maps(f, img_size, n=None):
    """Explicit check of one map file against the declared protocol."""
    n = PROTO["n_clusters"] if n is None else n
    at = {k: attr(f.attrs[k]) for k in f.attrs}
    for k, v in PROTO.items():
        exp = n if k == "n_clusters" else v
        check(k in at, f"{f.filename}: attribute {k} missing")
        check(at[k] == exp if not isinstance(exp, float) else abs(float(at[k]) - exp) < 1e-12,
              f"{f.filename}: {k} = {at[k]!r}, expected {exp!r}")
    check(at.get("img_size") == img_size, f"{f.filename}: img_size {at.get('img_size')} != {img_size}")
    check(f["images"].shape == (n, 3, img_size, img_size), f"{f.filename}: images shape {f['images'].shape}")
    hid = f["meta/halo_id"][:]
    check(len(set(hid.tolist())) == len(hid), f"{f.filename}: duplicate halo ids")
    if "meta/halo_id_snap" in f:
        check(np.array_equal(f["meta/halo_id_snap"][:], hid), f"{f.filename}: snapshot ids differ from z=0 ids")
    return at, hid


def reconstruct_folds(n, seed):
    """Outer fold of each row for train_cnn_pooled with one row per parent
    (groups = arange(n) in the saved row order), via its own group_split."""
    from sklearn.model_selection import KFold
    from train_cnn_pooled import group_split
    fold = np.full(n, -1)
    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(group_split(np.arange(n), np.arange(n), kf.split)):
        fold[te] = f
    return fold


def calib_halves(ids, seed, f):
    """Fixed balanced halves of one original fold's parents (sorted ids)."""
    ids = np.array(sorted(int(i) for i in ids))
    perm = np.random.default_rng(CAL_SEED + 1000 * seed + f).permutation(len(ids))
    h = len(ids) // 2
    return ids[perm[:h]], ids[perm[h:]]


def ridge_fit_predict(Xtr, ytr, Xte, inner_seed):
    """One declared ridge procedure: scaler inside the pipeline, alpha by
    5-fold inner CV (KFold shuffle, random_state=inner_seed, over the fitting
    rows in the order given) on the fitting rows only (13-value grid)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte = np.atleast_2d(np.asarray(Xtr, float).T).T, np.atleast_2d(np.asarray(Xte, float).T).T
    gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                      cv=KFold(5, shuffle=True, random_state=inner_seed), scoring="neg_mean_squared_error")
    gs.fit(Xtr, ytr)
    est = gs.best_estimator_
    return gs.predict(Xte), float(gs.best_params_["ridge__alpha"]), \
        dict(coef=est[-1].coef_.tolist(), intercept=float(est[-1].intercept_),
             mean=est[0].mean_.tolist(), scale=est[0].scale_.tolist(), inner_seed=int(inner_seed))


def outer_cv(X, y, fold, seed, ids):
    """Ordinary outer CV on a given partition; inner alpha CV in training folds."""
    pred, recs = np.full(len(y), np.nan), []
    for f in range(N_FOLD):
        tr, te = fold != f, fold == f
        p, a, model = ridge_fit_predict(X[tr], y[tr], X[te], inner_seed=seed * 10 + f)
        pred[te] = p
        recs.append(dict(fold=f, alpha=a, fit_ids=np.asarray(ids)[tr].tolist(), **model))
    return pred, recs


def within_fold_calibration(ids, y, fold, arms, seed, min_n=MIN_CAL):
    """Second-stage fits confined to halves of ONE original outer fold (the
    first-stage CNN saw neither half). arms: name -> (n, k) feature matrix.
    An arm is available only with every direction valid and exactly one
    finite held-out prediction per parent."""
    ids = np.asarray(ids, int)
    pos = {int(p): i for i, p in enumerate(ids)}
    pred = {a: np.full(len(ids), np.nan) for a in arms}
    npred = {a: np.zeros(len(ids), int) for a in arms}
    dirs = []
    for f in range(N_FOLD):
        A, B = calib_halves(ids[fold == f], seed, f)
        for name, fit, ev in (("A->B", A, B), ("B->A", B, A)):
            fi = np.array([pos[int(p)] for p in fit], int)
            ei = np.array([pos[int(p)] for p in ev], int)
            rec = dict(fold=f, direction=name, fit_ids=fit.tolist(), eval_ids=ev.tolist(), arms={}, status="ok")
            if len(fi) < min_n or len(ei) < min_n:
                rec["status"] = f"unavailable: n fit {len(fi)} / eval {len(ei)} < {min_n}"
                dirs.append(rec)
                continue
            for a, X in arms.items():
                X = np.atleast_2d(np.asarray(X, float).T).T
                p, alpha, model = ridge_fit_predict(X[fi], y[fi], X[ei], inner_seed=CAL_SEED + 100 * seed + f)
                pred[a][ei] = p
                npred[a][ei] += 1
                rec["arms"][a] = dict(alpha=alpha, **model)
            dirs.append(rec)
    full = {a: bool(all(d["status"] == "ok" for d in dirs) and (npred[a] == 1).all()
                    and np.isfinite(pred[a]).all()) for a in arms}
    return pred, full, dirs


def partial_spearman(a, b, c):
    """Pearson correlation of the rank residuals of a and b after linear
    regression on the rank of c (descriptive association)."""
    from scipy.stats import rankdata
    ra, rb, rc = (rankdata(v) for v in (a, b, c))
    Z = np.column_stack([np.ones(len(rc)), rc])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def tv_distance(m1, m2):
    """0.5 sum |p1 - p2| for unit-sum linear maps; NaN if either total <= 0."""
    m1, m2 = np.asarray(m1, np.float64), np.asarray(m2, np.float64)
    t1, t2 = m1.sum(), m2.sum()
    if not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0):
        return np.nan
    return float(0.5 * np.abs(m1 / t1 - m2 / t2).sum())


def overlap_weights(n, lo, hi):
    """Fractional overlap of pixels [k, k+1) with [lo, hi], k = 0..n-1."""
    k = np.arange(n, dtype=np.float64)
    return np.clip(np.minimum(k + 1, hi) - np.maximum(k, lo), 0.0, 1.0)


def box_sum(lin, half_px):
    """Sum of a linear map inside the centred square of half-width half_px
    pixels, splitting edge pixels by area (independent of forward_model_lotss)."""
    H, W = lin.shape
    oy = overlap_weights(H, H / 2 - half_px, H / 2 + half_px)
    ox = overlap_weights(W, W / 2 - half_px, W / 2 + half_px)
    return float(oy @ lin @ ox)


def collapse_realisations(logp, n=None, R=3, P=3, tol=TOL_REAL_DEX):
    """(N, R, P) box log-powers must be finite and repeat across every
    realisation; returns (N, P) and the maximum difference."""
    logp = np.asarray(logp, np.float64)
    n = logp.shape[0] if n is None else n
    check(logp.shape == (n, R, P), f"box powers have shape {logp.shape}, expected {(n, R, P)}")
    check(np.isfinite(logp).all(), f"{int((~np.isfinite(logp)).sum())} non-finite box powers")
    d = float(np.max(np.abs(logp - logp[:, :1, :])))
    check(d <= tol, f"box power differs across realisations by {d:.2e} dex (> {tol})")
    return logp[:, 0, :], d


def residual_bins(y, p, edges=RESID_EDGES):
    """Median and 16-84 of p - y per bin; last bin closed at the top (4 Gyr)."""
    y, r = np.asarray(y, float), np.asarray(p, float) - np.asarray(y, float)
    check(np.isfinite(y).all() and np.isfinite(r).all(), "non-finite labels or predictions")
    check(not ((y > edges[-1]).any() or (y < edges[0]).any()), "label outside the binning range")
    idx = np.clip(np.digitize(y, edges[1:-1], right=False), 0, len(edges) - 2)
    out = []
    for k in range(len(edges) - 1):
        m = idx == k
        lab = f"[{edges[k]:g}, {edges[k + 1]:g}{']' if k == len(edges) - 2 else ')'}"
        if not m.any():
            out.append(dict(bin=lab, n=0, status="unavailable: empty bin"))
            continue
        out.append(dict(bin=lab, n=int(m.sum()), median=float(np.median(r[m])),
                        p16=float(np.percentile(r[m], 16)), p84=float(np.percentile(r[m], 84)), status="ok"))
    return out


def cap_stats(y, p, cap=TSC_CAP):
    """Statistic for parents stored EXACTLY at the cap (no tolerance)."""
    y, p = np.asarray(y), np.asarray(p, float)
    m = y == np.asarray(cap, dtype=y.dtype)
    return dict(n=int(m.sum()), median_residual=float(np.median(p[m] - y[m])) if m.any() else None,
                rule="stored label == cap exactly")


def halo_summary(d_max):
    """(N, P) outermost-peak distance, NaN = no peak: mean over projections
    with a peak; halos with none are excluded (count returned)."""
    d = np.asarray(d_max, float)
    n_with = np.isfinite(d).sum(1)
    m = np.full(len(d), np.nan)
    ok = n_with > 0
    m[ok] = np.where(np.isfinite(d[ok]), d[ok], 0).sum(1) / n_with[ok]
    return m, n_with


def boot_spearman(x, y, n=BOOT, seed=BOOT_SEED):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearmanr(x, y)[0]) if len(x) > 2 else np.nan
    rng = np.random.default_rng(seed)
    bs, bad = [], 0
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        with np.errstate(invalid="ignore", divide="ignore"):
            v = spearmanr(x[i], y[i])[0] if len(x) > 2 else np.nan
        if not np.isfinite(v):
            bad += 1
            continue
        bs.append(v)
    ok = bad == 0 and len(bs) > 0
    return dict(n=len(x), rho=rho, lo=float(np.percentile(bs, 2.5)) if ok else np.nan,
                hi=float(np.percentile(bs, 97.5)) if ok else np.nan, draws=n, invalid_draws=bad,
                interval_status="ok" if ok else f"unavailable: {bad} invalid draws")


def norm_key(s):
    return re.sub(r"[\s_]+", "", s.decode() if isinstance(s, bytes) else str(s)).upper()


def legacy_join(hid, keys, values, name):
    """Identity join to a legacy label source: keys must be unique and cover
    every requested parent; a stored NaN is preserved as a valid value."""
    keys = [int(k) for k in keys]
    check(len(set(keys)) == len(keys), f"{name}: duplicate keys in the source")
    lut = dict(zip(keys, np.asarray(values, float)))
    missing = [int(h) for h in hid if int(h) not in lut]
    check(not missing, f"{name}: {len(missing)} requested parents missing from the source (e.g. {missing[:3]})")
    return np.array([lut[int(h)] for h in hid])


def relic_agreement(cat, d_max, n_rel, mt, mr, mr_now, hid):
    """Element-wise agreement of a redetection with the stored catalogue.
    NaN is the only supported missing/no-peak value; any infinity, shape
    mismatch or finite-mask mismatch is a disagreement. Every finite entry is
    compared. The caller saves both diagnostics and then fails on any False."""
    N = len(hid)
    shapes = dict(d_max=(N, 3), n_relics=(N, 3), merger_tsc=(N,), mass_ratio=(N,))
    agree = dict(shapes=all(np.shape(cat[k]) == v for k, v in shapes.items())
                 and np.shape(d_max) == (N, 3) and np.shape(n_rel) == (N, 3)
                 and np.shape(mt) == (N,) and np.shape(mr) == (N,) and np.shape(mr_now) == (N,),
                 halo_order=bool(np.array_equal(cat["halo_id"], hid)),
                 no_infinities=not any(np.isinf(np.asarray(v, float)).any()
                                       for v in (cat["d_max"], d_max, cat["merger_tsc"], mt, cat["mass_ratio"], mr,
                                                 mr_now)))
    if not agree["shapes"]:
        return agree
    same = lambda u, v, atol, rtol: bool(np.array_equal(np.isfinite(u), np.isfinite(v))
                                         and np.allclose(np.asarray(u, float)[np.isfinite(u)],
                                                         np.asarray(v, float)[np.isfinite(v)], rtol=rtol, atol=atol))
    agree.update(d_max=same(cat["d_max"], d_max, 0, 1e-6),
                 n_relics=bool(np.array_equal(cat["n_relics"], n_rel)),
                 merger_tsc=same(cat["merger_tsc"], mt, 1e-6, 0),
                 mass_ratio=same(cat["mass_ratio"], mr, TOL_RATIO, 0),
                 legacy_vs_current_mass_ratio=same(mr, mr_now, TOL_RATIO, 0))
    return agree


def validate_phase(path, expected_ids, ratio_sha, cells144_digest):
    """Job A artifact: success manifest with matching hash; production-complete
    with sources verified unchanged; every declared configuration value; the
    adopted ratio table and 144 MHz cell directory; exactly the expected unique
    parents; EVERY per-parent array of length N (then reindexed to the cohort
    order); histogram dimensions and edges as declared."""
    check(path and os.path.exists(path), f"phase artifact {path} missing")
    man_p = path + ".manifest.json"
    check(os.path.exists(man_p), f"no success manifest {man_p}: not a completed job A product")
    man = json.load(open(man_p))
    check(man.get("status") == "complete" and man.get("artifact_sha256") == PU.sha(path)
          and man.get("sources_verified_unchanged") is True,
          "job A manifest is not complete, not source-verified, or its hash does not match the artifact")
    d = dict(np.load(path, allow_pickle=True))
    pv = json.loads(str(d["provenance"]))
    check(pv.get("production_complete") is True and int(pv.get("limit", -1)) == 0
          and pv.get("sources_verified_unchanged") is True, "job A artifact is a smoke/limited or failed product")
    check(pv.get("n_check_failures") == 0, "job A reported cell-file check failures")
    cfg = pv.get("config", {})
    for k, v in PHASE_CONFIG.items():
        ok_ = k in cfg and (abs(float(cfg[k]) - v) < 1e-12 if isinstance(v, float) else
                            (len(cfg[k]) == len(v) and np.allclose(cfg[k], v)) if isinstance(v, list) else cfg[k] == v)
        check(ok_, f"job A config {k} = {cfg.get(k)!r}, expected {v!r}")
    check(pv.get("ratio_table") == ratio_sha, "job A used a different ratio table")
    check(pv.get("cells_cut", {}).get("digest") == cells144_digest,
          "job A's reference 144 MHz cell directory differs from the current one")
    ids = [int(h) for h in d["halo_id"]]
    n = len(ids)
    check(len(set(ids)) == n, "job A artifact has duplicate parents")
    exp = [int(h) for h in expected_ids]
    check(set(ids) == set(exp) and n == len(exp),
          f"job A parents differ from the adopted cohort ({len(set(ids) ^ set(exp))} mismatched)")
    check(int(pv.get("n_parents", -1)) == n, "job A n_parents disagrees with its arrays")
    for k in PHASE_PER_PARENT:
        check(k in d and np.shape(d[k]) == (n,), f"job A array {k} missing or not of length {n}")
    ne, te = np.linspace(*PHASE_CONFIG["nh_edges"]), np.linspace(*PHASE_CONFIG["t_edges"])
    check(np.allclose(d["nh_edges"], ne) and np.allclose(d["t_edges"], te), "job A histogram edges differ")
    check(d["H_n"].shape == (n, len(ne) - 1, len(te) - 1) and d["H_w"].shape == d["H_n"].shape,
          "job A histogram dimensions inconsistent")
    order = np.array([ids.index(h) for h in exp])
    out = {k: d[k][order] for k in PHASE_PER_PARENT + ("H_n", "H_w")}
    out.update(nh_edges=d["nh_edges"], t_edges=d["t_edges"])
    return out, pv, man


def validate_b_manifest(ds_path, cells_digest, ratio_sha):
    man_p = ds_path + ".manifest.json"
    check(os.path.exists(man_p), f"no success manifest {man_p}: not a completed job B product")
    man = json.load(open(man_p))
    check(man.get("status") == "complete" and man.get("kind") == "cells_1400"
          and man.get("sources_verified_unchanged") is True and isinstance(man.get("sources"), dict),
          "job B manifest not complete or not source-verified")
    pr = man.get("prescription", {})
    for k, v in B_PRESCRIPTION.items():
        check(k in pr and (pr[k] == v if not isinstance(v, float) else abs(float(pr[k]) - v) < 1e-12),
              f"job B prescription {k} = {pr.get(k)!r}, expected {v!r}")
    check(man.get("dataset", {}).get("sha256") == PU.sha(ds_path), "job B dataset hash does not match its manifest")
    check(man.get("cells", {}).get("digest") == cells_digest, "job B cell directory differs from its manifest")
    check(man.get("ratio_table", {}).get("sha256") == ratio_sha, "job B used a different ratio table")
    return man


def check_clean_arrays(hid, y, y_shape, logm_cat, logm_ds, cat_ids):
    check(len(set(int(c) for c in cat_ids)) == len(cat_ids), "catalogue has duplicate haloIDs")
    check(set(int(h) for h in hid) <= set(int(c) for c in cat_ids), "parents missing from the catalogue")
    check(np.isfinite(y).all() and np.isfinite(logm_cat).all(), "non-finite labels or catalogue masses")
    check(np.max(np.abs(np.asarray(y_shape, float) - y)) <= TOL_LABEL, "shape-dataset labels differ")
    check(np.max(np.abs(logm_cat - logm_ds)) <= TOL_LOGM, "catalogue and dataset log M500 differ")


# ======================================================================= run context
class Run:
    def __init__(self, run_dir, argv):
        if os.path.exists(run_dir):
            raise SystemExit(f"refusing to reuse existing run directory {run_dir}")
        os.makedirs(run_dir)
        self.dir, self.argv, self.t0 = run_dir, argv, time.time()
        self.head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.dirty = self.git_dirty()
        self.helpers = {h: PU.sha(h) for h in HELPERS if os.path.exists(h)}
        self.fp, self.task_inputs, self.status, self.cache = {}, {}, {}, {}

    def git_dirty(self):
        return subprocess.run(["git", "status", "--porcelain", "--"] + HELPERS,
                              capture_output=True, text=True).stdout.strip()

    def use(self, task, **paths):
        """Fingerprint inputs BEFORE they are read (cached per path)."""
        for name, p in paths.items():
            if p not in self.fp:
                if os.path.isdir(p):
                    self.fp[p] = dict(PU.dir_digest(p), kind="dir")
                else:
                    self.fp[p] = dict(sha256=PU.sha(p), **PU.stat_fp(p))
            self.task_inputs.setdefault(task, {})[name] = p
        return paths

    def path(self, name):
        p = os.path.join(self.dir, name)
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
        return p

    def save_fig(self, fig, name):
        import matplotlib.pyplot as plt
        fig.savefig(self.path(f"fig_{name}.pdf"))
        plt.close(fig)

    def sidecar(self, task, **arrays):
        np.savez(self.path(f"{task}_arrays.npz"), **arrays)

    def record(self, task, data):
        rec = dict(task=task, command=" ".join([sys.executable] + self.argv), git_head=self.head,
                   uncommitted_helpers=self.dirty, helper_sha256=self.helpers,
                   inputs={k: dict(path=p, **self.fp[p]) for k, p in self.task_inputs.get(task, {}).items()},
                   threads=os.environ.get("OMP_NUM_THREADS"), result=data)
        with open(self.path(f"{task}.json"), "w") as f:
            json.dump(jsonable(rec), f, indent=1)

    def manifest(self, requested):
        changed = [h for h, s in self.helpers.items() if PU.sha(h) != s]
        for p, fp in self.fp.items():
            if fp.get("kind") == "dir":
                if PU.dir_digest(p)["digest"] != fp["digest"]:
                    changed.append(p)
            elif PU.stat_fp(p) != dict(size=fp["size"], mtime_ns=fp["mtime_ns"]):
                changed.append(p)
        complete = all(self.status.get(t) == "ok" for t in requested)
        m = dict(requested=requested, status=self.status, git_head=self.head, command=" ".join(self.argv),
                 seconds=round(time.time() - self.t0, 1), sources_changed=changed,
                 uncommitted_helpers=self.dirty, complete=complete,
                 promotable=bool(complete and not changed and not self.dirty),
                 limitations="C/D inputs: full sha256 before use, size/mtime re-checked at the end (ordinary "
                             "concurrent changes); A/B raw cutouts: partial fingerprints (see producer records)")
        with open(self.path("manifest.json"), "w") as f:
            json.dump(m, f, indent=1)
        return m["promotable"]


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "savefig.bbox": "tight", "savefig.dpi": 200})
    return plt


def load_npz(p):
    return dict(np.load(p, allow_pickle=True))


def clean_inputs(run, task):
    """Clean maps, scalars and the two pooled-CNN prediction families with
    identity, label, mass, selection and fold-consistency checks."""
    import h5py
    import compare_syn as CS
    paths = dict(ds128=IN["ds128"], shape128=IN["shape128"], catalog=IN["catalog"])
    paths |= {f"{fam}_s{s}": PRED[fam].format(s) for fam in ("clean", "shape") for s in SEEDS}
    run.use(task, **paths)
    with h5py.File(IN["ds128"]) as f:
        validate_maps(f, 128)
        hid, y = f["meta/halo_id"][:], f["labels/pseudo_tsc"][:].astype(np.float64)
        imgs, r500 = f["images"][:].astype(np.float64), f["meta/r500c_kpc"][:].astype(np.float64)
        logm_ds = f["meta/log_m500"][:].astype(np.float64)
    with h5py.File(IN["shape128"]) as f:
        validate_maps(f, 128)
        check(np.array_equal(f["meta/halo_id"][:], hid), "shape dataset halo order differs")
        y_sh = f["labels/pseudo_tsc"][:]
    with h5py.File(IN["catalog"]) as f:
        cat_ids, cat_m = f["haloID"][:], f["mhalo_500c"][:]
    mm = dict(zip(cat_ids.tolist(), cat_m))
    check(all(int(h) in mm for h in hid), "parents missing from the catalogue")
    logm = np.array([mm[int(h)] for h in hid], dtype=np.float64)
    check_clean_arrays(hid, y, y_sh, logm, logm_ds, cat_ids)
    check(np.isfinite(imgs).all() and np.isfinite(r500).all() and (r500 > 0).all(), "non-finite maps or r500")
    lin = np.log10(np.sinh(imgs).sum(axis=(1, 2, 3)) + 1)
    X5 = np.column_stack([logm, np.log10(r500), lin, imgs.mean(axis=(1, 2, 3)), (imgs > 0).mean(axis=(1, 2, 3))])
    check(np.isfinite(X5).all(), "non-finite scalar features")
    n = len(hid)
    P, S, folds, fold_checks = {}, {}, {}, {}
    for fam, store, yref in (("clean", P, y), ("shape", S, y_sh.astype(np.float64))):
        ds = [load_npz(PRED[fam].format(s)) for s in SEEDS]
        CS.validate_set(ds, PRED[fam], SEEDS)            # clean branch: finite 1-D labels/predictions
        for s, d in zip(SEEDS, ds):
            check(np.array_equal(d["halo_id"], hid), f"{fam} s{s}: halo order differs from {IN['ds128']}")
            check(np.max(np.abs(d["label"] - yref)) <= TOL_LABEL, f"{fam} s{s}: labels differ")
            check((d["snap"] == 99).all() and int(d["seed"]) == s, f"{fam} s{s}: snapshot/seed metadata")
            check(str(d["select"]) == "inner" and str(d["modality"]) == "single",
                  f"{fam} s{s}: select/modality {d['select']}/{d['modality']} != inner/single")
            fold = folds.setdefault(s, reconstruct_folds(n, s))
            rec = np.array([r2(yref[fold == f], d["oof_pred"][fold == f]) for f in range(N_FOLD)])
            dmax = float(np.max(np.abs(rec - d["fold_r2"])))
            check(dmax < 1e-4, f"{fam} s{s}: reconstructed partition does not reproduce saved fold R2 "
                               f"(max diff {dmax:.2e})")
            fold_checks[f"{fam}_s{s}"] = dmax
            store[s] = d["oof_pred"].astype(np.float64)
    info = dict(n=n, fold_r2_reproduction_maxdiff=fold_checks,
                logm_catalog_vs_dataset_maxdiff=float(np.max(np.abs(logm - logm_ds))),
                fold_provenance=("partitions reconstructed with train_cnn_pooled.group_split + KFold(5, shuffle, "
                                 "random_state=seed) in the saved row order; agreement with the five saved fold R2 "
                                 "values is a consistency check, not recorded original membership"))
    return hid, y, logm, lin, X5, P, S, folds, info


# ======================================================================= W1
def task_w1(run, a):
    plt = style()
    hid, y, logm, lin, X5, P, S, folds, info = clean_inputs(run, "w1")
    n = len(y)
    out = dict(inputs_check=info)
    seeds_r2 = lambda store: np.array([r2(y, store[s]) for s in SEEDS])
    cnn_r2, shape_r2 = seeds_r2(P), seeds_r2(S)
    ens = np.mean([P[s] for s in SEEDS], 0)
    out["cnn"] = dict(per_seed=cnn_r2, mean=cnn_r2.mean(), sd=cnn_r2.std(ddof=1), ensemble_r2=r2(y, ens))
    out["shape_cnn"] = dict(per_seed=shape_r2, mean=shape_r2.mean(), sd=shape_r2.std(ddof=1))
    base = {"mass": logm[:, None], "integrated emission": lin[:, None], "5 scalars": X5}
    bpred, brec = {k: {} for k in base}, {k: {} for k in base}
    for k, X in base.items():
        sc = []
        for s in SEEDS:
            p, recs = outer_cv(X, y, folds[s], s, hid)
            bpred[k][s], brec[k][s] = p, recs
            sc.append(r2(y, p))
        out[f"baseline_{k}"] = dict(per_seed=sc, mean=np.mean(sc), sd=np.std(sc, ddof=1), fits=brec[k])
    out["baseline_estimand"] = ("ridge (scaler in pipeline, alpha by 5-fold inner CV, KFold(shuffle, random_state="
                                "10*seed+fold) over the training rows in saved order); outer test partitions = each "
                                "seed's reconstructed CNN partition: matching test partitions, not identical CNN "
                                "fitting/selection sets; mean +- sd over the 5 partitions")
    cal = {}
    for s in SEEDS:
        arms = {"scalars": X5, "scalars+CNN": np.column_stack([X5, P[s]])}
        cal[s] = dict(main=within_fold_calibration(hid, y, folds[s], arms, s),
                      tr=within_fold_calibration(hid, P[s], folds[s], {"scalars->CNN": X5}, s),
                      sh=within_fold_calibration(hid, S[s], folds[s], {"emission->shape CNN": lin[:, None]}, s))
    allfull = all(cal[s]["main"][1]["scalars"] and cal[s]["main"][1]["scalars+CNN"] for s in SEEDS)
    res = dict(available=allfull)
    PA = np.array([cal[s]["main"][0]["scalars"] for s in SEEDS])
    PB = np.array([cal[s]["main"][0]["scalars+CNN"] for s in SEEDS])
    if allfull:
        a_ = np.array([r2(y, PA[k]) for k in range(len(SEEDS))])
        b_ = np.array([r2(y, PB[k]) for k in range(len(SEEDS))])
        rng = np.random.default_rng(BOOT_SEED)
        bs, bad = [], 0
        for _ in range(BOOT):
            i = rng.integers(0, n, n)
            v = np.mean([r2(y[i], PB[k, i]) - r2(y[i], PA[k, i]) for k in range(len(SEEDS))])
            if not np.isfinite(v):
                bad += 1
                continue
            bs.append(v)
        res.update(scalars=dict(per_seed=a_, mean=a_.mean(), sd=a_.std(ddof=1)),
                   scalars_cnn=dict(per_seed=b_, mean=b_.mean(), sd=b_.std(ddof=1)),
                   increment=dict(per_seed=b_ - a_, mean=(b_ - a_).mean(), sd=(b_ - a_).std(ddof=1),
                                  lo=float(np.percentile(bs, 2.5)) if not bad else np.nan,
                                  hi=float(np.percentile(bs, 97.5)) if not bad else np.nan,
                                  invalid_draws=bad, draws=BOOT,
                                  interval="parent bootstrap of the seed-mean increment, conditional on the "
                                           "fitted CNNs and calibration regressions"))
    res["design"] = (f"per seed and reconstructed original outer fold: sorted parent ids split into fixed "
                     f"balanced halves (seed {CAL_SEED}+1000*seed+fold); ridge fitted on one half's true labels, "
                     f"evaluated on the other, then reversed; scaler and alpha (5-fold inner CV, KFold(shuffle, "
                     f"random_state={CAL_SEED}+100*seed+fold), 13 values) confined to the fitting half; min half "
                     f"{MIN_CAL}; each parent predicted once (finite) per arm; ~35-parent calibration halves, far "
                     f"fewer than the full-training stack behind the historical number")
    res["directions"] = {s: cal[s]["main"][2] for s in SEEDS}
    out["calibration_experiment"] = res
    trk = {}
    for key_, lab, resp in (("tr", "scalars->CNN", P), ("sh", "emission->shape CNN", S)):
        ok = all(cal[s][key_][1][lab] for s in SEEDS)
        v = np.array([r2(resp[s], cal[s][key_][0][lab]) for s in SEEDS]) if ok else None
        trk[lab] = dict(available=ok, per_seed=v, mean=None if v is None else v.mean(),
                        sd=None if v is None else v.std(ddof=1), directions={s: cal[s][key_][2] for s in SEEDS},
                        estimand="held-out R2 of the saved OOF output of the same seed, within-original-fold halves")
    out["tracking"] = trk
    q = np.quantile(logm, [1 / 3, 2 / 3])
    bins = [logm <= q[0], (logm > q[0]) & (logm <= q[1]), logm > q[1]]
    terc = {}
    for lab, src in (("mass", bpred["mass"]), ("5 scalars", bpred["5 scalars"]), ("CNN", P)):
        v = np.array([[r2(y[b], src[s][b]) for b in bins] for s in SEEDS])
        terc[lab] = dict(per_seed=v, mean=v.mean(0), sd=v.std(0, ddof=1))
    terc["CNN, 5-seed ensemble"] = dict(values=[r2(y[b], ens[b]) for b in bins])
    terc["edges_logm"], terc["n"] = q, [int(b.sum()) for b in bins]
    out["terciles"] = terc
    ps = np.array([partial_spearman(y, P[s], logm) for s in SEEDS])
    out["partial_spearman_tsc_cnn_given_mass"] = dict(per_seed=ps, mean=ps.mean(), sd=ps.std(ddof=1),
                                                      ensemble=partial_spearman(y, ens, logm),
                                                      estimand="rank residuals on rank(logM500); descriptive")
    run.sidecar("w1", halo_id=hid, y=y, logm=logm, X5=X5, folds=np.array([folds[s] for s in SEEDS]),
                cnn=np.array([P[s] for s in SEEDS]), shape_cnn=np.array([S[s] for s in SEEDS]), ensemble=ens,
                **{f"outer_{k.replace(' ', '_')}": np.array([bpred[k][s] for s in SEEDS]) for k in base},
                cal_scalars=PA, cal_scalars_cnn=PB,
                track_scalars_cnn=np.array([cal[s]["tr"][0]["scalars->CNN"] for s in SEEDS]),
                track_emission_shape=np.array([cal[s]["sh"][0]["emission->shape CNN"] for s in SEEDS]),
                seeds=np.array(SEEDS))
    fig, ax = plt.subplots(1, 2, figsize=(TEXT, TEXT * 0.34), gridspec_kw={"width_ratios": [1.35, 1]})
    rows = [("mass (ridge)", out["baseline_mass"]["mean"], out["baseline_mass"]["sd"], "0.6", None),
            ("integrated emission", out["baseline_integrated emission"]["mean"],
             out["baseline_integrated emission"]["sd"], "0.6", None),
            ("5 scalars", out["baseline_5 scalars"]["mean"], out["baseline_5 scalars"]["sd"], "0.6", None),
            ("normalised-map CNN", shape_r2.mean(), shape_r2.std(ddof=1), C_OBS, None),
            ("CNN", cnn_r2.mean(), cnn_r2.std(ddof=1), C_SIM, None),
            ("CNN, 5-seed mean", out["cnn"]["ensemble_r2"], None, C_SIM, None)]
    if allfull:
        rows += [("5 scalars [half-sample]", res["scalars"]["mean"], res["scalars"]["sd"], "0.85", "//"),
                 ("5 scalars + CNN [half-sample]", res["scalars_cnn"]["mean"], res["scalars_cnn"]["sd"], "0.85", "//")]
    lo = min([0.0] + [v - (e or 0) for _, v, e, _, _ in rows]) - 0.02
    hi = max(v + (e or 0) for _, v, e, _, _ in rows) + 0.09
    yy = np.arange(len(rows))[::-1]
    for y_, (lab, v, e, c, h) in zip(yy, rows):
        ax[0].barh(y_, v, xerr=e, color=c, hatch=h, edgecolor="k" if h else None, lw=0.4 if h else 0,
                   capsize=2, height=0.7)
        ax[0].text(max(v, 0) + (e or 0) + 0.01, y_, f"{v:.3f}", va="center", fontsize=6.5)
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set_yticks(yy, [r_[0] for r_ in rows])
    ax[0].set_xlabel("$R^2$ for pseudo-TSC")
    ax[0].set_xlim(lo, hi)
    x = np.arange(3)
    for off, lab, c in [(-0.27, "mass", "0.75"), (0.0, "5 scalars", "0.45"), (0.27, "CNN", C_SIM)]:
        ax[1].bar(x + off, terc[lab]["mean"], 0.25, yerr=terc[lab]["sd"], color=c, label=lab, capsize=2)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(x, ["low", "mid", "high"])
    ax[1].set_xlabel(r"$M_{500}$ tercile")
    ax[1].set_ylabel("$R^2$ within tercile")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    run.save_fig(fig, "baselines")
    out["figure_limits"] = dict(x=[lo, hi], rule="cover every bar and error bar")
    run.record("w1", out)
    print(f"  CNN {cnn_r2.mean():.3f} +- {cnn_r2.std(ddof=1):.3f}; ensemble {out['cnn']['ensemble_r2']:.3f}; "
          f"5 scalars {out['baseline_5 scalars']['mean']:.3f}; calibration available {allfull}")


# ======================================================================= D prerequisites
def prerequisites(run, a, need, tasks):
    """Validate the job A/B products ONCE, before any dependent fit: cohort,
    success manifests, recorded cell-directory digests and ratio table, map
    protocol, and (A and B) the per-parent summed-weight cross-check. Results
    are cached for the tasks."""
    import h5py
    paths = dict(ds128=IN["ds128"], ds1400=IN["ds1400"], b_manifest=IN["ds1400"] + ".manifest.json",
                 cells1400=IN["cells1400"], ratio_table=IN["ratio_table"])
    if "A" in need:
        check(a.phase_npz, "--phase-npz (job A output) is required")
        paths |= dict(phase_npz=a.phase_npz, phase_manifest=a.phase_npz + ".manifest.json",
                      cells144=IN["cells144"])
    for t in tasks:
        run.use(t, **paths)
    with h5py.File(IN["ds128"]) as f1, h5py.File(IN["ds1400"]) as f2:
        at1, hid = validate_maps(f1, 128)
        at2, hid2 = validate_maps(f2, 128)
        check(np.array_equal(hid, hid2), "map sets differ in halo order")
        check(at1.get("cells_dir") == IN["cells144"] and at2.get("cells_dir") == IN["cells1400"],
              "map sets were not built from the declared cell directories")
    ratio_sha = run.fp[IN["ratio_table"]]["sha256"]
    run.cache["bman"] = validate_b_manifest(IN["ds1400"], run.fp[IN["cells1400"]]["digest"], ratio_sha)
    if "A" in need:
        run.cache["phase"] = validate_phase(a.phase_npz, hid, ratio_sha, run.fp[IN["cells144"]]["digest"])
        d = run.cache["phase"][0]
        sc = B_PRESCRIPTION["w_scale"] / PHASE_CONFIG["w_scale_cells"]
        rel = []
        for h, rc in zip(hid, d["ratio_cut"]):
            f144 = glob.glob(os.path.join(IN["cells144"], f"cells_FOF{int(h)}_sub*.npz"))
            f14 = glob.glob(os.path.join(IN["cells1400"], f"cells_FOF{int(h)}_sub*.npz"))
            check(len(f144) == 1 and len(f14) == 1, f"parent {h}: cell files missing or duplicated")
            w1, w2 = np.load(f144[0])["w"], np.load(f14[0])["w"]
            check(len(w1) == len(w2), f"parent {h}: 144 MHz and 1.4 GHz cell counts differ")
            check(np.isfinite(rc) and rc > 0, f"parent {h}: unavailable job-A frequency ratio")
            rel.append(abs(w1.sum() / w2.sum() - rc * sc) / (rc * sc))
        rel = np.array(rel)
        check(len(rel) == len(hid) and np.all(rel <= TOL_CELL_RATIO),
              f"job B cells do not reproduce job A's 144/1400 ratio (max rel {np.max(rel):.2e})")
        run.cache["cells_ratio_rel"] = rel
    run.cache["hid"] = hid


# ======================================================================= W2
def mf_block(images):
    from minkowski_functionals import FRACS, mf_curves
    N, P = images.shape[:2]
    curves = np.array([[mf_curves(images[i, p].astype(np.float64)) for p in range(P)] for i in range(N)])
    dev = curves[..., 0] / FRACS - 1.0
    return curves, dict(max_abs_rel_area_dev=float(np.max(np.abs(dev))),
                        n_maps_area_dev_gt_10pct=int((np.abs(dev) > 0.1).any(-1).sum()),
                        median_abs_rel_area_dev=float(np.median(np.abs(dev))))


def task_w2(run, a):
    import h5py
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBRegressor
    from minkowski_functionals import FRACS, oof_r2
    plt = style()
    bman = run.cache["bman"]
    out, keep = dict(job_b_manifest=dict(path=IN["ds1400"] + ".manifest.json", prescription=bman["prescription"],
                                         dataset_sha256=bman["dataset"]["sha256"])), {}
    ref = None
    for lab, p in (("144 MHz", IN["ds128"]), ("1.4 GHz", IN["ds1400"])):
        with h5py.File(p) as f:
            at, hid = validate_maps(f, 128)
            imgs, y = f["images"][:], f["labels/pseudo_tsc"][:].astype(float)
        check(np.isfinite(imgs).all() and np.isfinite(y).all(), f"{p}: non-finite maps or labels")
        if ref is None:
            ref = (hid, y)
        else:
            check(np.array_equal(hid, ref[0]) and np.max(np.abs(y - ref[1])) <= TOL_LABEL,
                  "frequency sets differ in identity/labels")
        N, P = imgs.shape[:2]
        curves, ties = mf_block(imgs)
        feats = curves[:, :, :, 1:].reshape(N * P, -1)
        yy, groups = np.repeat(y, P), np.repeat(np.arange(N), P)
        gfold = np.full(N * P, -1)
        for f_, (_, te) in enumerate(GroupKFold(n_splits=5).split(feats, yy, groups)):
            gfold[te] = f_
        r2p, r2c, oof = oof_r2(feats, yy, groups, lambda: XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=0, n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1"))))
        cm = curves.mean(1)
        j = int(np.argmin(np.abs(FRACS - 0.5)))
        rho = spearmanr(cm[:, j, 2], y)
        pmean = np.array([oof[groups == g].mean() for g in range(N)])
        out[lab] = dict(dataset=p, attrs=at, n_parents=N, n_projections=N * P, n_features=feats.shape[1],
                        r2_projection=r2p, r2_parent_mean=r2c, area_fraction_ties=ties,
                        components_at_f05_spearman=float(rho[0]), components_at_f05_p=float(rho[1]),
                        f_threshold=float(FRACS[j]))
        keep[lab] = dict(curves=curves, oof=oof, pmean=pmean, gfold=gfold, cm=cm, y=y)
    check(np.array_equal(keep["144 MHz"]["gfold"], keep["1.4 GHz"]["gfold"]), "GroupKFold folds differ")
    out["difference_1400_minus_144"] = dict(
        projection=out["1.4 GHz"]["r2_projection"] - out["144 MHz"]["r2_projection"],
        parent_mean=out["1.4 GHz"]["r2_parent_mean"] - out["144 MHz"]["r2_parent_mean"])
    out["protocol"] = ("XGBoost fixed hyperparameters (400 trees, depth 4, lr 0.05, subsample/colsample 0.8, "
                       "seed 0), 48 features = 12 area-fraction thresholds x {perimeter, components, holes, "
                       "Euler}; parent GroupKFold(5) (deterministic, identical rows at both frequencies; not the "
                       "CNN partitions); scaler fitted on training folds; f = 0.5 is the lowest intensity "
                       "threshold")
    k1, k2 = keep["144 MHz"], keep["1.4 GHz"]
    run.sidecar("w2", halo_id=ref[0], y=ref[1], projection=np.tile(np.arange(3), len(ref[0])),
                row_halo=np.repeat(ref[0], 3), group_fold=k1["gfold"], fracs=FRACS,
                curves_144=k1["curves"], curves_1400=k2["curves"], oof_144=k1["oof"], oof_1400=k2["oof"],
                parent_mean_144=k1["pmean"], parent_mean_1400=k2["pmean"])
    y, cm = k1["y"], k1["cm"]
    terc = np.quantile(y, [1 / 3, 2 / 3])
    masks = [y < terc[0], (y >= terc[0]) & (y <= terc[1]), y > terc[1]]
    labs = [f"pseudo-TSC < {terc[0]:.1f} Gyr", "middle", f"pseudo-TSC > {terc[1]:.1f} Gyr"]
    fig, ax = plt.subplots(1, 3, figsize=(TEXT, TEXT * 0.3))
    for a_, j, name in [(ax[0], 2, "connected components"), (ax[1], 4, "Euler characteristic")]:
        for m, lab, c in zip(masks, labs, ["#c2455d", "#8a8a8a", "#3b6ea8"]):
            mu, sd = cm[m, :, j].mean(0), cm[m, :, j].std(0)
            a_.plot(FRACS, mu, color=c, label=lab)
            a_.fill_between(FRACS, mu - sd, mu + sd, color=c, alpha=0.15)
        a_.set_xscale("log")
        a_.set_xlabel("area-fraction threshold")
        a_.set_ylabel(name)
    ax[0].legend()
    pm_ = k1["pmean"]
    lo, hi = min(-0.1, pm_.min() - 0.1), max(4.2, pm_.max() + 0.1)
    ax[2].scatter(y, pm_, s=4, alpha=0.5, color=C_SIM)
    ax[2].plot([0, 4], [0, 4], "k:", lw=0.8)
    ax[2].set_ylim(lo, hi)
    ax[2].set_xlabel("pseudo-TSC [Gyr]")
    ax[2].set_ylabel("predicted from topology [Gyr]")
    ax[2].set_title(f"$R^2$ = {out['144 MHz']['r2_parent_mean']:.2f} (parent mean)")
    fig.tight_layout()
    run.save_fig(fig, "mf_tsc")
    run.record("w2", out)
    print(f"  MF 144: {out['144 MHz']['r2_projection']:.3f}/{out['144 MHz']['r2_parent_mean']:.3f}; "
          f"1.4 GHz: {out['1.4 GHz']['r2_projection']:.3f}/{out['1.4 GHz']['r2_parent_mean']:.3f}")


# ======================================================================= W3
def task_w3(run, a):
    import h5py
    plt = style()
    import h5py as _h5
    bman = run.cache["bman"]
    d, pv, aman = run.cache["phase"]
    hid = run.cache["hid"]
    with _h5.File(IN["ds128"]) as f1, _h5.File(IN["ds1400"]) as f2:
        m1, m2 = f1["images"][:].astype(np.float64), f2["images"][:].astype(np.float64)
    out = dict(job_a=dict(path=a.phase_npz, config=pv["config"], n_parents=pv["n_parents"],
                          cells_cut=pv["cells_cut"], ratio_table=pv["ratio_table"]),
               job_b=dict(prescription=bman["prescription"], dataset_sha256=bman["dataset"]["sha256"]))
    pct = lambda v: dict(median=float(np.nanmedian(v)), p16=float(np.nanpercentile(v, 16)),
                         p84=float(np.nanpercentile(v, 84)), n=int(np.isfinite(v).sum()),
                         n_unavailable=int((~np.isfinite(v)).sum()))
    for k in ("frac_above", "frac_cold", "top_all", "top_cut", "mach_all", "mach_cut"):
        out[k] = pct(d[k])
    out["definitions"] = dict(frac_above="144 MHz weight fraction in valid cells with n_H > 1e-4 (cut keeps <=)",
                              frac_cold="weight fraction with kT < 0.1 keV", top="brightest gas CELL share",
                              mach="weight-averaged Mach number", population="all valid shock cells, raw cutouts "
                              "(not the finite projected-map footprint)")
    out["cells"] = dict(total=int(d["n_cells"].sum()), invalid=int(d["n_invalid"].sum()),
                        zero_weight=int(d["n_zero_w"].sum()), out_of_range=int(d["n_out_of_range"].sum()),
                        weight_out_of_range_frac_median=float(np.median(d["w_out_of_range"] / d["w_total"])),
                        cut_status={str(s): int((d["cut_status"] == s).sum()) for s in np.unique(d["cut_status"])})
    rel = run.cache["cells_ratio_rel"]
    G = float(np.median(d["ratio_cut"]))
    out["frequency_cells"] = dict(
        G_median_ratio_cut=G, sd_log10_ratio_cut=float(np.std(np.log10(d["ratio_cut"]), ddof=1)),
        G_median_ratio_all=float(np.median(d["ratio_all"])),
        sd_log10_ratio_all=float(np.std(np.log10(d["ratio_all"]), ddof=1)), n=len(hid),
        job_b_cross_check_max_rel=float(np.max(rel)),
        job_b_cross_check_scope="per-parent summed-weight ratio and cell count only; it does not establish "
                                "cell-by-cell or spatial identity (bound instead by the recorded cell-directory "
                                "digests)",
        estimand="per parent sum(w144)/sum(w1400) over density-cut cells, unscaled physical weights; SD over "
                 "parents (ddof 1) of log10 of that ratio; w1400 = w144 (1.4/0.144)^(-s/2) at fixed cell "
                 "properties (implementation check, not independent physics validation)")
    check(np.isfinite(d["ratio_cut"]).all(), "unavailable per-parent frequency ratios")
    N, P = m1.shape[:2]
    tv = np.array([[tv_distance(np.sinh(m1[i, p]), np.sinh(m2[i, p])) for p in range(P)] for i in range(N)])
    navail = np.isfinite(tv).sum(1)
    pm = np.full(N, np.nan)
    pm[navail > 0] = np.nanmean(tv[navail > 0], 1)
    out["redistribution_tv"] = dict(parent_mean=pct(pm), pooled_projection=pct(tv.ravel()),
                                    n_unavailable_maps=int(np.isnan(tv).sum()),
                                    n_parents_partial=int(((navail > 0) & (navail < P)).sum()),
                                    estimand="0.5 sum |p144 - p1400| between unit-sum linear (sinh) 128-px maps; "
                                             "primary: median over parents of the projection mean")
    run.sidecar("w3", halo_id=hid, tv=tv, tv_parent_mean=pm, ratio_cut=d["ratio_cut"], ratio_all=d["ratio_all"],
                frac_above=d["frac_above"], frac_cold=d["frac_cold"], top_all=d["top_all"], top_cut=d["top_cut"],
                mach_all=d["mach_all"], mach_cut=d["mach_cut"], cells_ratio_rel=rel)
    Hn, Hw = np.nanmean(d["H_n"], 0), np.nanmean(d["H_w"], 0)
    ext = [d["nh_edges"][0], d["nh_edges"][-1], d["t_edges"][0], d["t_edges"][-1]]
    fig, ax = plt.subplots(1, 2, figsize=(TEXT, TEXT * 0.36), sharey=True)
    for a_, H, title in [(ax[0], Hn, "shock cells (by number)"), (ax[1], Hw, "radio weight, 144 MHz")]:
        with np.errstate(divide="ignore"):
            im = a_.imshow(np.log10(H.T), origin="lower", extent=ext, aspect="auto", cmap="viridis",
                           vmin=np.log10(H[H > 0]).max() - 4)
        a_.axvline(-4.0, color="w", ls="--", lw=1)
        a_.set_title(title)
        a_.set_xlabel(r"log $n_{\rm H}$ [cm$^{-3}$]")
        fig.colorbar(im, ax=a_, pad=0.01, label="log fraction per bin")
    ax[0].set_ylabel(r"log $kT$ [keV]")
    ax[1].text(0.03, 0.97, f"{out['frac_above']['median']:.0%} of the weight\nat $n_{{\\rm H}} > 10^{{-4}}$",
               transform=ax[1].transAxes, ha="left", va="top", fontsize=7)
    fig.tight_layout()
    run.save_fig(fig, "phase")
    run.record("w3", out)
    print(f"  weight above cut {out['frac_above']['median']:.3f}; G {G:.3f}; "
          f"TV parent-mean median {out['redistribution_tv']['parent_mean']['median']:.3f}")


# ======================================================================= W4 / W7 shared
def map_totals(ds512):
    """Per (parent, projection) full-footprint linear sum of the 512 maps."""
    import h5py
    with h5py.File(ds512) as f:
        _, hid = validate_maps(f, 512)
        r500 = f["meta/r500c_kpc"][:].astype(np.float64)
        ext = float(f.attrs["extent_r500"])
        N, P, H, _ = f["images"].shape
        T = np.zeros((N, P))
        for i in range(N):
            lin = np.sinh(f["images"][i].astype(np.float64))
            check(np.isfinite(lin).all(), f"non-finite 512 map for parent {hid[i]}")
            T[i] = lin.sum(axis=(1, 2))
    check((T > 0).all(), "a full-map linear total is not positive")
    return hid, r500, ext, H, T


def inj_box(path, hid, ds512):
    import h5py
    with h5py.File(path) as f:
        at = dict(f.attrs)
        check(np.array_equal(f["mock/halo_id"][:], hid), f"{path}: halo order differs")
        check(str(attr(at.get("source_dataset"))) == ds512 and str(attr(at.get("recipe"))) == "physical"
              and str(attr(at.get("flux_mode"))) == "sim" and not bool(at.get("control")),
              f"{path}: unexpected source/recipe/flux mode/control")
        logp = f["mock/logp150"][:]
        lm, tsc = f["mock/log_m500"][:].astype(float), f["mock/pseudo_tsc"][:].astype(float)
    off = float(at["sim_flux_offset_dex"])
    lb, dmax = collapse_realisations(logp, n=len(hid))
    return lb, off, float(at["box_kpc"]), dmax, lm, tsc


def task_w4(run, a):
    import h5py
    run.use("w4", ds512=IN["ds512"], inj3=IN["inj3"], inj1=IN["inj1"])
    hid, r500, ext, H, T = map_totals(IN["ds512"])
    out, arrays, offs = dict(extent_r500=ext, img_size=H), dict(halo_id=hid, total_linear=T), []
    for tag in ("inj3", "inj1"):
        lb, off, box, dmax, _, _ = inj_box(IN[tag], hid, IN["ds512"])
        offs.append(off)
        B = 10.0 ** (lb - off)
        F = B / T
        diff = np.zeros_like(B)
        with h5py.File(IN["ds512"]) as f:
            for i in range(len(hid)):
                px = 2 * ext * r500[i] / H
                lin = np.sinh(f["images"][i].astype(np.float64))
                for p in range(lin.shape[0]):
                    diff[i, p] = np.log10(box_sum(lin[p], box / 2 / px)) - (lb[i, p] - off)
        check(np.isfinite(diff).all() and np.max(np.abs(diff)) < TOL_CROP_DEX,
              f"{tag}: independent crop disagrees with logp150 (max {np.nanmax(np.abs(diff)):.2e} dex, "
              f"{int((~np.isfinite(diff)).sum())} non-finite)")
        nm = f"box_{box / 1000:.0f}Mpc"
        out[nm] = dict(file=IN[tag], box_kpc=box, offset_dex=off, realisation_maxdiff_dex=dmax,
                       crop_check_max_abs_dex=float(np.max(np.abs(diff))),
                       median_projection=float(np.median(F)), p16=float(np.percentile(F, 16)),
                       p84=float(np.percentile(F, 84)), median_parent_mean=float(np.median(F.mean(1))),
                       aggregate_sum_ratio=float(B.sum() / T.sum()))
        arrays[f"{nm}_linear"], arrays[f"{nm}_fraction"], arrays[f"{nm}_crop_diff_dex"] = B, F, diff
    check(abs(offs[0] - offs[1]) < 1e-12, "the two injected files use different flux offsets")
    out["estimand"] = ("linear emission inside the centred physical box (forward model's fractional-overlap "
                       "crop, 10**(logp150 - offset)) over the full projected 512-px map footprint "
                       "(+-extent_r500 R500); primary: median over parent/projection; also parent-mean median "
                       "and aggregate sum ratio")
    run.sidecar("w4", **arrays)
    run.record("w4", out)
    print("  " + "; ".join(f"{k}: {v['median_projection']:.3f}" for k, v in out.items() if k.startswith("box_")))


def task_w7(run, a):
    import h5py
    from scipy import stats
    plt = style()
    run.use("w7", ds512=IN["ds512"], inj3=IN["inj3"])
    hid, r500, ext, H, T = map_totals(IN["ds512"])
    lb, off, box, dmax, lm, tsc = inj_box(IN["inj3"], hid, IN["ds512"])
    with h5py.File(IN["ds512"]) as f:
        check(np.max(np.abs(f["meta/log_m500"][:] - lm)) <= TOL_LOGM, "log M500 differs between files")
        check(np.max(np.abs(f["labels/pseudo_tsc"][:] - tsc)) <= TOL_LABEL, "pseudo-TSC differs between files")
    lp_t = np.log10(T.mean(1)) + off
    lp_b = np.log10((10.0 ** (lb - off)).mean(1)) + off
    hist_b = lb.mean(1)
    ok = np.isfinite(lp_t) & np.isfinite(lp_b) & np.isfinite(lm) & np.isfinite(tsc)
    out = dict(n=int(ok.sum()), n_excluded=int((~ok).sum()), offset_dex=off, extent_r500=ext, box_kpc=box,
               averaging="log10 of the arithmetic-mean linear power over projections for both apertures "
                         "(realisations collapsed first); corrects the old mixed arithmetic/geometric recipe",
               footprint="full map = projected +-extent_r500 square footprint, not a 3-D sphere or cube",
               calibration="one global additive offset (sim flux mode); it does not impose the slope and does "
                           "not independently calibrate absolute power; 144 MHz emission on the 150 MHz scale")
    fits = {}
    for name, lp in (("total", lp_t), ("box_3Mpc", lp_b), ("box_3Mpc_historical_mean_log", hist_b)):
        fit = stats.linregress(lm[ok], lp[ok])
        res = lp[ok] - (fit.intercept + fit.slope * lm[ok])
        rho = stats.spearmanr(tsc[ok], res)
        fits[name] = dict(slope=fit.slope, slope_se=fit.stderr, intercept=fit.intercept,
                          rms_residual_dex=float(np.sqrt(np.mean(res ** 2))),
                          resid_vs_tsc_spearman=float(rho[0]), resid_vs_tsc_p=float(rho[1]))
    out["fits"] = fits
    res_t = lp_t[ok] - (fits["total"]["intercept"] + fits["total"]["slope"] * lm[ok])
    run.sidecar("w7", halo_id=hid, mask=ok, logm=lm, pseudo_tsc=tsc, logp_total=lp_t, logp_box=lp_b,
                logp_box_historical=hist_b, resid_total=res_t)
    fig, ax = plt.subplots(1, 3, figsize=(TEXT, TEXT * 0.3))
    xx = np.linspace(lm.min(), lm.max(), 10)
    for a_, name, lp, lab in [(ax[0], "total", lp_t, r"log $P$, full map [W Hz$^{-1}$]"),
                              (ax[1], "box_3Mpc", lp_b, rf"log $P$, {box / 1000:.0f} Mpc box [W Hz$^{{-1}}$]")]:
        ft = fits[name]
        sc = a_.scatter(lm[ok], lp[ok], c=np.minimum(tsc[ok], 4), s=4, cmap="plasma_r")
        x0, y0 = np.median(lm[ok]), np.median(lp[ok])
        a_.plot(xx, ft["intercept"] + ft["slope"] * xx, "k-", lw=1,
                label=f"fit: {ft['slope']:.2f}, {ft['rms_residual_dex']:.2f} dex")
        a_.plot(xx, y0 + 2.83 * (xx - x0), "k--", lw=0.8, label="relics: 2.83")
        a_.plot(xx, y0 + 3.55 * (xx - x0), "k:", lw=0.8, label="haloes: 3.55")
        a_.set_xlabel(r"log $M_{500}$ [M$_\odot$]")
        a_.set_ylabel(lab)
        a_.legend(fontsize=5.5, loc="upper left")
    fig.colorbar(sc, ax=ax[1], pad=0.01, label="pseudo-TSC [Gyr]")
    ax[2].scatter(np.minimum(tsc[ok], 4), res_t, s=4, alpha=0.5, color=C_SIM)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel("pseudo-TSC [Gyr]")
    ax[2].set_ylabel("full-map residual [dex]")
    ax[2].set_title(f"Spearman {fits['total']['resid_vs_tsc_spearman']:+.2f}", fontsize=7)
    fig.tight_layout()
    run.save_fig(fig, "pm_relation")
    run.record("w7", out)
    print("  " + "; ".join(f"{k} {v['slope']:.2f}+-{v['slope_se']:.2f}, {v['rms_residual_dex']:.2f} dex"
                           for k, v in fits.items()))


# ======================================================================= W5
def task_w5(run, a):
    import h5py
    from check_radio_realism import KS, NAMES, mfs, separation
    plt = style()
    run.use("w5", inj3=IN["inj3"], ds512=IN["ds512"])
    with h5py.File(IN["inj3"]) as f:
        mock_all = f["mock/images"][:]
        mh, tsc, mz = f["mock/halo_id"][:], f["mock/pseudo_tsc"][:], f["mock/z"][:]
        obs, oz = f["obs/images"][:], f["obs/z"][:]
        on = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/name"][:]]
    check(mock_all.shape[:3] == (PROTO["n_clusters"], 3, 3) and np.isfinite(mock_all).all()
          and np.isfinite(obs).all(), "injected images have unexpected shape or non-finite values")
    with h5py.File(IN["ds512"]) as f:
        _, simh = validate_maps(f, 512)
        r500 = f["meta/r500c_kpc"][:]
        check(np.array_equal(simh, mh), "injected and clean halo order differ")
        pick = [int(np.argmin(np.abs(tsc - t))) for t in (0.2, 0.8, 1.8, 4.0)]
        check(len(set(pick)) == 4, "gallery selection picked a parent twice")
        clean = [f["images"][i, 0] for i in pick]
        ext = float(f.attrs["extent_r500"])
    targets = ["A399", "A2443", "A1650", "A2033"]
    oi = [on.index(t) for t in targets if t in on]
    check(len(oi) == 4, f"gallery real targets missing: {[t for t in targets if t not in on]}")
    gal = dict(parents=[dict(halo_id=int(mh[i]), pseudo_tsc=float(tsc[i]), z=float(mz[i, 0]),
                             r500_kpc=float(r500[i]), realisation=0, projection="xy") for i in pick],
               real=[dict(name=on[k], z=float(oz[k])) for k in oi],
               display=dict(clean=f"arcsinh map, central +-{ext / 2:g} R500 of the +-{ext:g} R500 512-px map, "
                                  f"own colour scale per panel",
                            mock_real="arcsinh(S / sigma_rms), vmin -1, vmax arcsinh(30)", box_kpc=3000))
    fig, ax = plt.subplots(3, 4, figsize=(TEXT, TEXT * 0.78))
    vmax = np.arcsinh(30.0)
    for c, i in enumerate(pick):
        Hh = clean[c].shape[0]
        half = Hh // 4
        ax[0, c].imshow(clean[c][Hh // 2 - half:Hh // 2 + half, Hh // 2 - half:Hh // 2 + half],
                        origin="lower", cmap="magma")
        ax[0, c].set_title(f"pseudo-TSC = {tsc[i]:.1f} Gyr", fontsize=7)
        ax[1, c].imshow(mock_all[i, 0, 0], origin="lower", cmap="magma", vmin=-1, vmax=vmax)
        ax[1, c].set_title(f"mock, z = {mz[i, 0]:.3f}", fontsize=7)
    for c, k in enumerate(oi):
        im = ax[2, c].imshow(obs[k], origin="lower", cmap="magma", vmin=-1, vmax=vmax)
        ax[2, c].set_title(f"{on[k]}, z = {oz[k]:.3f}", fontsize=7)
    for a_ in ax.ravel():
        a_.set_xticks([]), a_.set_yticks([])
    ax[0, 0].set_ylabel("simulation")
    ax[1, 0].set_ylabel("mock LoTSS")
    ax[2, 0].set_ylabel("real LoTSS")
    cb = fig.colorbar(im, ax=ax[1:, :], shrink=0.8, pad=0.01)
    cb.set_label(r"arcsinh(S / $\sigma_{\rm rms}$)")
    run.save_fig(fig, "radio_gallery")
    M = mfs(mock_all.reshape(-1, *mock_all.shape[-2:]))
    O = mfs(obs)
    sep = separation(O, M)
    D = float(np.median(np.abs(sep)))
    mq, oq = np.percentile(M, [16, 50, 84], axis=0), np.percentile(O, [16, 50, 84], axis=0)
    out = dict(gallery=gal, n_mock=len(M), n_obs=len(O), thresholds_sigma=KS, functionals=NAMES,
               separation=sep, D=D, D_rounded=round(D, 3), D_expected_rounded=0.093,
               D_reproduced=bool(round(D, 3) == 0.093), max_abs_sep_above_3sigma=float(np.abs(sep[KS > 3.0]).max()),
               mock_sd_convention="numpy std, ddof 0, +1e-9 (check_radio_realism.separation)")
    run.sidecar("w5", thresholds=KS, mock_pct=mq, obs_pct=oq, separation=sep, obs_names=np.array(on),
                gallery_halo_ids=np.array([int(mh[i]) for i in pick]))
    check(out["D_reproduced"], f"realism D {D:.4f} does not reproduce the retained 0.093 -- investigate the "
                               f"definition/provenance, do not tune")
    fig, ax = plt.subplots(2, 4, figsize=(TEXT, TEXT * 0.45), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
    for c in range(4):
        ax[0, c].fill_between(KS, mq[0, :, c], mq[2, :, c], color=C_SIM, alpha=0.25)
        ax[0, c].plot(KS, mq[1, :, c], color=C_SIM, label=f"mocks ({len(M)})")
        ax[0, c].errorbar(KS, oq[1, :, c], yerr=[oq[1, :, c] - oq[0, :, c], oq[2, :, c] - oq[1, :, c]], fmt="o",
                          ms=3, color=C_OBS, label=f"LoTSS ({len(O)})")
        ax[0, c].set_title(NAMES[c])
        ax[1, c].axhspan(-1, 1, color="0.9")
        ax[1, c].plot(KS, sep[:, c], "o-", ms=3, color="k")
        ax[1, c].axhline(0, color="0.5", lw=0.6)
        ax[1, c].set_xscale("log")
        ax[1, c].set_xlabel(r"threshold [$\sigma$]")
    ax[0, 0].legend()
    ax[1, 0].set_ylabel("separation\n[mock s.d.]")
    fig.align_ylabels(ax[:, 0])
    fig.tight_layout()
    run.save_fig(fig, "radio_mf")
    run.record("w5", out)
    print(f"  realism D {D:.4f} (reproduced {out['D_reproduced']}); gallery parents "
          f"{[p['halo_id'] for p in gal['parents']]}")


# ======================================================================= W6
def task_w6(run, a):
    import compare_syn as CS
    plt = style()
    hid, y_c, _, _, _, P, _, _, info = clean_inputs(run, "w6")
    run.use("w6", **{f"joint1_s{s}": "cnn_preds/" + PRED["joint1"].format(s) for s in SEEDS})
    p_c = np.mean([P[s] for s in SEEDS], 0)
    ds = CS.load(PRED["joint1"])
    m = ds[0]["snap"] == 99
    y_r = ds[0]["tsc"][m]
    p_r = np.mean([d["oof_cluster"] for d in ds], 0)[m]
    yv, gv, preds = CS.single(ds)
    sv = r2(yv, np.mean(preds, 0))
    out = dict(clean=dict(estimand="pooled clean CNN, 5-seed mean of OOF predictions, snapshot 99", n=len(y_c),
                          r2=r2(y_c, p_c), bins=residual_bins(y_c, p_c), at_cap=cap_stats(y_c, p_c)),
               realistic=dict(estimand="1 Mpc-radio + X-ray joint model (syn_phys_b1000_joint), snapshot-99 "
                                       "rows, pooled views with realisations averaged (oof_cluster), 5-seed mean",
                              n=int(m.sum()), r2=r2(y_r, p_r), bins=residual_bins(y_r, p_r),
                              at_cap=cap_stats(y_r, p_r), single_view_5model_r2=sv,
                              single_view_estimand="compare_syn.single: every (realisation, projection) "
                                                   "observation, 5-seed ensemble, snapshot 99",
                              single_view_expected_rounded=0.426, single_view_reproduced=bool(round(sv, 3) == 0.426)))
    for k in ("clean", "realistic"):
        out[k]["youngest_bin"] = out[k]["bins"][0]
    run.sidecar("w6", clean_halo_id=hid, clean_y=y_c, clean_pred=p_c, realistic_group_id=ds[0]["group_id"][m],
                realistic_y=y_r, realistic_pred=p_r)
    check(out["realistic"]["single_view_reproduced"], f"single-view score {sv:.4f} does not reproduce 0.426")
    allp = np.r_[p_c, p_r]
    lo, hi = min(-0.1, allp.min() - 0.1), max(4.2, allp.max() + 0.1)
    out["figure_limits"] = dict(y=[lo, hi], rule="cover every plotted prediction")
    fig, ax = plt.subplots(1, 3, figsize=(TEXT, TEXT * 0.32))
    for a_, y, p, lab, c in [(ax[0], y_c, p_c, "clean radio (144 MHz)", C_SIM),
                             (ax[1], y_r, p_r, "1 Mpc radio + X-ray", C_OBS)]:
        a_.scatter(y, p, s=4, alpha=0.5, color=c)
        a_.plot([0, 4.2], [0, 4.2], "k:", lw=0.8)
        a_.set_xlim(-0.1, 4.2), a_.set_ylim(lo, hi)
        a_.set_xlabel("true pseudo-TSC [Gyr]")
        a_.set_title(f"{lab}: $R^2$ = {r2(y, p):.2f}", fontsize=7)
    ax[0].set_ylabel("predicted [Gyr]")
    mid = 0.5 * (RESID_EDGES[1:] + RESID_EDGES[:-1])
    for k, lab, c, dx in [("clean", "clean", C_SIM, -0.04), ("realistic", "1 Mpc radio + X-ray", C_OBS, 0.04)]:
        b = out[k]["bins"]
        okb = np.array([x["status"] == "ok" for x in b])
        med = np.array([x.get("median", np.nan) for x in b])
        lo_ = np.array([x.get("p16", np.nan) for x in b])
        hi_ = np.array([x.get("p84", np.nan) for x in b])
        ax[2].errorbar(mid[okb] + dx, med[okb], yerr=[med[okb] - lo_[okb], hi_[okb] - med[okb]], fmt="o-", ms=3,
                       color=c, label=lab, capsize=2, lw=0.9)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel("true pseudo-TSC [Gyr]")
    ax[2].set_ylabel("predicted $-$ true [Gyr]")
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    run.save_fig(fig, "tsc_pred")
    run.record("w6", out)
    print(f"  clean R2 {out['clean']['r2']:.3f}; realistic pooled {out['realistic']['r2']:.3f}; single-view {sv:.3f}")


# ======================================================================= W8
def task_w8(run, a):
    import pandas as pd
    from scipy import stats
    import compare_syn as CS
    plt = style()
    run.use("w8", wl=IN["wl"], **{f"{t}_s{s}": "cnn_preds/" + PRED[t].format(s)
                                  for t in ("radio3", "xray") for s in SEEDS})
    ens = {}
    for tag in ("radio3", "xray"):
        ds = CS.load(PRED[tag])
        names = [str(n) for n in ds[0]["obs_name"]]
        keys = [norm_key(n) for n in names]
        check(len(set(keys)) == len(keys), f"{tag}: duplicate normalised target keys")
        for d in ds:
            check([str(n) for n in d["obs_name"]] == names, f"{tag}: target order differs across seeds")
            check(d["obs_pred_folds"].shape == (5, len(names)), f"{tag}: obs_pred_folds is not 5 folds")
            check(np.allclose(d["obs_pred"], d["obs_pred_folds"].mean(0), atol=1e-6),
                  f"{tag}: obs_pred is not the fold mean")
        Pm = np.array([d["obs_pred"] for d in ds])
        ens[tag] = {k: (Pm[:, i].mean(), Pm[:, i].std(ddof=1), names[i]) for i, k in enumerate(keys)}
    both = sorted(set(ens["radio3"]) & set(ens["xray"]))
    r = np.array([ens["radio3"][k][0] for k in both])
    x = np.array([ens["xray"][k][0] for k in both])
    rho, p = stats.spearmanr(r, x)
    dr = r - x
    wl = pd.read_csv(IN["wl"])
    check(wl["key"].is_unique, "weak-lensing table has duplicate keys")
    wl = wl.set_index("key")
    lmv = np.array([float(wl.loc[k, "log_m500c"]) if k in wl.index else np.nan for k in both])
    out = dict(n=len(both), keys=both, targets=[ens["radio3"][k][2] for k in both], radio=r, xray=x,
               radio_seed_sd=[ens["radio3"][k][1] for k in both], xray_seed_sd=[ens["xray"][k][1] for k in both],
               spearman=float(rho), p=float(p), median_offset=float(np.median(dr)), n_radio_older=int((dr > 0).sum()),
               radio_median=float(np.median(r)), xray_median=float(np.median(x)), wl_log_m500=lmv,
               error_bars="SD (ddof 1) across the five seeds' fold-averaged predictions; not 25-model scatter "
                          "or a calibrated age uncertainty",
               ensemble="25 models: 5 seeds x 5 folds, equal weights")
    out["reproduction"] = dict(n=out["n"] == 15, rho=round(rho, 2) == 0.67, p=round(p, 3) == 0.006,
                               offset=round(out["median_offset"], 2) == 0.49, older=out["n_radio_older"] == 14)
    check(all(out["reproduction"].values()), f"observed summary does not reproduce compare_syn: {out['reproduction']}")
    fig, ax = plt.subplots(figsize=(COL, COL * 0.9))
    vmin, vmax = np.nanmin(lmv), np.nanmax(lmv)
    for i, k in enumerate(both):
        c = plt.cm.viridis((lmv[i] - vmin) / (vmax - vmin)) if np.isfinite(lmv[i]) else "0.6"
        ax.errorbar(r[i], x[i], xerr=ens["radio3"][k][1], yerr=ens["xray"][k][1], fmt="o", ms=4, color=c,
                    mec="k", mew=0.4, lw=0.7, capsize=0)
        lab = ens["radio3"][k][2].replace("_", " ").replace("RXC J1217.6+0339", "RXC J1217")
        ax.annotate(lab, (r[i], x[i]), fontsize=4.5, xytext=(3, 2), textcoords="offset points")
    lim = [min(0, r.min() - 0.3, x.min() - 0.3), max(r.max(), x.max()) + 0.3]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("radio prediction [Gyr]")
    ax.set_ylabel("X-ray prediction [Gyr]")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax, pad=0.01, label=r"weak-lensing log $M_{500}$")
    ax.set_title(f"{len(both)} LoVoCCS clusters, Spearman {rho:.2f}", fontsize=7)
    run.save_fig(fig, "real_preds")
    run.record("w8", out)
    print(f"  n {len(both)}, rho {rho:.2f}, offset {out['median_offset']:+.2f}; medians radio "
          f"{out['radio_median']:.2f}, X-ray {out['xray_median']:.2f}")


# ======================================================================= W9
def task_w9(run, a):
    import h5py
    import detect_relics_radio as DR
    plt = style()
    run.use("w9", relic_cat=IN["relic_cat"], ds512=IN["ds512"], tsc_legacy=IN["tsc_legacy"],
            ds_legacy=IN["ds_legacy"])
    with h5py.File(IN["relic_cat"]) as f:
        cat = {k: f[k][:] for k in ("halo_id", "d_max", "n_relics", "merger_tsc", "mass_ratio")}
        cat_attrs = {k: attr(f.attrs[k]) for k in f.attrs}
    check(str(cat_attrs["source"]) == IN["ds512"], "relic catalogue source is not the current 512 maps")
    with h5py.File(IN["ds512"]) as f:
        _, hid = validate_maps(f, 512)
        imgs, r500 = f["images"][:], f["meta/r500c_kpc"][:]
        ext, mr_now = float(f.attrs["extent_r500"]), f["meta/mass_ratio"][:].astype(float)
    with h5py.File(IN["tsc_legacy"]) as f:
        mt = legacy_join(hid, f["halo_id"][:], f["tsc_gyr"][:], "TSC_Cutimages tsc_gyr")
    with h5py.File(IN["ds_legacy"]) as f:
        mr = legacy_join(hid, f["meta/halo_id"][:], f["meta/mass_ratio"][:], "dataset.h5 mass_ratio")
    sig = inspect.signature(DR.detect_in_image).parameters
    params = dict(r_min_frac=float(cat_attrs["r_min_frac"]), r_max_frac=float(cat_attrs["r_max_frac"]),
                  peak_thresh_frac=float(cat_attrs["peak_thresh_frac"]), min_sep_kpc=float(cat_attrs["min_sep_kpc"]),
                  smooth_kpc=float(sig["smooth_kpc"].default))
    N = len(hid)
    d_max, n_rel = np.full((N, 3), np.nan), np.zeros((N, 3), int)
    for i in range(N):
        for p in range(3):
            pk = DR.detect_in_image(imgs[i, p], float(r500[i]), ext * float(r500[i]), **params)
            if pk:
                n_rel[i, p], d_max[i, p] = len(pk), max(r_ for r_, _ in pk)
    agree = relic_agreement(cat, d_max, n_rel, mt, mr, mr_now, hid)
    hm, n_with = halo_summary(d_max / 1000.0)
    neg = np.isfinite(mt) & (mt < 0)
    act = np.isfinite(mt) & ~neg & (mt <= 2.0) & np.isfinite(hm)
    big = act & np.isfinite(mr) & (mr >= 0.1)
    run.sidecar("w9", halo_id=hid, d_max_redetected=d_max, n_relics_redetected=n_rel, d_max_catalogue=cat["d_max"],
                n_relics_catalogue=cat["n_relics"], merger_tsc=mt, mass_ratio=mr, mass_ratio_current=mr_now,
                halo_mean_dmax_mpc=hm, n_proj_with_peak=n_with, cohort_active=act, cohort_mass_ratio=big,
                params=np.array(json.dumps(params)))
    out = dict(params=params, agreement_with_existing_catalogue=agree, peak_definition=(
        "local maxima of the Gaussian-smoothed (smooth_kpc) arcsinh-stretched 512-px map inside the "
        "r_min_frac-r_max_frac R500 annulus, above peak_thresh_frac of the smoothed annulus maximum, "
        "maximum-filter separation 2*ceil(min_sep_kpc/pixel)+1 pixels; not a fraction of linear flux"),
        clock=("legacy snapshot-based tsc_gyr as supplied (TSC_Cutimages); its event selection and whether the "
               "stored pkl mass_ratio refers to the same event are not established here; the ratio cut is a cut "
               "on the stored catalogue field; this is not an event-associated relic test"),
        n_halos=N, n_halos_no_peak=int((n_with == 0).sum()), n_projections_no_peak=int((n_rel == 0).sum()),
        n_negative_legacy_age_excluded=int(neg.sum()), n_missing_legacy_age=int(np.isnan(mt).sum()),
        halo_summary="mean outermost-peak distance over projections with >= 1 peak; halos with none excluded",
        active=boot_spearman(hm[act], mt[act]), mass_ratio_ge_0p1=boot_spearman(hm[big], mt[big]))
    x, yv, m2 = (d_max / 1000.0).ravel(), np.repeat(mt, 3), np.repeat(mr, 3)
    a_p = np.isfinite(x) & np.isfinite(yv) & (yv >= 0) & (yv <= 2.0)
    s_p = a_p & (m2 >= 0.1)
    fig, ax = plt.subplots(figsize=(COL, COL * 0.8))
    ax.scatter(yv[a_p & ~s_p], x[a_p & ~s_p], s=3, color="0.7", label="TSC $\\leq$ 2 Gyr")
    ax.scatter(yv[s_p], x[s_p], s=5, color=C_SIM, label="stored mass ratio $\\geq$ 0.1")
    t = np.linspace(0, 2.0, 50)
    ax.plot(t, 1500 * 1.0227e-3 * t, color=C_OBS, ls="--", label="1500 km s$^{-1}$ shock")
    ax.set_xlabel("legacy time since collision [Gyr]")
    ax.set_ylabel("outermost radio peak distance [Mpc]")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0, np.max(x[a_p]) * 1.05)
    ax.legend()
    run.save_fig(fig, "relic")
    run.record("w9", out)
    print(f"  redetection agreement: {agree}; active rho {out['active']['rho']:+.3f} "
          f"[{out['active']['lo']:+.3f}, {out['active']['hi']:+.3f}] n={out['active']['n']}")
    check(all(agree.values()), f"review required: redetection or label joins disagree with the stored catalogue "
                               f"{ {k: v for k, v in agree.items() if not v} } (both diagnostics saved)")


TASKS = dict(w1=task_w1, w2=task_w2, w3=task_w3, w4=task_w4, w5=task_w5, w6=task_w6, w7=task_w7, w8=task_w8,
             w9=task_w9)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--phase-npz", default=None)
    a = ap.parse_args(argv)
    req = [t.strip() for t in a.tasks.split(",") if t.strip()]
    bad = [t for t in req if t not in TASKS]
    if bad or len(set(req)) != len(req) or not req:
        raise SystemExit(f"unknown, duplicate or empty task list: {a.tasks}")
    run = Run(a.run_dir, [os.path.relpath(__file__)] + argv)
    dep_tasks = [t for t in req if t in DEPS]
    need = set().union(*(DEPS[t] for t in dep_tasks)) if dep_tasks else set()
    if need:
        print(f"== prerequisites {sorted(need)} for {dep_tasks}", flush=True)
        try:
            prerequisites(run, a, need, dep_tasks)
        except Exception as e:
            for t in dep_tasks:
                run.status[t] = f"unavailable: prerequisite failed: {type(e).__name__}: {e}"
            print(f"  PREREQUISITE FAILED: {e}")
    for t in req:
        if t in run.status:
            continue
        print(f"== {t}", flush=True)
        try:
            TASKS[t](run, a)
            run.status[t] = "ok"
        except Fail as e:
            run.status[t] = f"failed check: {e}"
            print(f"  FAILED CHECK: {e}")
        except Exception as e:
            run.status[t] = f"error: {type(e).__name__}: {e}"
            traceback.print_exc()
    ok = run.manifest(req)
    print(f"manifest: {run.status}; ok for promotion: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
