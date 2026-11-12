#!/usr/bin/env python3
"""Regenerate the manuscript items marked \\stale{} (review rounds 48-49).

Run from the repository root, under SLURM:
    ./venv/bin/python paper/scripts/regen_stale.py --run-dir paper/staged/<new dir> \
        --tasks w1,w4,w5,w6,w7,w8,w9            (job C)
    ./venv/bin/python paper/scripts/regen_stale.py --run-dir paper/staged/<new dir> \
        --tasks w2,w3 --phase-npz <job A output>  (job D, after jobs A and B)

Every task writes fig_<name>.pdf (where a figure exists) and <task>.json into
the run directory: all quoted numbers, estimand definitions, cohort sizes,
check results, input/helper sha256 and the exact command. manifest.json is
written last with each task's status; a run is a completed refresh only if
manifest.json exists and every requested task has status "ok". The run
directory must not exist beforehand. Nothing outside it is written.

Tasks
  w1  clean CNN, scalar/mass baselines (outer CV on each seed's reconstructed
      CNN partition), within-original-fold calibration experiment and
      held-out tracking diagnostics, mass terciles, partial rank correlation
  w2  Minkowski-functional XGBoost at 144 MHz and (paired) 1.4 GHz
  w3  phase diagram / gas-phase fractions (extract_phase_syn.py output) and
      the 144 MHz vs 1.4 GHz frequency comparison
  w4  aperture retention (1 and 3 Mpc boxes)
  w5  radio gallery and realism curves (check_radio_realism definition)
  w6  predicted vs true pseudo-TSC and age-binned residuals
  w7  power-mass relation (log of arithmetic-mean linear power)
  w8  LoVoCCS radio vs X-ray observed predictions
  w9  projected outermost-peak diagnostic (bounded redetection)
"""
import argparse
import hashlib
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

IN = dict(ds128="dataset_syn_nh4_144_128.h5", ds512="dataset_syn_nh4_144_512.h5",
          shape128="dataset_syn_nh4_144_128_shape.h5", ds1400="dataset_syn_nh4_1400_128.h5",
          inj3="injected_syn_phys_b3000.h5", inj1="injected_syn_phys_b1000.h5",
          catalog="Radio_Data/TNG-Cluster_Catalog.hdf5", relic_cat="relic_catalog_radio_syn_nh4_144.h5",
          tsc_legacy="TSC_Cutimages/TSC_eachhalo_snap99.hdf5", ds_legacy="dataset.h5",
          wl="lovoccs_wl_masses.csv")
PRED = dict(clean="cnn_preds/pooled_syn_inner_s{}.npz", shape="cnn_preds/pooled_shape_syn_inner_s{}.npz",
            joint1="syn_phys_b1000_joint_s{}_preds.npz", radio3="syn_phys_b3000_sfimg_s{}_preds.npz",
            xray="fix_x_s{}_preds.npz")
HELPERS = ["paper/scripts/regen_stale.py", "minkowski_functionals.py", "check_radio_realism.py",
           "detect_relics_radio.py", "train_cnn_pooled.py", "compare_syn.py", "forward_model_lotss.py",
           "extract_phase_syn.py", "build_radio_cells.py", "build_dataset.py", "build_injected_dataset.py"]


class Fail(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


# ======================================================================= pure helpers
def r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    if len(y) < 2 or not np.var(y) > 0:
        return np.nan
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


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
    5-fold inner CV on the fitting rows only (13-value grid)."""
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
             mean=est[0].mean_.tolist(), scale=est[0].scale_.tolist())


def outer_cv(X, y, fold, seed):
    """Ordinary outer CV on a given partition; inner alpha CV in training folds."""
    pred, alphas = np.full(len(y), np.nan), []
    for f in range(N_FOLD):
        tr, te = fold != f, fold == f
        p, a, _ = ridge_fit_predict(X[tr], y[tr], X[te], inner_seed=seed * 10 + f)
        pred[te], alphas = p, alphas + [a]
    return pred, alphas


def within_fold_calibration(ids, y, fold, arms, seed, min_n=MIN_CAL):
    """Second-stage fits confined to halves of ONE original outer fold (the
    first-stage CNN saw neither half). arms: name -> (n, k) feature matrix.
    Returns per-arm held-out predictions (each parent exactly once) and the
    full per-direction record; coverage must be complete or the arm is
    unavailable."""
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
    full = {a: bool(all(d["status"] == "ok" for d in dirs) and (npred[a] == 1).all()) for a in arms}
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


def collapse_realisations(logp, tol=TOL_REAL_DEX):
    """(N, R, P) box log-powers must repeat across realisations; returns (N, P)."""
    logp = np.asarray(logp, np.float64)
    d = np.nanmax(np.abs(logp - logp[:, :1, :]))
    if not d <= tol:
        raise Fail(f"box power differs across realisations by {d:.2e} dex (> {tol})")
    return logp[:, 0, :], float(d)


def residual_bins(y, p, edges=RESID_EDGES):
    """Median and 16-84 of p - y per bin; last bin closed at the top (4 Gyr)."""
    y, r = np.asarray(y, float), np.asarray(p, float) - np.asarray(y, float)
    if (y > edges[-1] + 1e-9).any() or (y < edges[0]).any():
        raise Fail("label outside the binning range")
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


def halo_summary(d_max):
    """(N, P) outermost-peak distance, NaN = no peak: mean over projections
    with a peak; halos with none are excluded (count returned)."""
    d = np.asarray(d_max, float)
    n_with = np.isfinite(d).sum(1)
    with np.errstate(invalid="ignore"):
        m = np.where(n_with > 0, np.nansum(np.where(np.isfinite(d), d, 0), 1) / np.maximum(n_with, 1), np.nan)
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


# ======================================================================= run context
class Run:
    def __init__(self, run_dir, argv):
        if os.path.exists(run_dir):
            raise SystemExit(f"refusing to reuse existing run directory {run_dir}")
        os.makedirs(run_dir)
        self.dir, self.argv, self.t0 = run_dir, argv, time.time()
        self.head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.dirty = subprocess.run(["git", "status", "--porcelain", "--"] + HELPERS,
                                    capture_output=True, text=True).stdout.strip()
        self.helpers = {h: sha(h) for h in HELPERS if os.path.exists(h)}
        self.status = {}

    def path(self, name):
        p = os.path.join(self.dir, name)
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
        return p

    def save_fig(self, fig, name):
        import matplotlib.pyplot as plt
        fig.savefig(self.path(f"fig_{name}.pdf"))
        plt.close(fig)

    def record(self, task, data, inputs):
        rec = dict(task=task, command=" ".join([sys.executable] + self.argv), git_head=self.head,
                   uncommitted_helpers=self.dirty, helper_sha256=self.helpers,
                   input_sha256={k: sha(v) for k, v in inputs.items()}, threads=os.environ.get("OMP_NUM_THREADS"),
                   result=data)
        with open(self.path(f"{task}.json"), "w") as f:
            json.dump(jsonable(rec), f, indent=1)

    def manifest(self, requested):
        m = dict(requested=requested, status=self.status, git_head=self.head, command=" ".join(self.argv),
                 seconds=round(time.time() - self.t0, 1),
                 complete=all(self.status.get(t) == "ok" for t in requested))
        with open(self.path("manifest.json"), "w") as f:
            json.dump(m, f, indent=1)
        return m["complete"]


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


def clean_inputs():
    """Clean maps, scalars and the two pooled-CNN prediction families, with
    identity/label/fold-consistency checks."""
    import h5py
    with h5py.File(IN["ds128"]) as f:
        hid, y = f["meta/halo_id"][:], f["labels/pseudo_tsc"][:].astype(np.float64)
        imgs, r500 = f["images"][:].astype(np.float64), f["meta/r500c_kpc"][:].astype(np.float64)
        logm_ds = f["meta/log_m500"][:].astype(np.float64)
    with h5py.File(IN["shape128"]) as f:
        check(np.array_equal(f["meta/halo_id"][:], hid), "shape dataset halo order differs")
        y_sh = f["labels/pseudo_tsc"][:]
    with h5py.File(IN["catalog"]) as f:
        mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
    logm = np.array([mm[h] for h in hid], dtype=np.float64)
    lin = np.log10(np.sinh(imgs).sum(axis=(1, 2, 3)) + 1)
    X5 = np.column_stack([logm, np.log10(r500), lin, imgs.mean(axis=(1, 2, 3)), (imgs > 0).mean(axis=(1, 2, 3))])
    n = len(hid)
    info = dict(n=n, logm_catalog_vs_dataset_maxdiff=float(np.max(np.abs(logm - logm_ds))),
                shape_label_maxdiff=float(np.max(np.abs(y_sh - y))))
    P, S, folds, fold_checks = {}, {}, {}, {}
    for s in SEEDS:
        fold = reconstruct_folds(n, s)
        folds[s] = fold
        for fam, store in (("clean", P), ("shape", S)):
            d = load_npz(PRED[fam].format(s))
            check(np.array_equal(d["halo_id"], hid), f"{fam} s{s}: halo order differs from {IN['ds128']}")
            check(np.allclose(d["label"], y, atol=1e-6), f"{fam} s{s}: labels differ")
            check((d["snap"] == 99).all() and int(d["seed"]) == s, f"{fam} s{s}: snapshot/seed metadata")
            rec = np.array([r2(y[fold == f], d["oof_pred"][fold == f]) for f in range(N_FOLD)])
            dmax = float(np.max(np.abs(rec - d["fold_r2"])))
            check(dmax < 1e-4, f"{fam} s{s}: reconstructed partition does not reproduce saved fold R2 "
                               f"(max diff {dmax:.2e})")
            fold_checks[f"{fam}_s{s}"] = dmax
            store[s] = d["oof_pred"].astype(np.float64)
    info["fold_r2_reproduction_maxdiff"] = fold_checks
    info["fold_provenance"] = ("partitions reconstructed with train_cnn_pooled.group_split + KFold(5, shuffle, "
                               "random_state=seed) in the saved row order; agreement with the five saved fold R2 "
                               "values is a consistency check, not recorded original membership")
    return hid, y, logm, lin, X5, P, S, folds, info


# ======================================================================= W1
def task_w1(run, a):
    plt = style()
    hid, y, logm, lin, X5, P, S, folds, info = clean_inputs()
    n = len(y)
    out = dict(inputs_check=info, estimands={}, per_seed={})
    seeds_r2 = lambda store: np.array([r2(y, store[s]) for s in SEEDS])
    cnn_r2, shape_r2 = seeds_r2(P), seeds_r2(S)
    ens = np.mean([P[s] for s in SEEDS], 0)
    out["cnn"] = dict(per_seed=cnn_r2, mean=cnn_r2.mean(), sd=cnn_r2.std(ddof=1), ensemble_r2=r2(y, ens))
    out["shape_cnn"] = dict(per_seed=shape_r2, mean=shape_r2.mean(), sd=shape_r2.std(ddof=1))
    # ordinary outer-CV scalar baselines on each seed's CNN test partition
    base = {"mass": logm[:, None], "integrated emission": lin[:, None], "5 scalars": X5}
    bpred = {k: {} for k in base}
    for k, X in base.items():
        sc, al = [], []
        for s in SEEDS:
            p, alphas = outer_cv(X, y, folds[s], s)
            bpred[k][s] = p
            sc.append(r2(y, p)), al.append(alphas)
        out[f"baseline_{k}"] = dict(per_seed=sc, mean=np.mean(sc), sd=np.std(sc, ddof=1), alphas=al)
    out["estimands"]["baselines"] = ("ridge (scaler in pipeline, alpha by 5-fold inner CV within the outer "
                                     "training folds), outer test partitions = each seed's reconstructed CNN "
                                     "partition; matching test partitions, not identical CNN fitting/selection "
                                     "sets; mean +- sd over the 5 partitions")
    # within-original-fold calibration experiment and tracking diagnostics
    cal = {}
    for s in SEEDS:
        arms = {"scalars": X5, "scalars+CNN": np.column_stack([X5, P[s]])}
        pred, full, dirs = within_fold_calibration(hid, y, folds[s], arms, s)
        tr_pred, tr_full, tr_dirs = within_fold_calibration(hid, P[s], folds[s], {"scalars->CNN": X5}, s)
        sh_pred, sh_full, sh_dirs = within_fold_calibration(hid, S[s], folds[s],
                                                            {"emission->shape CNN": lin[:, None]}, s)
        cal[s] = dict(pred=pred, full=full, dirs=dirs, tr=(tr_pred, tr_full, tr_dirs), sh=(sh_pred, sh_full, sh_dirs))
    allfull = all(c["full"]["scalars"] and c["full"]["scalars+CNN"] for c in cal.values())
    res = dict(available=allfull, per_seed={})
    if allfull:
        a_ = np.array([r2(y, cal[s]["pred"]["scalars"]) for s in SEEDS])
        b_ = np.array([r2(y, cal[s]["pred"]["scalars+CNN"]) for s in SEEDS])
        rng = np.random.default_rng(BOOT_SEED)
        bs, bad = [], 0
        PA = np.array([cal[s]["pred"]["scalars"] for s in SEEDS])
        PB = np.array([cal[s]["pred"]["scalars+CNN"] for s in SEEDS])
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
                     f"evaluated on the other, then reversed; scaler and alpha (5-fold inner CV, 13 values) "
                     f"confined to the fitting half; min half {MIN_CAL}; each parent predicted once per arm; "
                     f"calibration halves hold ~35 parents, far fewer than the full-training stack behind the "
                     f"historical number")
    res["directions"] = {s: cal[s]["dirs"] for s in SEEDS}
    res["predictions"] = {s: cal[s]["pred"] for s in SEEDS}
    out["calibration_experiment"] = res
    trk = {}
    for key_, lab in (("tr", "scalars->CNN"), ("sh", "emission->shape CNN")):
        ok = all(cal[s][key_][1][lab] for s in SEEDS)
        resp = P if key_ == "tr" else S
        v = np.array([r2(resp[s], cal[s][key_][0][lab]) for s in SEEDS]) if ok else None
        trk[lab] = dict(available=ok, per_seed=v, mean=None if v is None else v.mean(),
                        sd=None if v is None else v.std(ddof=1),
                        directions={s: cal[s][key_][2] for s in SEEDS},
                        predictions={s: cal[s][key_][0][lab] for s in SEEDS},
                        estimand="held-out R2 of the saved OOF output of the same seed, within-original-fold halves")
    out["tracking"] = trk
    # terciles (per-seed primary; ensemble labelled) and partial rank correlation
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
    out["folds"] = {s: folds[s] for s in SEEDS}
    out["halo_id"] = hid
    # figure
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
    yy = np.arange(len(rows))[::-1]
    for y_, (lab, v, e, c, h) in zip(yy, rows):
        ax[0].barh(y_, v, xerr=e, color=c, hatch=h, edgecolor="k" if h else None, lw=0.4 if h else 0,
                   capsize=2, height=0.7)
        ax[0].text(v + (e or 0) + 0.01, y_, f"{v:.3f}", va="center", fontsize=6.5)
    ax[0].set_yticks(yy, [r_[0] for r_ in rows])
    ax[0].set_xlabel("$R^2$ for pseudo-TSC")
    ax[0].set_xlim(0, 0.7)
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
    run.record("w1", out, {k: IN[k] for k in ("ds128", "shape128", "catalog")}
               | {f"{fam}_s{s}": PRED[fam].format(s) for fam in ("clean", "shape") for s in SEEDS})
    print(f"  CNN {cnn_r2.mean():.3f} +- {cnn_r2.std(ddof=1):.3f}; ensemble {out['cnn']['ensemble_r2']:.3f}; "
          f"5 scalars {out['baseline_5 scalars']['mean']:.3f}; calibration available {allfull}")


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
    from xgboost import XGBRegressor
    from minkowski_functionals import FRACS, oof_r2
    plt = style()
    out, curves_by = {}, {}
    sets = [("144 MHz", IN["ds128"]), ("1.4 GHz", IN["ds1400"])]
    ref = None
    for lab, p in sets:
        with h5py.File(p) as f:
            imgs, hid, y = f["images"][:], f["meta/halo_id"][:], f["labels/pseudo_tsc"][:].astype(float)
            attrs = {k: f.attrs[k] for k in ("extent_r500", "center_mode", "spread_cells", "cells_dir")
                     if k in f.attrs}
        if ref is None:
            ref = (hid, y, attrs)
        else:
            check(np.array_equal(hid, ref[0]) and np.allclose(y, ref[1]), "frequency sets differ in identity/labels")
            for k in ("extent_r500", "center_mode", "spread_cells"):
                check(str(attrs.get(k)) == str(ref[2].get(k)), f"frequency sets differ in {k}")
        N, P = imgs.shape[:2]
        curves, ties = mf_block(imgs)
        feats = curves[:, :, :, 1:].reshape(N * P, -1)
        yy, groups = np.repeat(y, P), np.repeat(np.arange(N), P)
        r2p, r2c, oof = oof_r2(feats, yy, groups, lambda: XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=0, n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1"))))
        cm = curves.mean(1)
        j = int(np.argmin(np.abs(FRACS - 0.5)))
        rho = spearmanr(cm[:, j, 2], y)
        out[lab] = dict(dataset=p, attrs=attrs, n_parents=N, n_projections=N * P, n_features=feats.shape[1],
                        r2_projection=r2p, r2_parent_mean=r2c, area_fraction_ties=ties,
                        components_at_f05_spearman=float(rho[0]), components_at_f05_p=float(rho[1]),
                        f_threshold=float(FRACS[j]), oof_projection=oof,
                        oof_parent_mean=np.array([oof[groups == g].mean() for g in range(N)]))
        curves_by[lab] = (curves, cm, y)
    out["difference_1400_minus_144"] = dict(
        projection=out["1.4 GHz"]["r2_projection"] - out["144 MHz"]["r2_projection"],
        parent_mean=out["1.4 GHz"]["r2_parent_mean"] - out["144 MHz"]["r2_parent_mean"])
    out["protocol"] = ("XGBoost fixed hyperparameters (400 trees, depth 4, lr 0.05, subsample/colsample 0.8, "
                       "seed 0), 48 features = 12 area-fraction thresholds x {perimeter, components, holes, "
                       "Euler}; parent GroupKFold(5) (deterministic, identical rows at both frequencies; not the "
                       "CNN partitions); scaler fitted on training folds; f = 0.5 is the lowest intensity "
                       "threshold")
    curves, cm, y = curves_by["144 MHz"]
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
    ax[2].scatter(y, out["144 MHz"]["oof_parent_mean"], s=4, alpha=0.5, color=C_SIM)
    ax[2].plot([0, 4], [0, 4], "k:", lw=0.8)
    ax[2].set_xlabel("pseudo-TSC [Gyr]")
    ax[2].set_ylabel("predicted from topology [Gyr]")
    ax[2].set_title(f"$R^2$ = {out['144 MHz']['r2_parent_mean']:.2f} (parent mean)")
    fig.tight_layout()
    run.save_fig(fig, "mf_tsc")
    run.record("w2", out, {"ds128": IN["ds128"], "ds1400": IN["ds1400"]})
    print(f"  MF 144: {out['144 MHz']['r2_projection']:.3f}/{out['144 MHz']['r2_parent_mean']:.3f}; "
          f"1.4 GHz: {out['1.4 GHz']['r2_projection']:.3f}/{out['1.4 GHz']['r2_parent_mean']:.3f}")


# ======================================================================= W3
def task_w3(run, a):
    import h5py
    plt = style()
    check(a.phase_npz and os.path.exists(a.phase_npz), "--phase-npz (job A output) is required for w3")
    d = load_npz(a.phase_npz)
    pv = json.loads(str(d["provenance"]))
    check(pv["n_check_failures"] == 0, "job A reported cell-file check failures")
    out = dict(job_a=dict(path=a.phase_npz, provenance=pv))
    pct = lambda v: dict(median=float(np.nanmedian(v)), p16=float(np.nanpercentile(v, 16)),
                         p84=float(np.nanpercentile(v, 84)), n=int(np.isfinite(v).sum()))
    out["frac_above_nh_cut"] = pct(d["frac_above"])
    out["frac_cold_kT_lt_0.1keV"] = pct(d["frac_cold"])
    out["top_cell_share_all"] = pct(d["top_all"])
    out["top_cell_share_cut"] = pct(d["top_cut"])
    out["weighted_mach_all"] = pct(d["mach_all"])
    out["weighted_mach_cut"] = pct(d["mach_cut"])
    out["cells"] = dict(total=int(d["n_cells"].sum()), invalid=int(d["n_invalid"].sum()),
                        zero_weight=int(d["n_zero_w"].sum()), out_of_range=int(d["n_out_of_range"].sum()),
                        weight_out_of_range_frac_median=float(np.median(d["w_out_of_range"] / d["w_total"])),
                        cut_status={s: int((d["cut_status"] == s).sum()) for s in np.unique(d["cut_status"])})
    G = float(np.nanmedian(d["ratio_cut"]))
    out["frequency_cells"] = dict(
        G_median_ratio_cut=G, sd_log10_ratio_cut=float(np.nanstd(np.log10(d["ratio_cut"]), ddof=1)),
        G_median_ratio_all=float(np.nanmedian(d["ratio_all"])),
        sd_log10_ratio_all=float(np.nanstd(np.log10(d["ratio_all"]), ddof=1)),
        n=int(np.isfinite(d["ratio_cut"]).sum()), per_parent_ratio_cut=d["ratio_cut"],
        estimand="per parent sum(w144)/sum(w1400) over density-cut cells, unscaled physical weights; SD over "
                 "parents (ddof 1) of log10 of that ratio; w1400 = w144 (1.4/0.144)^(-s/2) at fixed cell "
                 "properties (implementation check, not independent physics validation)")
    # map-level redistribution on identical grids
    with h5py.File(IN["ds128"]) as f1, h5py.File(IN["ds1400"]) as f2:
        for k in ("extent_r500", "center_mode", "spread_cells"):
            check(str(f1.attrs.get(k)) == str(f2.attrs.get(k)), f"map sets differ in {k}")
        check(np.array_equal(f1["meta/halo_id"][:], f2["meta/halo_id"][:]), "map sets differ in halo order")
        check(not bool(f1.attrs.get("spread_cells")), "current maps must be point-deposited (spread_cells False)")
        m1, m2 = f1["images"][:].astype(np.float64), f2["images"][:].astype(np.float64)
        hid = f1["meta/halo_id"][:]
    N, P = m1.shape[:2]
    tv = np.array([[tv_distance(np.sinh(m1[i, p]), np.sinh(m2[i, p])) for p in range(P)] for i in range(N)])
    pm = np.nanmean(tv, 1)
    out["redistribution_tv"] = dict(parent_mean=pct(pm), pooled_projection=pct(tv.ravel()),
                                    n_unavailable_maps=int(np.isnan(tv).sum()), per_map=tv, halo_id=hid,
                                    estimand="0.5 sum |p144 - p1400| between unit-sum linear (sinh) 128-px maps; "
                                             "primary: median over parents of the projection mean")
    # phase figure (mean over parents of per-parent normalised histograms)
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
    ax[1].text(0.03, 0.97, f"{out['frac_above_nh_cut']['median']:.0%} of the weight\nat $n_{{\\rm H}} > 10^{{-4}}$",
               transform=ax[1].transAxes, ha="left", va="top", fontsize=7)
    fig.tight_layout()
    run.save_fig(fig, "phase")
    run.record("w3", out, {"phase_npz": a.phase_npz, "ds128": IN["ds128"], "ds1400": IN["ds1400"]})
    print(f"  weight above cut {out['frac_above_nh_cut']['median']:.3f}; G {G:.3f}; "
          f"TV parent-mean median {out['redistribution_tv']['parent_mean']['median']:.3f}")


# ======================================================================= W4 / W7 shared
def map_totals(ds512):
    """Per (parent, projection) full-footprint linear sum of the 512 maps."""
    import h5py
    with h5py.File(ds512) as f:
        hid, r500 = f["meta/halo_id"][:], f["meta/r500c_kpc"][:].astype(np.float64)
        ext = float(f.attrs["extent_r500"])
        N, P, H, _ = f["images"].shape
        T = np.zeros((N, P))
        for i in range(N):
            lin = np.sinh(f["images"][i].astype(np.float64))
            T[i] = lin.sum(axis=(1, 2))
    return hid, r500, ext, H, T


def inj_box(path, hid, ds512):
    import h5py
    with h5py.File(path) as f:
        at = dict(f.attrs)
        check(np.array_equal(f["mock/halo_id"][:], hid), f"{path}: halo order differs")
        check(str(at.get("source_dataset")) == ds512 and str(at.get("recipe")) == "physical"
              and str(at.get("flux_mode")) == "sim", f"{path}: unexpected source/recipe/flux mode")
        logp = f["mock/logp150"][:]
        lm, tsc = f["mock/log_m500"][:].astype(float), f["mock/pseudo_tsc"][:].astype(float)
    off = float(at["sim_flux_offset_dex"])
    lb, dmax = collapse_realisations(logp)
    return lb, off, float(at["box_kpc"]), dmax, lm, tsc


def task_w4(run, a):
    import h5py
    hid, r500, ext, H, T = map_totals(IN["ds512"])
    out = dict(extent_r500=ext, img_size=H)
    offs = []
    for tag in ("inj3", "inj1"):
        lb, off, box, dmax, _, _ = inj_box(IN[tag], hid, IN["ds512"])
        offs.append(off)
        B = 10.0 ** (lb - off)
        F = B / T
        # independent fractional-overlap crop of every (parent, projection)
        diff = []
        with h5py.File(IN["ds512"]) as f:
            for i in range(len(hid)):
                px = 2 * ext * r500[i] / H
                lin = np.sinh(f["images"][i].astype(np.float64))
                for p in range(lin.shape[0]):
                    diff.append(np.log10(box_sum(lin[p], box / 2 / px)) - (lb[i, p] - off))
        diff = np.array(diff)
        check(np.nanmax(np.abs(diff)) < TOL_CROP_DEX, f"{tag}: independent crop disagrees with logp150 by "
                                                      f"{np.nanmax(np.abs(diff)):.2e} dex")
        out[f"box_{box / 1000:.0f}Mpc"] = dict(
            file=IN[tag], box_kpc=box, offset_dex=off, realisation_maxdiff_dex=dmax,
            crop_check_max_abs_dex=float(np.nanmax(np.abs(diff))), n_nan=int(np.isnan(F).sum()),
            median_projection=float(np.nanmedian(F)), p16=float(np.nanpercentile(F, 16)),
            p84=float(np.nanpercentile(F, 84)), median_parent_mean=float(np.nanmedian(np.nanmean(F, 1))),
            aggregate_sum_ratio=float(np.nansum(B) / np.nansum(T[np.isfinite(B)])), per_projection=F)
    check(abs(offs[0] - offs[1]) < 1e-12, "the two injected files use different flux offsets")
    out["estimand"] = ("linear emission inside the centred physical box (forward model's fractional-overlap "
                       "crop, 10**(logp150 - offset)) over the full projected 512-px map footprint "
                       "(+-extent_r500 R500); primary: median over parent/projection; also parent-mean median "
                       "and aggregate sum ratio")
    run.record("w4", out, {k: IN[k] for k in ("ds512", "inj3", "inj1")})
    print("  " + "; ".join(f"{k}: {v['median_projection']:.3f}" for k, v in out.items() if k.startswith("box_")))


def task_w7(run, a):
    import h5py
    from scipy import stats
    plt = style()
    hid, r500, ext, H, T = map_totals(IN["ds512"])
    lb, off, box, dmax, lm, tsc = inj_box(IN["inj3"], hid, IN["ds512"])
    with h5py.File(IN["ds512"]) as f:
        check(np.allclose(f["meta/log_m500"][:], lm, atol=1e-5), "log M500 differs between files")
        check(np.allclose(f["labels/pseudo_tsc"][:], tsc, atol=1e-6), "pseudo-TSC differs between files")
    lp_t = np.log10(T.mean(1)) + off
    lp_b = np.log10((10.0 ** (lb - off)).mean(1)) + off
    hist_b = lb.mean(1)
    ok = np.isfinite(lp_t) & np.isfinite(lp_b) & np.isfinite(lm)
    out = dict(n=int(ok.sum()), n_excluded=int((~ok).sum()), offset_dex=off, extent_r500=ext, box_kpc=box,
               averaging="log10 of the arithmetic-mean linear power over projections for both apertures "
                         "(realisations collapsed first); corrects the old mixed arithmetic/geometric recipe",
               calibration="one global additive offset (sim flux mode); it does not impose the slope and does "
                           "not independently calibrate absolute power; 144 MHz emission on the 150 MHz scale")
    fits = {}
    for name, lp in (("total", lp_t), ("box_3Mpc", lp_b), ("box_3Mpc_historical_mean_log", hist_b)):
        fit = stats.linregress(lm[ok], lp[ok])
        res = lp[ok] - (fit.intercept + fit.slope * lm[ok])
        rho = stats.spearmanr(tsc[ok], res)
        fits[name] = dict(slope=fit.slope, slope_se=fit.stderr, intercept=fit.intercept,
                          rms_residual_dex=float(np.sqrt(np.mean(res ** 2))),
                          resid_vs_tsc_spearman=float(rho[0]), resid_vs_tsc_p=float(rho[1]), residuals=res)
    out["fits"] = fits
    fig, ax = plt.subplots(1, 3, figsize=(TEXT, TEXT * 0.3))
    xx = np.linspace(lm.min(), lm.max(), 10)
    for a_, name, lp, lab in [(ax[0], "total", lp_t, r"log $P$, full map [W Hz$^{-1}$]"),
                              (ax[1], "box_3Mpc", lp_b, rf"log $P$, {box / 1000:.0f} Mpc box [W Hz$^{{-1}}$]")]:
        ft = fits[name]
        sc = a_.scatter(lm, lp, c=np.minimum(tsc, 4), s=4, cmap="plasma_r")
        x0, y0 = np.median(lm[ok]), np.median(lp[ok])
        a_.plot(xx, ft["intercept"] + ft["slope"] * xx, "k-", lw=1,
                label=f"fit: {ft['slope']:.2f}, {ft['rms_residual_dex']:.2f} dex")
        a_.plot(xx, y0 + 2.83 * (xx - x0), "k--", lw=0.8, label="relics: 2.83")
        a_.plot(xx, y0 + 3.55 * (xx - x0), "k:", lw=0.8, label="haloes: 3.55")
        a_.set_xlabel(r"log $M_{500}$ [M$_\odot$]")
        a_.set_ylabel(lab)
        a_.legend(fontsize=5.5, loc="upper left")
    fig.colorbar(sc, ax=ax[1], pad=0.01, label="pseudo-TSC [Gyr]")
    ax[2].scatter(np.minimum(tsc[ok], 4), fits["total"]["residuals"], s=4, alpha=0.5, color=C_SIM)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel("pseudo-TSC [Gyr]")
    ax[2].set_ylabel("full-map residual [dex]")
    ax[2].set_title(f"Spearman {fits['total']['resid_vs_tsc_spearman']:+.2f}", fontsize=7)
    fig.tight_layout()
    run.save_fig(fig, "pm_relation")
    run.record("w7", out, {k: IN[k] for k in ("ds512", "inj3")})
    print("  " + "; ".join(f"{k} {v['slope']:.2f}+-{v['slope_se']:.2f}, {v['rms_residual_dex']:.2f} dex"
                           for k, v in fits.items()))


# ======================================================================= W5
def task_w5(run, a):
    import h5py
    from check_radio_realism import KS, NAMES, mfs, separation
    plt = style()
    with h5py.File(IN["inj3"]) as f:
        mock_all = f["mock/images"][:]
        mh, tsc, mz = f["mock/halo_id"][:], f["mock/pseudo_tsc"][:], f["mock/z"][:]
        obs, oz = f["obs/images"][:], f["obs/z"][:]
        on = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/name"][:]]
    with h5py.File(IN["ds512"]) as f:
        simh, r500 = f["meta/halo_id"][:], f["meta/r500c_kpc"][:]
        order = {h: i for i, h in enumerate(simh)}
        pick = [int(np.argmin(np.abs(tsc - t))) for t in (0.2, 0.8, 1.8, 4.0)]
        check(len(set(pick)) == 4, "gallery selection picked a parent twice")
        clean = [f["images"][order[mh[i]], 0] for i in pick]
        ext = float(f.attrs["extent_r500"])
    targets = ["A399", "A2443", "A1650", "A2033"]
    oi = [on.index(t) for t in targets if t in on]
    check(len(oi) == 4, f"gallery real targets missing: {[t for t in targets if t not in on]}")
    gal = dict(parents=[dict(halo_id=int(mh[i]), pseudo_tsc=float(tsc[i]), z=float(mz[i, 0]),
                             r500_kpc=float(r500[order[mh[i]]]), realisation=0, projection=0) for i in pick],
               real=[dict(name=on[k], z=float(oz[k])) for k in oi],
               display=dict(clean=f"arcsinh map, central +-{ext / 2:g} R500 of the +-{ext:g} R500 512-px map",
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
    # realism curves with the adopted check_radio_realism definition
    M = mfs(mock_all.reshape(-1, *mock_all.shape[-2:]))
    O = mfs(obs)
    sep = separation(O, M)
    D = float(np.median(np.abs(sep)))
    above3 = KS > 3.0
    out = dict(gallery=gal, n_mock=len(M), n_obs=len(O), thresholds_sigma=KS, functionals=NAMES,
               separation=sep, D=D, D_rounded=round(D, 3), D_expected_rounded=0.093,
               D_reproduced=bool(round(D, 3) == 0.093), max_abs_sep_above_3sigma=float(np.abs(sep[above3]).max()),
               mock_sd_convention="numpy std, ddof 0, +1e-9 (check_radio_realism.separation)")
    check(out["D_reproduced"], f"realism D {D:.4f} does not reproduce the retained 0.093 -- investigate the "
                               f"definition/provenance, do not tune")
    fig, ax = plt.subplots(2, 4, figsize=(TEXT, TEXT * 0.45), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
    for c in range(4):
        lo, med, hi = np.percentile(M[:, :, c], [16, 50, 84], axis=0)
        ax[0, c].fill_between(KS, lo, hi, color=C_SIM, alpha=0.25)
        ax[0, c].plot(KS, med, color=C_SIM, label=f"mocks ({len(M)})")
        olo, omed, ohi = np.percentile(O[:, :, c], [16, 50, 84], axis=0)
        ax[0, c].errorbar(KS, omed, yerr=[omed - olo, ohi - omed], fmt="o", ms=3, color=C_OBS,
                          label=f"LoTSS ({len(O)})")
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
    run.record("w5", out, {k: IN[k] for k in ("inj3", "ds512")})
    print(f"  realism D {D:.4f} (reproduced {out['D_reproduced']}); gallery parents "
          f"{[p['halo_id'] for p in gal['parents']]}")


# ======================================================================= W6
def task_w6(run, a):
    import compare_syn as CS
    plt = style()
    hid, y_c, _, _, _, P, _, _, info = clean_inputs()
    p_c = np.mean([P[s] for s in SEEDS], 0)
    ds = CS.load(PRED["joint1"])
    m = ds[0]["snap"] == 99
    for d in ds[1:]:
        check(np.array_equal(d["group_id"], ds[0]["group_id"]) and np.array_equal(d["tsc"], ds[0]["tsc"]),
              "joint seeds differ in identity/labels")
    y_r = ds[0]["tsc"][m].astype(float)
    p_r = np.mean([d["oof_cluster"] for d in ds], 0)[m]
    yv, gv, preds = CS.single(ds)
    sv = r2(yv, np.mean(preds, 0))
    out = dict(clean=dict(estimand="pooled clean CNN, 5-seed mean of OOF predictions, snapshot 99", n=len(y_c),
                          r2=r2(y_c, p_c), bins=residual_bins(y_c, p_c)),
               realistic=dict(estimand="1 Mpc-radio + X-ray joint model (syn_phys_b1000_joint), snapshot-99 "
                                       "rows, pooled views with realisations averaged (oof_cluster), 5-seed mean",
                              n=int(m.sum()), r2=r2(y_r, p_r), bins=residual_bins(y_r, p_r),
                              single_view_5model_r2=sv, single_view_expected_rounded=0.426,
                              single_view_reproduced=bool(round(sv, 3) == 0.426)))
    for k, (y, p) in (("clean", (y_c, p_c)), ("realistic", (y_r, p_r))):
        cap = np.isclose(y, TSC_CAP)
        out[k]["at_cap"] = dict(n=int(cap.sum()), median_residual=float(np.median(p[cap] - y[cap]))
                                if cap.any() else None)
        out[k]["youngest_bin"] = out[k]["bins"][0]
    check(out["realistic"]["single_view_reproduced"], f"single-view score {sv:.4f} does not reproduce 0.426")
    fig, ax = plt.subplots(1, 3, figsize=(TEXT, TEXT * 0.32))
    for a_, y, p, lab, c in [(ax[0], y_c, p_c, "clean radio (144 MHz)", C_SIM),
                             (ax[1], y_r, p_r, "1 Mpc radio + X-ray", C_OBS)]:
        a_.scatter(y, p, s=4, alpha=0.5, color=c)
        a_.plot([0, 4.2], [0, 4.2], "k:", lw=0.8)
        a_.set_xlim(-0.1, 4.2), a_.set_ylim(-0.1, 4.2)
        a_.set_xlabel("true pseudo-TSC [Gyr]")
        a_.set_title(f"{lab}: $R^2$ = {r2(y, p):.2f}", fontsize=7)
    ax[0].set_ylabel("predicted [Gyr]")
    mid = 0.5 * (RESID_EDGES[1:] + RESID_EDGES[:-1])
    for k, lab, c, dx in [("clean", "clean", C_SIM, -0.04), ("realistic", "1 Mpc radio + X-ray", C_OBS, 0.04)]:
        b = out[k]["bins"]
        okb = np.array([x["status"] == "ok" for x in b])
        med = np.array([x.get("median", np.nan) for x in b])
        lo = np.array([x.get("p16", np.nan) for x in b])
        hi = np.array([x.get("p84", np.nan) for x in b])
        ax[2].errorbar(mid[okb] + dx, med[okb], yerr=[med[okb] - lo[okb], hi[okb] - med[okb]], fmt="o-", ms=3,
                       color=c, label=lab, capsize=2, lw=0.9)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel("true pseudo-TSC [Gyr]")
    ax[2].set_ylabel("predicted $-$ true [Gyr]")
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    run.save_fig(fig, "tsc_pred")
    run.record("w6", out, {"ds128": IN["ds128"]} | {f"clean_s{s}": PRED["clean"].format(s) for s in SEEDS}
               | {f"joint1_s{s}": "cnn_preds/" + PRED["joint1"].format(s) for s in SEEDS})
    print(f"  clean R2 {out['clean']['r2']:.3f}; realistic pooled {out['realistic']['r2']:.3f}; single-view {sv:.3f}")


# ======================================================================= W8
def task_w8(run, a):
    import pandas as pd
    from scipy import stats
    import compare_syn as CS
    plt = style()
    ens = {}
    for tag in ("radio3", "xray"):
        ds = CS.load(PRED[tag])
        names = [str(n) for n in ds[0]["obs_name"]]
        keys = [norm_key(n) for n in names]
        check(len(set(keys)) == len(keys), f"{tag}: duplicate normalised target keys")
        for d in ds:
            check([str(n) for n in d["obs_name"]] == names, f"{tag}: target order differs across seeds")
            check(np.allclose(d["obs_pred"], d["obs_pred_folds"].mean(0), atol=1e-6),
                  f"{tag}: obs_pred is not the fold mean")
        Pm = np.array([d["obs_pred"] for d in ds])           # (seeds, targets), fold-averaged
        ens[tag] = {k: (Pm[:, i].mean(), Pm[:, i].std(ddof=1), names[i]) for i, k in enumerate(keys)}
    both = sorted(set(ens["radio3"]) & set(ens["xray"]))
    r = np.array([ens["radio3"][k][0] for k in both])
    x = np.array([ens["xray"][k][0] for k in both])
    rho, p = stats.spearmanr(r, x)
    dr = r - x
    wl = pd.read_csv(IN["wl"]).set_index("key")
    lmv = np.array([float(wl.loc[k, "log_m500c"]) if k in wl.index else np.nan for k in both])
    out = dict(n=len(both), targets=[ens["radio3"][k][2] for k in both], radio=r, xray=x,
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
    lim = [0, max(r.max(), x.max()) + 0.3]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("radio prediction [Gyr]")
    ax.set_ylabel("X-ray prediction [Gyr]")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax, pad=0.01, label=r"weak-lensing log $M_{500}$")
    ax.set_title(f"{len(both)} LoVoCCS clusters, Spearman {rho:.2f}", fontsize=7)
    run.save_fig(fig, "real_preds")
    run.record("w8", out, {"wl": IN["wl"]} | {f"{t}_s{s}": "cnn_preds/" + PRED[t].format(s)
                                             for t in ("radio3", "xray") for s in SEEDS})
    print(f"  n {len(both)}, rho {rho:.2f}, offset {out['median_offset']:+.2f}; medians radio "
          f"{out['radio_median']:.2f}, X-ray {out['xray_median']:.2f}")


# ======================================================================= W9
def task_w9(run, a):
    import h5py
    import detect_relics_radio as DR
    plt = style()
    with h5py.File(IN["relic_cat"]) as f:
        cat = {k: f[k][:] for k in ("halo_id", "d_max", "n_relics", "merger_tsc", "mass_ratio")}
        cat_attrs = {k: f.attrs[k] for k in f.attrs}
    check(str(cat_attrs["source"]) == IN["ds512"], "relic catalogue source is not the current 512 maps")
    with h5py.File(IN["ds512"]) as f:
        imgs, hid, r500 = f["images"][:], f["meta/halo_id"][:], f["meta/r500c_kpc"][:]
        ext, mr_now = float(f.attrs["extent_r500"]), f["meta/mass_ratio"][:]
    with h5py.File(IN["tsc_legacy"]) as f:
        mt_map = {int(h): float(t) for h, t in zip(f["halo_id"][:], f["tsc_gyr"][:])}
    with h5py.File(IN["ds_legacy"]) as f:
        mr_map = {int(h): float(m) for h, m in zip(f["meta/halo_id"][:], f["meta/mass_ratio"][:])}
    mt = np.array([mt_map.get(int(h), np.nan) for h in hid])
    mr = np.array([mr_map.get(int(h), np.nan) for h in hid])
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
    agree = dict(halo_order=bool(np.array_equal(cat["halo_id"], hid)),
                 d_max=bool(np.array_equal(np.isnan(cat["d_max"]), np.isnan(d_max))
                            and np.allclose(cat["d_max"][np.isfinite(d_max)], d_max[np.isfinite(d_max)], rtol=1e-6)),
                 n_relics=bool(np.array_equal(cat["n_relics"], n_rel)),
                 merger_tsc=bool(np.allclose(cat["merger_tsc"], mt, equal_nan=True)),
                 mass_ratio=bool(np.allclose(cat["mass_ratio"], mr, equal_nan=True)),
                 mass_ratio_legacy_vs_current_maxdiff=float(np.nanmax(np.abs(mr - mr_now))))
    np.savez(run.path("relic_redetected.npz"), halo_id=hid, d_max=d_max, n_relics=n_rel, merger_tsc=mt,
             mass_ratio=mr, params=np.array(json.dumps(params)))
    hm, n_with = halo_summary(d_max / 1000.0)
    act = np.isfinite(mt) & (mt <= 2.0) & np.isfinite(hm)
    big = act & np.isfinite(mr) & (mr >= 0.1)
    out = dict(params=params, peak_definition=(
        "local maxima of the Gaussian-smoothed (smooth_kpc) arcsinh-stretched 512-px map inside the "
        "r_min_frac-r_max_frac R500 annulus, above peak_thresh_frac of the smoothed annulus maximum, "
        "maximum-filter separation 2*ceil(min_sep_kpc/pixel)+1 pixels; not a fraction of linear flux"),
        clock=("merger_tsc = TSC_Cutimages tsc_gyr: snapshot-based time since the last recorded collision of any "
               "mass ratio (legacy clock, not the T_coll endpoint); mass_ratio = the pkl mass_ratio field via "
               "dataset.h5 (semantics as stored)"),
        agreement_with_existing_catalogue=agree, n_halos=N,
        n_halos_no_peak=int((n_with == 0).sum()), n_projections_no_peak=int((n_rel == 0).sum()),
        halo_summary="mean outermost-peak distance over projections with >= 1 peak; halos with none excluded",
        active=boot_spearman(hm[act], mt[act]), mass_ratio_ge_0p1=boot_spearman(hm[big], mt[big]))
    x, yv, m2 = (d_max / 1000.0).ravel(), np.repeat(mt, 3), np.repeat(mr, 3)
    a_p = np.isfinite(x) & np.isfinite(yv) & (yv <= 2.0)
    s_p = a_p & (m2 >= 0.1)
    fig, ax = plt.subplots(figsize=(COL, COL * 0.8))
    ax.scatter(yv[a_p & ~s_p], x[a_p & ~s_p], s=3, color="0.7", label="TSC $\\leq$ 2 Gyr")
    ax.scatter(yv[s_p], x[s_p], s=5, color=C_SIM, label="mass ratio $\\geq$ 0.1")
    t = np.linspace(0, 2.0, 50)
    ax.plot(t, 1500 * 1.0227e-3 * t, color=C_OBS, ls="--", label="1500 km s$^{-1}$ shock")
    ax.set_xlabel("time since collision [Gyr]")
    ax.set_ylabel("outermost radio peak distance [Mpc]")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0, np.nanmax(x[a_p]) * 1.05)
    ax.legend()
    run.save_fig(fig, "relic")
    run.record("w9", out, {k: IN[k] for k in ("relic_cat", "ds512", "tsc_legacy", "ds_legacy")})
    print(f"  redetection agrees: {agree}; active rho {out['active']['rho']:+.3f} "
          f"[{out['active']['lo']:+.3f}, {out['active']['hi']:+.3f}] n={out['active']['n']}")


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
    for t in req:
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
    complete = run.manifest(req)
    print(f"manifest: {run.status}; complete refresh: {complete}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
