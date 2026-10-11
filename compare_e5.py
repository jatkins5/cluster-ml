#!/usr/bin/env python3
"""E5/E4 comparison driver (review rounds 15-17): loss/head arms and one
contrastive pilot on the E0 X-ray-only anchor, evaluated on single
observations.

  compare_e5.py --expect-manifest <sha256>                       (production)
  compare_e5.py --mode smoke --seeds 42,43 --expect-epochs 1 \
      --expect-cl-epochs 1 --pred-dir <dir> --expect-manifest <sha256>

MODES (review R17-A). production: exactly the unique seeds 42-46, 60
epochs, 300 pretraining epochs; prints the adoption decisions. smoke: the
declared seeds/epochs; every line is labelled DIAGNOSTIC and no adoption
decision is printed. Empty or duplicate seed lists are rejected in both.

ARM CONTRACT. Four arms, identical except the declared switches:
  h  loss huber (delta 0.5), no pretraining   (matched frozen-source control)
  m  loss mse                 d  loss class7 (seven classes)
  c  loss huber + --contrastive-pretrain (E4: D4 pairs, fitting halos only)
plus seed and output path. Required in every run: the E0 X-ray dataset and
manifest; pooled training, inner selection on single views; batch 32; no
mass/redshift/view scalars, no joint X-ray, no zeroed images; the label
curves passed and hashed; hashed inputs exactly {dataset, catalog,
label_curves}; identical defining-source hashes across ALL arms.

ARTIFACT VALIDATION (review R17-B), on every run, not only by agreement:
integer classes 0..6 with the fixed edges and consistent with the labels;
boolean no-crossing == C6; integer folds 0..4 constant over a row's
realizations; five-fold class statistics with finite representatives
inside their class supports (C0 = 0.1, C6 = 4.0), positive integer counts
and climatology = normalised counts; per-fold selected epoch and inner
score; probability arrays of exactly sample/single shape + (7,), finite,
nonnegative, summing to 1 (never renormalised) and matching the decoder;
for c, the pretraining partition equal to each fold's fitting parents
(outer and inner halos excluded), positive update counts identical across
seeds, and the saved loss curves.

SCORING: snapshot-99 single observations, ensemble over seeds for the SAME
observation; paired parent-halo bootstrap contrasts m-h, d-h, c-h (R2 and
MAE separately), per-seed s.d. separately; frozen adoption rule (round 16)
Delta R2 >= +0.02 and Delta MAE <= +0.02 Gyr, a practical preference for
further exploratory work. The class7 probabilistic block is always
emitted (review R16/R17-C): categorical log score over the seven classes
(zero true-class probabilities are counted and make the mean infinite, its
interval unavailable -- never epsilon-clipped), RPS over six cumulative
thresholds /6, no-crossing Brier, fitting-halo climatology on the same
scores, ten reliability bins with counts, and the q10-q90 class set's
coverage, enclosed mass and descriptive span.
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from compare_e0 import IDENT

N_CLASS, N_FOLD = 7, 5
CLASS_EDGES = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
CLS_LO = np.array([0.1, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])   # class supports
CLS_HI = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 4.0])
PRODUCTION_SEEDS = [42, 43, 44, 45, 46]
ARMS = {  # name: (pattern, loss, contrastive_pretrain)
    "h": ("e5_h_s{}_preds.npz", "huber", False),
    "m": ("e5_m_s{}_preds.npz", "mse", False),
    "d": ("e5_d_s{}_preds.npz", "class7", False),
    "c": ("e4_c_s{}_preds.npz", "huber", True),
}
REQUIRED_CFG = dict(dataset="xray_fix_s9991.h5", train_view="pooled",
                    select_view="single", select="inner", batch_size=32,
                    huber_delta=0.5, folds=N_FOLD, mass=False, mass_noise=False,
                    redshift_scalar=False, view_scalars=None, view_mask="",
                    no_image=False, xray_dataset=None, cl_temp=0.2, cl_batch=128)
REQUIRED_INPUTS = {"dataset", "catalog", "label_curves"}
FREE = ("seed", "out_prefix", "loss", "contrastive_pretrain")
EQUAL = ["class_of", "nocross", "class_edges", "fold_class_reps", "fold_class_counts",
         "fold_climatology"]


def parse_seeds(text, mode):
    try:
        seeds = [int(s) for s in text.split(",") if s.strip()]
    except ValueError:
        raise SystemExit(f"bad --seeds {text!r}")
    if not seeds:
        raise SystemExit("empty seed list")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate seeds in {seeds}")
    if mode == "production" and sorted(seeds) != PRODUCTION_SEEDS:
        raise SystemExit(f"production requires exactly seeds {PRODUCTION_SEEDS}, got {seeds}")
    return seeds


def load(pred_dir, seeds, arms=ARMS):
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit(f"seed list must be non-empty and unique: {seeds}")
    runs = {}
    for a, spec in arms.items():
        runs[a] = {}
        for s in seeds:
            p = os.path.join(pred_dir, spec[0].format(s))
            if not os.path.exists(p):
                raise SystemExit(f"{a}: missing seed {s}: {p}")
            runs[a][s] = dict(np.load(p, allow_pickle=True))
        if sorted(runs[a]) != sorted(seeds):
            raise SystemExit(f"{a}: seeds {sorted(runs[a])} != required {sorted(seeds)}")
    return runs


def decode(prob, reps_fold, fold_of):
    r = reps_fold[fold_of]
    return np.einsum("ik,ik->i", prob, r) if prob.ndim == 2 else \
        np.einsum("ipk,ik->ip", prob, r)


def validate_meta(tag, d, epochs):
    """Per-run artifact validation (review R17-B)."""
    tsc = np.asarray(d["tsc"], dtype=np.float64)
    n_rows = len(tsc)
    n_s = len(d["y_sample"])
    if n_rows == 0 or n_s % n_rows:
        raise SystemExit(f"{tag}: {n_s} samples not a multiple of {n_rows} label rows")
    R = n_s // n_rows
    c = d["class_of"]
    if c.shape != (n_rows,) or not np.issubdtype(c.dtype, np.integer) \
            or c.min() < 0 or c.max() >= N_CLASS:
        raise SystemExit(f"{tag}: class_of must be integer 0..6 per label row")
    if not np.array_equal(d["class_edges"], CLASS_EDGES):
        raise SystemExit(f"{tag}: class_edges {d['class_edges']} != {CLASS_EDGES}")
    nc = d["nocross"]
    if nc.shape != (n_rows,) or nc.dtype != bool or not np.array_equal(nc, c == 6):
        raise SystemExit(f"{tag}: nocross must be boolean and equal to class C6")
    ok = np.where(c == 0, np.isclose(tsc, 0.1),
                  np.where(c == 6, np.isclose(tsc, 4.0),
                           (tsc > CLS_LO[c] + 1e-9) & (tsc <= CLS_HI[c] + 1e-9)))
    if not ok.all():
        raise SystemExit(f"{tag}: {int((~ok).sum())} class labels inconsistent with pseudo-TSC")
    fo = d["fold_of"]
    if fo.shape != (n_s,) or not np.issubdtype(fo.dtype, np.integer) \
            or fo.min() < 0 or fo.max() >= N_FOLD:
        raise SystemExit(f"{tag}: fold_of must be integer 0..{N_FOLD - 1} per sample")
    if not (fo.reshape(n_rows, R) == fo.reshape(n_rows, R)[:, :1]).all():
        raise SystemExit(f"{tag}: fold differs between realizations of one row")
    reps, cnt, clim = d["fold_class_reps"], d["fold_class_counts"], d["fold_climatology"]
    for nm, a_ in (("fold_class_reps", reps), ("fold_class_counts", cnt),
                   ("fold_climatology", clim)):
        if a_.shape != (N_FOLD, N_CLASS):
            raise SystemExit(f"{tag}: {nm} shape {a_.shape} != {(N_FOLD, N_CLASS)}")
    if not np.isfinite(reps).all() or not (reps[:, 0] == 0.1).all() or not (reps[:, 6] == 4.0).all():
        raise SystemExit(f"{tag}: representatives must be finite with C0 = 0.1, C6 = 4.0")
    inside = (reps[:, 1:6] > CLS_LO[1:6]) & (reps[:, 1:6] <= CLS_HI[1:6] + 1e-12)
    if not inside.all():
        raise SystemExit(f"{tag}: representatives outside their class supports")
    if not np.issubdtype(cnt.dtype, np.integer) or (cnt <= 0).any():
        raise SystemExit(f"{tag}: class counts must be positive integers")
    if not np.allclose(clim, cnt / cnt.sum(1, keepdims=True), atol=1e-12):
        raise SystemExit(f"{tag}: climatology != normalised counts")
    be, ss = d["fold_best_epoch"], d["fold_sel_score"]
    if be.shape != (N_FOLD,) or not np.issubdtype(be.dtype, np.integer) \
            or be.min() < 1 or be.max() > epochs or ss.shape != (N_FOLD,) \
            or not np.isfinite(ss).all():
        raise SystemExit(f"{tag}: per-fold selected epoch/inner score missing or invalid")


def check_probs(tag, d):
    for k, ref in (("oof_prob", "oof_sample"), ("oof_single_prob", "oof_single")):
        p = d[k]
        if p.shape != d[ref].shape + (N_CLASS,):
            raise SystemExit(f"{tag}: {k} shape {p.shape} != {d[ref].shape + (N_CLASS,)}")
        p = p.astype(np.float64)
        if not np.isfinite(p).all() or (p < 0).any():
            raise SystemExit(f"{tag}: {k} must be finite and nonnegative")
        if not np.allclose(p.sum(-1), 1.0, atol=1e-6):
            raise SystemExit(f"{tag}: {k} does not sum to 1 (not renormalised here)")
        if not np.allclose(decode(p, d["fold_class_reps"], d["fold_of"].astype(int)),
                           d[ref], atol=1e-4):
            raise SystemExit(f"{tag}: saved {ref} disagrees with the decoder")


def fitting_parents(d, f):
    sg, fo = d["sample_group"], d["fold_of"]
    train = set(np.unique(sg[fo != f]).tolist())
    inner = set(d["sel_groups_flat"][d["sel_groups_fold"] == f].astype(int).tolist())
    return sorted(train - inner)


def check_cl(tag, d, cl_epochs):
    if not bool(d["contrastive_pretrain"]):
        raise SystemExit(f"{tag}: contrastive_pretrain must be True")
    cfg = json.loads(str(d["cl_config"]))
    if cfg.get("epochs") != cl_epochs:
        raise SystemExit(f"{tag}: pretraining epochs {cfg.get('epochs')} != {cl_epochs}")
    up, pr = d["fold_cl_updates"], d["fold_cl_pairs"]
    if up.shape != (N_FOLD,) or (up <= 0).any() or pr.shape != (N_FOLD,) or (pr <= 0).any():
        raise SystemExit(f"{tag}: pretraining update/pair counts missing or nonpositive")
    cur = d["cl_loss_curve_fold"]
    if cur.shape != (N_FOLD, cl_epochs) or not np.isfinite(cur).all():
        raise SystemExit(f"{tag}: pretraining loss curves {cur.shape} invalid")
    gf, gi = d["cl_groups_flat"], d["cl_groups_fold"]
    for f in range(N_FOLD):
        got = sorted(gf[gi == f].astype(int).tolist())
        if got != fitting_parents(d, f):
            raise SystemExit(f"{tag}: fold {f} pretraining parents != fitting parents "
                             f"(inner/outer halos must be excluded)")


def check(runs, expect_manifest, arms=ARMS, epochs=60, cl_epochs=300):
    a0 = next(iter(arms))
    s0 = next(iter(runs[a0]))
    ref = runs[a0][s0]
    rprov = json.loads(str(ref["provenance"]))
    required = dict(REQUIRED_CFG, epochs=epochs, cl_epochs=cl_epochs)
    cl_ref = None
    for a, (_, loss, cl) in arms.items():
        for s, d in runs[a].items():
            tag = f"{a} seed {s}"
            for k in IDENT + EQUAL + ["oof_single", "oof_sample", "oof_prob",
                                      "oof_single_prob", "provenance", "loss",
                                      "provenance_changed_during_run", "fold_best_epoch",
                                      "fold_sel_score", "contrastive_pretrain"]:
                if k not in d:
                    raise SystemExit(f"{tag}: missing {k}")
            for k in IDENT + ["oof_single", "oof_sample"]:
                if d[k].shape != ref[k].shape:
                    raise SystemExit(f"{tag}: {k} shape {d[k].shape} != {ref[k].shape}")
            validate_meta(tag, d, epochs)
            for k in IDENT + EQUAL:
                if d[k].shape != ref[k].shape or not np.array_equal(d[k], ref[k]):
                    raise SystemExit(f"{tag}: {k} differs from {a0} seed {s0}")
            for k in ("oof_single", "oof_sample"):
                if not np.isfinite(d[k]).all():
                    raise SystemExit(f"{tag}: non-finite {k}")
            prov = json.loads(str(d["provenance"]))
            cfg = prov["config"]
            if prov["manifest_sha256"] != expect_manifest:
                raise SystemExit(f"{tag}: manifest {prov['manifest_sha256']} != {expect_manifest}")
            if bool(d["provenance_changed_during_run"]):
                raise SystemExit(f"{tag}: sources/manifest/inputs changed during the run")
            if prov["sha256"] != rprov["sha256"]:
                raise SystemExit(f"{tag}: defining-source hashes differ between runs")
            if set(prov.get("inputs_sha256") or {}) != REQUIRED_INPUTS:
                raise SystemExit(f"{tag}: hashed inputs {sorted(prov.get('inputs_sha256') or {})} "
                                 f"!= required {sorted(REQUIRED_INPUTS)}")
            if prov["inputs_sha256"] != rprov["inputs_sha256"]:
                raise SystemExit(f"{tag}: input-file hashes differ between runs")
            if int(cfg["seed"]) != s:
                raise SystemExit(f"{tag}: file holds a run with seed {cfg['seed']}")
            for k, v in required.items():
                if cfg.get(k) != v:
                    raise SystemExit(f"{tag}: protocol requires {k}={v!r}, run has {cfg.get(k)!r}")
            if cfg.get("loss") != loss or str(d["loss"]) != loss:
                raise SystemExit(f"{tag}: loss {cfg.get('loss')!r}/{d['loss']} != declared {loss!r}")
            if bool(cfg.get("contrastive_pretrain")) != cl or bool(d["contrastive_pretrain"]) != cl:
                raise SystemExit(f"{tag}: contrastive_pretrain must be {cl}")
            strip = lambda c_: {k: v for k, v in c_.items() if k not in FREE}
            if strip(cfg) != strip(rprov["config"]):
                diff = sorted(k for k in set(strip(cfg)) | set(strip(rprov["config"]))
                              if strip(cfg).get(k) != strip(rprov["config"]).get(k))
                raise SystemExit(f"{tag}: undeclared configuration differences {diff}")
            if loss == "class7":
                check_probs(tag, d)
            elif d["oof_prob"].size or d["oof_single_prob"].size:
                raise SystemExit(f"{tag}: scalar arm carries probabilities")
            if cl:
                check_cl(tag, d, cl_epochs)
                if cl_ref is None:
                    cl_ref = d
                elif not (np.array_equal(d["fold_cl_updates"], cl_ref["fold_cl_updates"])
                          and np.array_equal(d["cl_groups_flat"], cl_ref["cl_groups_flat"])):
                    raise SystemExit(f"{tag}: pretraining partition/update counts differ across seeds")
            elif d.get("cl_groups_flat", np.zeros(0)).size:
                raise SystemExit(f"{tag}: non-contrastive arm carries pretraining records")
    return ref


def single_obs(ref, arm_runs, key="oof_single"):
    m = ref["sample_snap"] == 99
    P = ref["oof_single"].shape[1]
    y = np.repeat(ref["y_sample"][m][:, None], P, 1).ravel()
    g = np.repeat(ref["sample_group"][m][:, None], P, 1).ravel()
    per_seed = [d[key][m].reshape(-1, *d[key].shape[2:]) for d in arm_runs.values()]
    return y, g, per_seed


def halo_boot(g, stat, n, seed=0):
    """Percentile interval of stat(sel) over parent-halo resamples."""
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {h: np.where(g == h)[0] for h in ug}
    out = [stat(np.concatenate([idx[h] for h in rng.choice(ug, len(ug))])) for _ in range(n)]
    return np.percentile(out, [2.5, 97.5])


def prob_scores(prob, cls):
    """Per-observation categorical log score (inf where the true class has
    zero probability -- never clipped), RPS over six thresholds /6, and the
    no-crossing Brier score."""
    p_true = prob[np.arange(len(cls)), cls]
    with np.errstate(divide="ignore"):
        logs = -np.log(p_true)
    F = np.cumsum(prob, axis=1)[:, :N_CLASS - 1]
    O = (cls[:, None] <= np.arange(N_CLASS - 1)[None, :]).astype(float)
    rps = ((F - O) ** 2).sum(1) / (N_CLASS - 1)
    brier = (prob[:, 6] - (cls == 6)) ** 2
    return logs, rps, brier


def pred_set(prob):
    F = np.cumsum(prob, axis=1)
    lo = np.argmax(F >= 0.1 - 1e-12, axis=1)
    hi = np.argmax(F >= 0.9 - 1e-12, axis=1)
    below = np.where(lo > 0, F[np.arange(len(lo)), np.maximum(lo - 1, 0)], 0.0)
    return lo, hi, F[np.arange(len(hi)), hi] - below


def prob_block(ref, runs, g, n_boot, label):
    m = ref["sample_snap"] == 99
    P = ref["oof_single"].shape[1]
    R = len(ref["y_sample"]) // len(ref["class_of"])
    cls = np.repeat(np.repeat(ref["class_of"], R)[m][:, None], P, 1).ravel()
    fo = np.repeat(ref["fold_of"][m][:, None], P, 1).ravel().astype(int)
    pd_ = np.mean([d["oof_single_prob"][m].reshape(-1, N_CLASS) for d in runs["d"].values()], 0)
    ens = np.mean([d["oof_single"][m].ravel() for d in runs["d"].values()], 0)
    if not np.allclose(np.einsum("ik,ik->i", pd_, ref["fold_class_reps"][fo]), ens, atol=1e-4):
        raise SystemExit("ensemble decode of averaged probabilities != averaged decoded points")
    clim = ref["fold_climatology"][fo]
    print(f"\n{label}class7 probabilistic scores (snapshot-99 single observations, "
          f"{len(runs['d'])}-seed probability mixture; classes are proxy-label bins, not "
          f"physical collision dates)")
    print(f"  true-class counts C0..C6: {np.bincount(cls, minlength=N_CLASS).tolist()}")
    sm, sc = prob_scores(pd_, cls), prob_scores(clim, cls)
    for lab, i in (("log score", 0), ("RPS (/6)", 1), ("Brier P(no crossing)", 2)):
        a_, b_ = sm[i], sc[i]
        nz_m, nz_c = int(np.isinf(a_).sum()), int(np.isinf(b_).sum())
        if nz_m or nz_c:
            print(f"  {lab:<22} model {a_.mean():.4f}  climatology {b_.mean():.4f}  "
                  f"(zero true-class probability: model {nz_m}, climatology {nz_c} "
                  f"observations; mean infinite, interval unavailable)")
            continue
        iv = halo_boot(g, lambda s_: a_[s_].mean() - b_[s_].mean(), n_boot)
        print(f"  {lab:<22} model {a_.mean():.4f}  climatology {b_.mean():.4f}  "
              f"model - clim {a_.mean() - b_.mean():+.4f} [{iv[0]:+.4f}, {iv[1]:+.4f}] "
              f"(lower is better)")
    edges = np.linspace(0, 1, 11)
    bi = np.clip(np.digitize(pd_[:, 6], edges[1:-1]), 0, 9)
    print("  reliability P(no crossing), 10 fixed bins: [bin] n  mean forecast  observed freq")
    for j in range(10):
        s_ = bi == j
        print(f"    [{edges[j]:.1f},{edges[j + 1]:.1f}) {s_.sum():5d}  "
              + (f"{pd_[s_, 6].mean():.3f}  {(cls[s_] == 6).mean():.3f}" if s_.any() else "(empty)"))
    for lab, pr in (("model", pd_), ("climatology", clim)):
        lo, hi, mass = pred_set(pr)
        print(f"  q10-q90 class set ({lab}): empirical coverage of the true class "
              f"{((cls >= lo) & (cls <= hi)).mean():.3f}; mean enclosed forecast mass "
              f"{mass.mean():.3f}; mean span {(CLS_HI[hi] - CLS_LO[lo]).mean():.2f} Gyr "
              f"(descriptive, capped proxy scale)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-manifest", required=True)
    ap.add_argument("--mode", choices=["production", "smoke"], default="production")
    ap.add_argument("--pred-dir", default="cnn_preds")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--expect-epochs", type=int, default=None)
    ap.add_argument("--expect-cl-epochs", type=int, default=None)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds, a.mode)
    if a.mode == "production":
        if a.expect_epochs not in (None, 60) or a.expect_cl_epochs not in (None, 300):
            raise SystemExit("production requires 60 epochs and 300 pretraining epochs")
        epochs, cl_epochs = 60, 300
    else:
        if a.expect_epochs is None or a.expect_cl_epochs is None:
            raise SystemExit("smoke mode needs --expect-epochs and --expect-cl-epochs")
        epochs, cl_epochs = a.expect_epochs, a.expect_cl_epochs
    lab = "DIAGNOSTIC (smoke, not a result): " if a.mode == "smoke" else ""
    runs = load(a.pred_dir, seeds)
    ref = check(runs, a.expect_manifest, epochs=epochs, cl_epochs=cl_epochs)
    print(f"{lab}checks passed: {len(ARMS)} arms x seeds {seeds}; per-run artifacts valid; "
          f"identities, folds, inner sets, source/input hashes, classes and per-fold "
          f"representatives identical; declared switches only; manifest {a.expect_manifest[:12]}...")
    y, g, _ = single_obs(ref, runs["h"])
    ps = {k: single_obs(ref, runs[k])[2] for k in ARMS}
    ens = {k: np.mean(v, 0) for k, v in ps.items()}
    print(f"\n{lab}snapshot-99 single observations (n={len(y)}; {len(np.unique(g))} parent "
          f"halos), {len(seeds)}-seed ensembles")
    for k in ARMS:
        print(f"  {k} ({ARMS[k][1]}{' + contrastive' if ARMS[k][2] else ''}) "
              f"R2 {r2_score(y, ens[k]):.3f}  MAE {mean_absolute_error(y, ens[k]):.3f} Gyr")
    for b in ("m", "d", "c"):
        dr = r2_score(y, ens[b]) - r2_score(y, ens["h"])
        dm = mean_absolute_error(y, ens[b]) - mean_absolute_error(y, ens["h"])
        ir = halo_boot(g, lambda s_: r2_score(y[s_], ens[b][s_]) - r2_score(y[s_], ens["h"][s_]), a.boot)
        im = halo_boot(g, lambda s_: mean_absolute_error(y[s_], ens[b][s_])
                       - mean_absolute_error(y[s_], ens["h"][s_]), a.boot)
        ds = [r2_score(y, pb) - r2_score(y, pa) for pa, pb in zip(ps["h"], ps[b])]
        sd = f" ± {np.std(ds, ddof=1):.3f} (s.d.)" if len(ds) > 1 else ""
        print(f"\n{lab}{b} - h: R2 {dr:+.3f} [{ir[0]:+.3f}, {ir[1]:+.3f}]; MAE {dm:+.3f} "
              f"[{im[0]:+.3f}, {im[1]:+.3f}] Gyr (paired halo bootstrap, {a.boot} draws; "
              f"conditional on the fitted runs)")
        print(f"  per-seed single-model R2 difference {np.mean(ds):+.3f}{sd}")
        if a.mode == "production":
            ok = dr >= 0.02 and dm <= 0.02
            print(f"  adoption rule (frozen, round 16: dR2 >= +0.02 and dMAE <= +0.02 Gyr): "
                  f"{'meets' if ok else 'does not meet'} the adoption criterion")
    prob_block(ref, runs, g, a.boot, lab)


if __name__ == "__main__":
    main()
