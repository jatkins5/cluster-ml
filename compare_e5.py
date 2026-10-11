#!/usr/bin/env python3
"""E5 comparison driver (review rounds 15-16): loss/head arms on the E0
X-ray-only anchor, evaluated on single observations.

  compare_e5.py --expect-manifest <sha256>

ARM CONTRACT. Three arms, identical except the declared `loss`:
  h  huber (delta 0.5; legacy default)   m  mse   d  class7 (seven classes)
plus seed and output path. Required common protocol in every run: the
E0 X-ray dataset and manifest; pooled training, inner selection on single
views; 60 epochs, batch 32; no mass/redshift/view scalars, no joint X-ray,
no zeroed images; the label curves passed and hashed (identity metadata,
not a predictor input); hashed inputs exactly {dataset, catalog,
label_curves}. Cross-run equality of identities, labels, folds, inner
sets, source and input hashes, classes, and per-fold class representatives,
counts and climatology (identical across seeds under the common manifest,
so averaging probabilities then decoding equals averaging decoded points).
Class7 runs: finite, nonnegative probabilities summing to 1 (never
renormalised here), and saved points equal to the decoder.

Primary estimator: snapshot-99 single observations, ensemble over seeds
for the SAME observation. Paired parent-halo bootstrap contrasts m - h and
d - h (R2 and MAE separately), per-seed scatter separately, and the frozen
adoption rule (round 16): Delta R2 >= +0.02 and Delta MAE <= +0.02 Gyr on
the point estimates -- a practical preference for further exploratory work,
not a detection threshold. Arms missing it "do not meet the adoption
criterion". The class7 probabilistic block is ALWAYS emitted: categorical
log score (over these seven classes; not a density likelihood), RPS over
the six cumulative thresholds divided by six, no-crossing Brier score, a
fitting-halo climatology baseline on the same scores with paired halo
intervals, ten fixed reliability bins for P(no crossing) with counts, and
the q10-q90 contiguous class set's empirical coverage of the true class,
its enclosed forecast mass, and its descriptive span on the capped proxy
scale.
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from compare_e0 import IDENT

ARMS = {"h": ("e5_h_s{}_preds.npz", "huber"),
        "m": ("e5_m_s{}_preds.npz", "mse"),
        "d": ("e5_d_s{}_preds.npz", "class7")}
REQUIRED_CFG = dict(dataset="xray_fix_s9991.h5", train_view="pooled",
                    select_view="single", select="inner", epochs=60, batch_size=32,
                    huber_delta=0.5, folds=5, mass=False, mass_noise=False,
                    redshift_scalar=False, view_scalars=None, view_mask="",
                    no_image=False, xray_dataset=None)
REQUIRED_INPUTS = {"dataset", "catalog", "label_curves"}
FREE = ("seed", "out_prefix", "loss")
EQUAL = ["class_of", "nocross", "class_edges", "fold_class_reps", "fold_class_counts",
         "fold_climatology"]
N_CLASS = 7
# descriptive numeric span of each class on the capped proxy scale
CLS_LO = np.array([0.1, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
CLS_HI = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 4.0])


def load(pred_dir, seeds, arms=ARMS):
    runs = {}
    for a, (pat, _) in arms.items():
        runs[a] = {}
        for s in seeds:
            p = os.path.join(pred_dir, pat.format(s))
            if not os.path.exists(p):
                raise SystemExit(f"{a}: missing seed {s}: {p}")
            runs[a][s] = dict(np.load(p, allow_pickle=True))
    return runs


def decode(prob, reps_fold, fold_of):
    return np.einsum("...k,...k->...", prob, reps_fold[fold_of][:, None, :]
                     if prob.ndim == 3 else reps_fold[fold_of])


def check(runs, expect_manifest, arms=ARMS, expect_epochs=60):
    required = dict(REQUIRED_CFG, epochs=expect_epochs)
    a0 = next(iter(arms))
    s0 = next(iter(runs[a0]))
    ref = runs[a0][s0]
    rprov = json.loads(str(ref["provenance"]))
    for a, (_, loss) in arms.items():
        for s, d in runs[a].items():
            tag = f"{a} seed {s}"
            for k in IDENT + EQUAL + ["oof_single", "oof_sample", "oof_prob",
                                      "oof_single_prob", "provenance",
                                      "provenance_changed_during_run", "loss"]:
                if k not in d:
                    raise SystemExit(f"{tag}: missing {k}")
            for k in IDENT + ["oof_single", "oof_sample"]:
                if d[k].shape != ref[k].shape:
                    raise SystemExit(f"{tag}: {k} shape {d[k].shape} != {ref[k].shape}")
            for k in IDENT + EQUAL:
                if d[k].shape != ref[k].shape or not np.array_equal(d[k], ref[k]):
                    raise SystemExit(f"{tag}: {k} differs from {a0} seed {s0}")
            for k in ("oof_single", "oof_sample"):
                if not np.isfinite(d[k]).all():
                    raise SystemExit(f"{tag}: non-finite {k}")
            if not np.all(d["fold_of"] >= 0):
                raise SystemExit(f"{tag}: incomplete out-of-fold coverage")
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
            strip = lambda c: {k: v for k, v in c.items() if k not in FREE}
            if strip(cfg) != strip(rprov["config"]):
                diff = sorted(k for k in set(strip(cfg)) | set(strip(rprov["config"]))
                              if strip(cfg).get(k) != strip(rprov["config"]).get(k))
                raise SystemExit(f"{tag}: undeclared configuration differences {diff}")
            if loss == "class7":
                check_probs(tag, d)
            elif d["oof_prob"].size or d["oof_single_prob"].size:
                raise SystemExit(f"{tag}: scalar arm carries probabilities")
    return ref


def check_probs(tag, d):
    for k in ("oof_prob", "oof_single_prob"):
        p = d[k].astype(np.float64)
        if p.shape[-1] != N_CLASS or not np.isfinite(p).all() or (p < 0).any():
            raise SystemExit(f"{tag}: {k} must be finite, nonnegative, {N_CLASS} classes")
        if not np.allclose(p.sum(-1), 1.0, atol=1e-5):
            raise SystemExit(f"{tag}: {k} does not sum to 1 (not renormalised here)")
    reps = d["fold_class_reps"]
    fo = d["fold_of"].astype(int)
    if not np.allclose(decode(d["oof_prob"].astype(np.float64), reps, fo),
                       d["oof_sample"], atol=1e-4):
        raise SystemExit(f"{tag}: saved pooled points disagree with the decoder")
    if not np.allclose(decode(d["oof_single_prob"].astype(np.float64), reps, fo),
                       d["oof_single"], atol=1e-4):
        raise SystemExit(f"{tag}: saved single-view points disagree with the decoder")


def single_obs(ref, arm_runs, key="oof_single"):
    m = ref["sample_snap"] == 99
    P = ref["oof_single"].shape[1]
    y = np.repeat(ref["y_sample"][m][:, None], P, 1).ravel()
    g = np.repeat(ref["sample_group"][m][:, None], P, 1).ravel()
    per_seed = [d[key][m].reshape(-1, *d[key].shape[2:]) for d in arm_runs.values()]
    return y, g, per_seed


def boot(y, g, fa, fb, metric, n, seed=0):
    """Paired parent-halo bootstrap of metric(b) - metric(a)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {h: np.where(g == h)[0] for h in ug}
    out = []
    for _ in range(n):
        sel = np.concatenate([idx[h] for h in rng.choice(ug, len(ug))])
        out.append(metric(y[sel], fb[sel]) - metric(y[sel], fa[sel]))
    return np.percentile(out, [2.5, 97.5])


def prob_scores(prob, cls):
    """Per-observation categorical log score, RPS (/6) and no-crossing Brier."""
    p = np.clip(prob, 0.0, 1.0)
    logs = -np.log(p[np.arange(len(cls)), cls])
    F = np.cumsum(p, axis=1)[:, :N_CLASS - 1]
    O = (cls[:, None] <= np.arange(N_CLASS - 1)[None, :]).astype(float)
    rps = ((F - O) ** 2).sum(1) / (N_CLASS - 1)
    brier = (p[:, 6] - (cls == 6)) ** 2
    return logs, rps, brier


def pred_set(prob):
    F = np.cumsum(prob, axis=1)
    lo = np.argmax(F >= 0.1 - 1e-12, axis=1)
    hi = np.argmax(F >= 0.9 - 1e-12, axis=1)
    below = np.where(lo > 0, F[np.arange(len(lo)), np.maximum(lo - 1, 0)], 0.0)
    mass = F[np.arange(len(hi)), hi] - below
    return lo, hi, mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-manifest", required=True)
    ap.add_argument("--pred-dir", default="cnn_preds")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--expect-epochs", type=int, default=60,
                    help="required training epochs (60 in production; smoke fixtures use 1)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = load(a.pred_dir, seeds)
    ref = check(runs, a.expect_manifest, expect_epochs=a.expect_epochs)
    print(f"checks passed: {len(ARMS)} arms x {len(seeds)} seeds; identities, folds, inner "
          f"sets, source/input hashes, classes and per-fold representatives identical; "
          f"declared loss only; manifest {a.expect_manifest[:12]}...")
    y, g, _ = single_obs(ref, runs["h"])
    ps = {k: single_obs(ref, runs[k])[2] for k in ARMS}
    ens = {k: np.mean(v, 0) for k, v in ps.items()}
    print(f"\nsnapshot-99 single observations (n={len(y)}; {len(np.unique(g))} parent halos), "
          f"{len(seeds)}-seed ensembles")
    for k in ARMS:
        print(f"  {k} ({ARMS[k][1]:<6}) R2 {r2_score(y, ens[k]):.3f}  "
              f"MAE {mean_absolute_error(y, ens[k]):.3f} Gyr")
    for b in ("m", "d"):
        dr = r2_score(y, ens[b]) - r2_score(y, ens["h"])
        dm = mean_absolute_error(y, ens[b]) - mean_absolute_error(y, ens["h"])
        ir = boot(y, g, ens["h"], ens[b], r2_score, a.boot)
        im = boot(y, g, ens["h"], ens[b], mean_absolute_error, a.boot)
        ds = [r2_score(y, pb) - r2_score(y, pa) for pa, pb in zip(ps["h"], ps[b])]
        sd = f" ± {np.std(ds, ddof=1):.3f} (s.d.)" if len(ds) > 1 else ""
        ok = dr >= 0.02 and dm <= 0.02
        print(f"\n{b} - h: R2 {dr:+.3f} [{ir[0]:+.3f}, {ir[1]:+.3f}]; MAE {dm:+.3f} "
              f"[{im[0]:+.3f}, {im[1]:+.3f}] Gyr (paired halo bootstrap, {a.boot} draws; "
              f"conditional on the fitted runs)")
        print(f"  per-seed single-model R2 difference {np.mean(ds):+.3f}{sd}")
        print(f"  adoption rule (frozen, round 16: dR2 >= +0.02 and dMAE <= +0.02 Gyr): "
              f"{'meets' if ok else 'does not meet'} the adoption criterion")

    # ---- class7 probabilistic block (always emitted) ----
    m = ref["sample_snap"] == 99
    P = ref["oof_single"].shape[1]
    cls = np.repeat(np.repeat(ref["class_of"], len(ref["y_sample"]) // len(ref["class_of"]))[m][:, None],
                    P, 1).ravel()
    fo = np.repeat(ref["fold_of"][m][:, None], P, 1).ravel().astype(int)
    pd_ = np.mean([d["oof_single_prob"][m].reshape(-1, N_CLASS) for d in runs["d"].values()], 0)
    # same reps across seeds => decode(mean p) == mean decode(p)
    dec = np.einsum("ik,ik->i", pd_, ref["fold_class_reps"][fo])
    if not np.allclose(dec, ens["d"], atol=1e-4):
        raise SystemExit("ensemble decode of averaged probabilities != averaged decoded points")
    clim = ref["fold_climatology"][fo]
    print(f"\nclass7 probabilistic scores (snapshot-99 single observations, {len(seeds)}-seed "
          f"probability mixture; classes are proxy-label bins, not physical collision dates)")
    print(f"  true-class counts C0..C6: {np.bincount(cls, minlength=N_CLASS).tolist()}")
    sd_, sc_ = prob_scores(pd_, cls), prob_scores(clim, cls)
    for lab, i in (("log score", 0), ("RPS (/6)", 1), ("Brier P(no crossing)", 2)):
        dv = sd_[i].mean() - sc_[i].mean()
        rng = np.random.default_rng(0)
        ug = np.unique(g)
        idx = {h: np.where(g == h)[0] for h in ug}
        bs = []
        for _ in range(a.boot):
            sel = np.concatenate([idx[h] for h in rng.choice(ug, len(ug))])
            bs.append(sd_[i][sel].mean() - sc_[i][sel].mean())
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        print(f"  {lab:<22} model {sd_[i].mean():.4f}  climatology {sc_[i].mean():.4f}  "
              f"model - clim {dv:+.4f} [{lo_:+.4f}, {hi_:+.4f}] (lower is better)")
    edges = np.linspace(0, 1, 11)
    b_ = np.clip(np.digitize(pd_[:, 6], edges[1:-1]), 0, 9)
    print("  reliability P(no crossing), 10 fixed bins: [bin] n  mean forecast  observed freq")
    for j in range(10):
        s_ = b_ == j
        if s_.any():
            print(f"    [{edges[j]:.1f},{edges[j + 1]:.1f}) {s_.sum():5d}  "
                  f"{pd_[s_, 6].mean():.3f}  {(cls[s_] == 6).mean():.3f}")
        else:
            print(f"    [{edges[j]:.1f},{edges[j + 1]:.1f})     0  (empty)")
    for lab, pr in (("model", pd_), ("climatology", clim)):
        lo, hi, mass = pred_set(pr)
        cov = ((cls >= lo) & (cls <= hi)).mean()
        span = CLS_HI[hi] - CLS_LO[lo]
        print(f"  q10-q90 class set ({lab}): empirical coverage of the true class {cov:.3f}; "
              f"mean enclosed forecast mass {mass.mean():.3f}; mean span {span.mean():.2f} Gyr "
              f"(descriptive, capped proxy scale)")


if __name__ == "__main__":
    main()
