#!/usr/bin/env python3
"""Training-label comparison (2026-11-05, review rounds 22-23): pseudo-TSC vs
Chuiyang's catalogue TSC from T_coll (TSC_new: tsc_gyr_from_Tcoll), as the
CAPPED time since the last qualifying catalogue collision: finite times above
4 Gyr and NaN (no qualifying collision recorded in the catalogue's covered
history; check_tsc_missing.py) both map to 4 Gyr. tsc_gyr (snapshot-based)
is saved as a truth only.

  compare_labels.py --expect-manifest <sha256> [--mode smoke --seeds 42,43 --expect-epochs 1]

Two modalities x two labels, seeds 42-46 (20 runs):
  x_* : X-ray only (xray_fix_s9991.h5; the E5-H anchor protocol)
  r_* : radio, physical 3 Mpc, regenerated radio model (injected_syn_phys_b3000_s9991.h5)
Within a modality the ONLY declared difference is --label (plus seed and
output). Each arm selects its checkpoint on ITS OWN training target (inner
single-observation R2), so this compares complete target-specific
training/selection pipelines; the common outer truth never influences
selection. R2 against a model's own target is not comparable across labels,
so the PRIMARY score is how well each arm's snapshot-99 single-observation
5-seed ensemble recovers the SAME truth (capped T_coll TSC): R2, Spearman
and MAE, with paired parent-halo bootstrap contrasts tcoll - pseudo for all
three, seed s.d. separately, and fold-fitting-label mean/median baselines on
the same cohort. Secondaries: each arm against the pseudo-TSC truth, and
Spearman on the finite-below-cap rows (observation and parent-halo counts).
Prespecified practical tolerance for further exploratory use (not
noninferiority, not validation of the event definition, and not by itself a
manuscript decision): the tcoll arm's R2 against the T_coll truth within
-0.02 of the pseudo arm's, in BOTH modalities (point estimates; intervals
and MAE reported alongside).
"""
import argparse
import json
import os

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

from compare_e0 import IDENT

PRODUCTION_SEEDS = [42, 43, 44, 45, 46]
LABELS = ("pseudo", "tsc_tcoll")      # review R22: tsc_gyr kept as a saved truth only
R_EXPECT, P_EXPECT, N_FOLD, CAP = 3, 3, 5, 4.0
SELECT_CRITERION = "inner:single:r2"   # each arm selects on ITS OWN training target
MODALITIES = {"x": "xray_fix_s9991.h5", "r": "injected_syn_phys_b3000_s9991.h5"}
TRUTHS = ("truth_pseudo", "truth_tsc_capped", "truth_tsc_raw", "truth_tcoll_capped",
          "truth_tcoll_raw")
TARGET_OF = {"pseudo": "truth_pseudo", "tsc": "truth_tsc_capped", "tsc_tcoll": "truth_tcoll_capped"}
REQUIRED_CFG = dict(fold_manifest="folds_e0.npz", train_view="pooled", select_view="single",
                    select="inner", batch_size=32, huber_delta=0.5, folds=5, loss="huber",
                    mass=False, mass_noise=False, redshift_scalar=False, view_scalars=None,
                    no_image=False, xray_dataset=None, label_curves=None,
                    contrastive_pretrain=False)
REQUIRED_INPUTS = {"dataset", "catalog", "tsc_snap99", "tsc_snap91"}
FREE = ("seed", "out_prefix", "label")


def pattern(m, lab):
    return f"lbl_{m}_{lab}_s{{}}_preds.npz"


def parse_seeds(text, mode):
    seeds = [int(s) for s in text.split(",") if s.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit(f"seed list must be non-empty and unique: {text!r}")
    if mode == "production" and sorted(seeds) != PRODUCTION_SEEDS:
        raise SystemExit(f"production requires exactly seeds {PRODUCTION_SEEDS}")
    return seeds


def load(pred_dir, m, seeds):
    runs = {}
    for lab in LABELS:
        runs[lab] = {}
        for s in seeds:
            p = os.path.join(pred_dir, pattern(m, lab).format(s))
            if not os.path.exists(p):
                raise SystemExit(f"{m}/{lab}: missing seed {s}: {p}")
            runs[lab][s] = dict(np.load(p, allow_pickle=True))
    return runs


def validate_artifact(tag, d, epochs):
    """Per-file validation before any cross-run comparison (review R22-C)."""
    tsc = d["tsc"]
    n = len(tsc)
    if tsc.ndim != 1 or n == 0:
        raise SystemExit(f"{tag}: per-row target must be 1-D and non-empty")
    if len(d["y_sample"]) != n * R_EXPECT:
        raise SystemExit(f"{tag}: {len(d['y_sample'])} samples != {n} rows x R={R_EXPECT}")
    if d["oof_sample"].shape != (n * R_EXPECT,) or d["oof_single"].shape != (n * R_EXPECT, P_EXPECT):
        raise SystemExit(f"{tag}: prediction shapes {d['oof_sample'].shape}/{d['oof_single'].shape} "
                         f"!= ({n * R_EXPECT},)/({n * R_EXPECT}, {P_EXPECT})")
    for k in ("group_id", "snap"):
        if k not in d or d[k].shape != (n,):
            raise SystemExit(f"{tag}: per-row {k} missing or mis-shaped")
    if not (np.array_equal(d["sample_group"], np.repeat(d["group_id"], R_EXPECT))
            and np.array_equal(d["sample_snap"], np.repeat(d["snap"], R_EXPECT))
            and np.array_equal(d["sample_realization"], np.tile(np.arange(R_EXPECT), n))):
        raise SystemExit(f"{tag}: row-to-sample parent/snapshot/realization mapping inconsistent")
    ids = list(zip(d["sample_halo"].tolist(), d["sample_realization"].tolist()))
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{tag}: duplicate sample identities")
    if len(set(zip(d["group_id"].tolist(), d["snap"].tolist()))) != n:
        raise SystemExit(f"{tag}: duplicate (parent, snapshot) label rows")
    if float(d["tsc_cap"]) != CAP:
        raise SystemExit(f"{tag}: saved cap {d['tsc_cap']} != {CAP}")
    for k in TRUTHS:
        if d[k].shape != (n,):
            raise SystemExit(f"{tag}: {k} shape {d[k].shape} != ({n},)")
    for k in ("truth_tsc_raw", "truth_tcoll_raw"):
        v = d[k]
        if np.isinf(v).any() or (v[np.isfinite(v)] < 0).any():
            raise SystemExit(f"{tag}: {k} has infinite or negative values")
    for k in ("truth_pseudo", "truth_tsc_capped", "truth_tcoll_capped"):
        v = d[k]
        if not np.isfinite(v).all() or v.min() < 0 or v.max() > CAP + 1e-9:
            raise SystemExit(f"{tag}: {k} must be finite in [0, {CAP}]")
    for raw, capd in (("truth_tsc_raw", "truth_tsc_capped"), ("truth_tcoll_raw", "truth_tcoll_capped")):
        exp = np.where(np.isnan(d[raw]), CAP, np.minimum(d[raw], CAP))
        if not np.array_equal(exp, d[capd]):
            raise SystemExit(f"{tag}: {capd} != capped {raw} (NaN -> {CAP})")
    fo = d["fold_of"]
    if not np.issubdtype(fo.dtype, np.integer) or fo.min() < 0 or fo.max() >= N_FOLD:
        raise SystemExit(f"{tag}: fold_of must be integer 0..{N_FOLD - 1}")
    sg, sf = d["sel_groups_flat"], d["sel_groups_fold"]
    for f in range(N_FOLD):
        inner = set(sg[sf == f].astype(int).tolist())
        if not inner:
            raise SystemExit(f"{tag}: fold {f} has no inner-selection halos")
        if inner & set(d["sample_group"][fo == f].astype(int).tolist()):
            raise SystemExit(f"{tag}: fold {f} inner-selection halos overlap its outer halos")
    be, ss = d.get("fold_best_epoch"), d.get("fold_sel_score")
    if be is None or ss is None or be.shape != (N_FOLD,) or not np.issubdtype(be.dtype, np.integer) \
            or be.min() < 1 or be.max() > epochs or ss.shape != (N_FOLD,) or not np.isfinite(ss).all():
        raise SystemExit(f"{tag}: per-fold checkpoint record missing or invalid")
    if str(d.get("select_criterion", "")) != SELECT_CRITERION:
        raise SystemExit(f"{tag}: select_criterion {d.get('select_criterion')} != {SELECT_CRITERION}")


def check(runs, m, expect_manifest, epochs):
    lab0, s0 = LABELS[0], next(iter(runs[LABELS[0]]))
    ref = runs[lab0][s0]
    rprov = json.loads(str(ref["provenance"]))
    required = dict(REQUIRED_CFG, epochs=epochs, dataset=MODALITIES[m])
    for lab in LABELS:
        for s, d in runs[lab].items():
            tag = f"{m}/{lab} seed {s}"
            for k in IDENT + list(TRUTHS) + ["oof_single", "oof_sample", "label_name",
                                             "provenance", "provenance_changed_during_run"]:
                if k not in d:
                    raise SystemExit(f"{tag}: missing {k}")
            validate_artifact(tag, d, epochs)
            for k in IDENT:
                if k in ("y_sample", "tsc"):
                    continue                      # the training target differs by arm
                if d[k].shape != ref[k].shape or not np.array_equal(d[k], ref[k]):
                    raise SystemExit(f"{tag}: {k} differs from {lab0} seed {s0}")
            for k in TRUTHS:
                if not np.array_equal(d[k], ref[k], equal_nan=True):
                    raise SystemExit(f"{tag}: {k} differs between runs")
            if str(d["label_name"]) != lab:
                raise SystemExit(f"{tag}: label_name {d['label_name']} != {lab}")
            if not np.allclose(d["tsc"], d[TARGET_OF[lab]], atol=1e-6):
                raise SystemExit(f"{tag}: training target != {TARGET_OF[lab]}")
            R = len(d["y_sample"]) // len(d["tsc"])
            if not np.allclose(d["y_sample"], np.repeat(d["tsc"], R), atol=1e-6):
                raise SystemExit(f"{tag}: y_sample is not the declared target")
            for k in ("oof_single", "oof_sample"):
                if not np.isfinite(d[k]).all():
                    raise SystemExit(f"{tag}: non-finite {k}")
            prov = json.loads(str(d["provenance"]))
            cfg = prov["config"]
            if prov["manifest_sha256"] != expect_manifest:
                raise SystemExit(f"{tag}: manifest {prov['manifest_sha256']} != {expect_manifest}")
            if bool(d["provenance_changed_during_run"]):
                raise SystemExit(f"{tag}: sources/manifest/inputs changed during the run")
            if prov["sha256"] != rprov["sha256"] or prov.get("inputs_sha256") != rprov.get("inputs_sha256"):
                raise SystemExit(f"{tag}: source or input hashes differ between runs")
            if set(prov.get("inputs_sha256") or {}) != REQUIRED_INPUTS:
                raise SystemExit(f"{tag}: hashed inputs {sorted(prov.get('inputs_sha256') or {})}")
            if int(cfg["seed"]) != s or cfg.get("label") != lab:
                raise SystemExit(f"{tag}: config seed/label {cfg['seed']}/{cfg.get('label')}")
            for k, v in required.items():
                if cfg.get(k) != v:
                    raise SystemExit(f"{tag}: protocol requires {k}={v!r}, run has {cfg.get(k)!r}")
            strip = lambda c_: {k: v for k, v in c_.items() if k not in FREE}
            if strip(cfg) != strip(rprov["config"]):
                diff = sorted(k for k in set(strip(cfg)) | set(strip(rprov["config"]))
                              if strip(cfg).get(k) != strip(rprov["config"]).get(k))
                raise SystemExit(f"{tag}: undeclared configuration differences {diff}")
    return ref


def halo_boot(g, stat, n, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {h: np.where(g == h)[0] for h in ug}
    return np.percentile([stat(np.concatenate([idx[h] for h in rng.choice(ug, len(ug))]))
                          for _ in range(n)], [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-manifest", required=True)
    ap.add_argument("--mode", choices=["production", "smoke"], default="production")
    ap.add_argument("--pred-dir", default="cnn_preds")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--expect-epochs", type=int, default=None)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds, a.mode)
    if a.mode == "production" and a.expect_epochs not in (None, 60):
        raise SystemExit("production requires 60 epochs")
    if a.mode == "smoke" and a.expect_epochs is None:
        raise SystemExit("smoke mode needs --expect-epochs")
    epochs = 60 if a.mode == "production" else a.expect_epochs
    tag = "DIAGNOSTIC (smoke, not a result): " if a.mode == "smoke" else ""
    verdict = {}
    for m, ds in MODALITIES.items():
        runs = load(a.pred_dir, m, seeds)
        ref = check(runs, m, a.expect_manifest, epochs)
        msk = ref["sample_snap"] == 99
        P = P_EXPECT
        g = np.repeat(ref["sample_group"][msk][:, None], P, 1).ravel()
        fo = np.repeat(ref["fold_of"][msk][:, None], P, 1).ravel()
        tr = {k: np.repeat(np.repeat(ref[k], R_EXPECT)[msk][:, None], P, 1).ravel() for k in TRUTHS}
        ens = {lab: np.mean([d["oof_single"][msk].ravel() for d in runs[lab].values()], 0)
               for lab in LABELS}
        per = {lab: [d["oof_single"][msk].ravel() for d in runs[lab].values()] for lab in LABELS}
        y = tr["truth_tcoll_capped"]
        print(f"\n{tag}== {m}: {ds}; checks passed ({len(LABELS)} labels x seeds {seeds}); "
              f"snapshot-99 single observations n={len(g)}, {len(np.unique(g))} parent halos; "
              f"checkpoints selected on each arm's own target ({SELECT_CRITERION})")
        # baselines from each fold's FITTING label rows (train minus inner), each row once
        rows_g, rows_t = ref["group_id"], {k: ref[k] for k in ("truth_tcoll_capped", "truth_pseudo")}
        sg, sf, fo_s = ref["sel_groups_flat"], ref["sel_groups_fold"], ref["fold_of"]
        row_fold = fo_s[::R_EXPECT]
        for src in ("truth_tcoll_capped", "truth_pseudo"):
            for stat_name, fn in (("mean", np.mean), ("median", np.median)):
                pred = np.empty(len(y))
                for f in range(N_FOLD):
                    fit = (row_fold != f) & ~np.isin(rows_g, sg[sf == f])
                    pred[fo == f] = fn(rows_t[src][fit])
                print(f"  baseline: fitting-row {stat_name} of {src:<18} vs T_coll truth: "
                      f"R2 {r2_score(y, pred):+.3f}  MAE {mean_absolute_error(y, pred):.3f}")
        unc = np.isfinite(tr["truth_tcoll_raw"]) & (tr["truth_tcoll_raw"] <= CAP)
        for lab in LABELS:
            e = ens[lab]
            print(f"  trained on {lab:<10} vs T_coll truth: R2 {r2_score(y, e):.3f} "
                  f"Spearman {stats.spearmanr(y, e)[0]:.3f} MAE {mean_absolute_error(y, e):.3f} "
                  f"| vs pseudo truth: R2 {r2_score(tr['truth_pseudo'], e):.3f} "
                  f"Spearman {stats.spearmanr(tr['truth_pseudo'], e)[0]:.3f} "
                  f"| finite-below-cap rows ({unc.sum()} obs, {len(np.unique(g[unc]))} halos) Spearman "
                  f"{stats.spearmanr(tr['truth_tcoll_raw'][unc], e[unc])[0]:.3f}")
        b_, p_ = ens["tsc_tcoll"], ens["pseudo"]
        d_r2 = r2_score(y, b_) - r2_score(y, p_)
        d_sp = stats.spearmanr(y, b_)[0] - stats.spearmanr(y, p_)[0]
        d_ma = mean_absolute_error(y, b_) - mean_absolute_error(y, p_)
        iv = halo_boot(g, lambda s_: r2_score(y[s_], b_[s_]) - r2_score(y[s_], p_[s_]), a.boot)
        ivs = halo_boot(g, lambda s_: stats.spearmanr(y[s_], b_[s_])[0] - stats.spearmanr(y[s_], p_[s_])[0], a.boot)
        ivm = halo_boot(g, lambda s_: mean_absolute_error(y[s_], b_[s_]) - mean_absolute_error(y[s_], p_[s_]), a.boot)
        ds_ = [r2_score(y, pb) - r2_score(y, pa) for pa, pb in zip(per["pseudo"], per["tsc_tcoll"])]
        sd = f" ± {np.std(ds_, ddof=1):.3f} (s.d.)" if len(ds_) > 1 else ""
        print(f"  {tag}tsc_tcoll - pseudo on the T_coll truth: R2 {d_r2:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]; "
              f"Spearman {d_sp:+.3f} [{ivs[0]:+.3f}, {ivs[1]:+.3f}]; MAE {d_ma:+.3f} "
              f"[{ivm[0]:+.3f}, {ivm[1]:+.3f}] Gyr (paired halo bootstrap; conditional on the fits); "
              f"per-seed R2 diff {np.mean(ds_):+.3f}{sd}")
        yp = tr["truth_pseudo"]
        print(f"  secondary: pseudo - tsc_tcoll on the PSEUDO truth: R2 "
              f"{r2_score(yp, p_) - r2_score(yp, b_):+.3f}")
        verdict[m] = d_r2
    if a.mode == "production":
        ok = all(v >= -0.02 for v in verdict.values())
        print(f"\nprespecified practical tolerance (tcoll - pseudo R2 on the T_coll truth >= -0.02 in "
              f"both modalities; not noninferiority): {verdict} -> "
              f"{'within' if ok else 'outside'} the tolerance for further exploratory use of the "
              f"catalogue label (any manuscript change is a separate decision)")


if __name__ == "__main__":
    main()
