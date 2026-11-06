#!/usr/bin/env python3
"""Training-label comparison (2026-11-05): pseudo-TSC vs Chuiyang's catalogue
TSC (TSC_new: tsc_gyr and tsc_gyr_from_Tcoll), each capped at 4 Gyr with
NaN (no recorded collision) -> 4 Gyr.

  compare_labels.py --expect-manifest <sha256> [--mode smoke --seeds 42,43 --expect-epochs 1]

Two modalities, three labels each, seeds 42-46:
  x_* : X-ray only (xray_fix_s9991.h5; the E5-H anchor protocol)
  r_* : radio, physical 3 Mpc, regenerated radio model (injected_syn_phys_b3000_s9991.h5)
Within a modality the ONLY declared difference is --label (plus seed and
output). R2 against a model's own target is not comparable across labels
(different targets, different variance), so the PRIMARY score is how well
each model's snapshot-99 single-observation 5-seed ensemble recovers the
SAME truth -- the capped catalogue TSC from T_coll -- with paired
parent-halo bootstrap contrasts tcoll - pseudo and tsc - pseudo. The
reverse (each model against pseudo-TSC truth) and a check restricted to
uncensored rows (finite raw T_coll <= 4 Gyr) are reported as secondaries.
Frozen decision rule (proposed for review, before results): adopt the T_coll
catalogue TSC as the paper's target if, in BOTH modalities, the tcoll-trained
ensemble's R2 against the T_coll truth is no worse than the pseudo-trained
ensemble's by more than 0.02 (point estimate). The reason to switch is
interpretability, so a non-loss suffices; it is a practical rule, not an
equivalence test.
"""
import argparse
import json
import os

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

from compare_e0 import IDENT

PRODUCTION_SEEDS = [42, 43, 44, 45, 46]
LABELS = ("pseudo", "tsc", "tsc_tcoll")
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
            for k in IDENT:
                if k in ("y_sample", "tsc"):
                    continue                      # the training target differs by arm
                if d[k].shape != ref[k].shape or not np.array_equal(d[k], ref[k]):
                    raise SystemExit(f"{tag}: {k} differs from {lab0} seed {s0}")
            for k in TRUTHS:
                if not np.array_equal(d[k], ref[k], equal_nan=True):
                    raise SystemExit(f"{tag}: {k} differs between runs")
            for k in ("truth_pseudo", "truth_tsc_capped", "truth_tcoll_capped"):
                v = d[k]
                if v.shape != d["tsc"].shape or not np.isfinite(v).all() or v.min() < 0 or v.max() > 4.0 + 1e-9:
                    raise SystemExit(f"{tag}: {k} must be finite in [0, 4] per label row")
            for raw, capd in (("truth_tsc_raw", "truth_tsc_capped"), ("truth_tcoll_raw", "truth_tcoll_capped")):
                exp = np.where(np.isfinite(d[raw]), np.minimum(d[raw], 4.0), 4.0)
                if not np.array_equal(exp, d[capd]):
                    raise SystemExit(f"{tag}: {capd} != capped {raw} (NaN -> 4)")
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
        P = ref["oof_single"].shape[1]
        R = len(ref["y_sample"]) // len(ref["tsc"])
        g = np.repeat(ref["sample_group"][msk][:, None], P, 1).ravel()
        tr = {k: np.repeat(np.repeat(ref[k], R)[msk][:, None], P, 1).ravel() for k in TRUTHS}
        ens = {lab: np.mean([d["oof_single"][msk].ravel() for d in runs[lab].values()], 0)
               for lab in LABELS}
        per = {lab: [d["oof_single"][msk].ravel() for d in runs[lab].values()] for lab in LABELS}
        print(f"\n{tag}== {m}: {ds}; checks passed ({len(LABELS)} labels x seeds {seeds}); "
              f"snapshot-99 single observations n={len(g)}, {len(np.unique(g))} parent halos")
        unc = np.isfinite(tr["truth_tcoll_raw"]) & (tr["truth_tcoll_raw"] <= 4.0)
        for lab in LABELS:
            e = ens[lab]
            print(f"  trained on {lab:<10} vs T_coll TSC: R2 {r2_score(tr['truth_tcoll_capped'], e):.3f} "
                  f"Spearman {stats.spearmanr(tr['truth_tcoll_capped'], e)[0]:.3f} "
                  f"MAE {mean_absolute_error(tr['truth_tcoll_capped'], e):.3f} | vs pseudo: R2 "
                  f"{r2_score(tr['truth_pseudo'], e):.3f} Spearman {stats.spearmanr(tr['truth_pseudo'], e)[0]:.3f} "
                  f"| uncensored rows ({unc.sum()}) Spearman vs raw T_coll "
                  f"{stats.spearmanr(tr['truth_tcoll_raw'][unc], e[unc])[0]:.3f}")
        y = tr["truth_tcoll_capped"]
        for b in ("tsc_tcoll", "tsc"):
            d_r2 = r2_score(y, ens[b]) - r2_score(y, ens["pseudo"])
            iv = halo_boot(g, lambda s_: r2_score(y[s_], ens[b][s_]) - r2_score(y[s_], ens["pseudo"][s_]), a.boot)
            ivs = halo_boot(g, lambda s_: stats.spearmanr(y[s_], ens[b][s_])[0]
                            - stats.spearmanr(y[s_], ens["pseudo"][s_])[0], a.boot)
            ds_ = [r2_score(y, pb) - r2_score(y, pa) for pa, pb in zip(per["pseudo"], per[b])]
            sd = f" ± {np.std(ds_, ddof=1):.3f} (s.d.)" if len(ds_) > 1 else ""
            print(f"  {tag}{b} - pseudo, on the T_coll truth: R2 {d_r2:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]; "
                  f"Spearman {stats.spearmanr(y, ens[b])[0] - stats.spearmanr(y, ens['pseudo'])[0]:+.3f} "
                  f"[{ivs[0]:+.3f}, {ivs[1]:+.3f}] (paired halo bootstrap; conditional on the fits); "
                  f"per-seed R2 diff {np.mean(ds_):+.3f}{sd}")
            if b == "tsc_tcoll":
                verdict[m] = d_r2
        yp = tr["truth_pseudo"]
        d_rp = r2_score(yp, ens["pseudo"]) - r2_score(yp, ens["tsc_tcoll"])
        print(f"  secondary: pseudo - tsc_tcoll on the PSEUDO truth: R2 {d_rp:+.3f}")
    if a.mode == "production":
        ok = all(v >= -0.02 for v in verdict.values())
        print(f"\ndecision rule (frozen before results; tcoll - pseudo on the T_coll truth >= -0.02 "
              f"in both modalities): {verdict} -> {'adopt' if ok else 'do not adopt'} the T_coll "
              f"catalogue TSC as the paper's target")


if __name__ == "__main__":
    main()
