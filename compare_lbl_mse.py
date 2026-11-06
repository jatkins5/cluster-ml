#!/usr/bin/env python3
"""Phase 4 nominated follow-up (review rounds 33-36): MSE vs Huber, both
trained on the capped catalogue TSC (T_coll), X-ray only.

  compare_lbl_mse.py --expect-manifest <sha256> [--mode smoke --seeds 42,43 --expect-epochs 1]

Arms: huber = cnn_preds/lbl_x_tsc_tcoll_s{seed}_preds.npz (accepted Phase 3
artifacts), mse = cnn_preds/lbl_x_tsc_tcoll_mse_s{seed}_preds.npz. Contract:
every artifact passes compare_labels.validate_artifact (shapes, identities,
truths, cap, folds, inner sets, checkpoint record); both arms train on
tsc_tcoll; identities, labels, all truths, outer folds and inner (fold,
parent) memberships identical across every run; defining-source and input
hashes identical across every run and unchanged in-run; configuration
identical apart from loss (huber vs mse), seed and output path; the
expected manifest; production requires exactly seeds 42-46.

PRIMARY (frozen): MSE - Huber snapshot-99 single-observation 5-seed-ensemble
R2 against the capped T_coll truth, paired parent-halo bootstrap (4000
draws); MAE and Spearman contrasts and per-seed scatter alongside.
SECONDARY: per TSC-band bias and MSE for both arms (Tier-0 strata).
Interpretation: the effect of choosing MSE instead of Huber under the fixed
protocol; not a measurement of delta-specific shrinkage, not a sweep.
"""
import argparse
import json
import os

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

import compare_labels as L
from compare_e0 import IDENT

ARMS = {"huber": "lbl_x_tsc_tcoll_s{}_preds.npz", "mse": "lbl_x_tsc_tcoll_mse_s{}_preds.npz"}
FREE = {"seed", "out_prefix", "loss"}
TRUTHS = L.TRUTHS


def load(pred_dir, seeds):
    runs = {}
    for arm, pat in ARMS.items():
        runs[arm] = {}
        for s in seeds:
            p = os.path.join(pred_dir, pat.format(s))
            if not os.path.exists(p):
                raise SystemExit(f"{arm}: missing seed {s}: {p}")
            runs[arm][s] = dict(np.load(p, allow_pickle=True))
    return runs


def _inner(d):
    return set(zip(d["sel_groups_fold"].astype(int).tolist(), d["sel_groups_flat"].astype(int).tolist()))


def check(runs, expect_manifest, epochs):
    ref = runs["huber"][next(iter(runs["huber"]))]
    rp = json.loads(str(ref["provenance"]))
    for arm, rr in runs.items():
        for s, d in rr.items():
            tag = f"{arm} seed {s}"
            L.validate_artifact(tag, d, epochs)
            for k in IDENT + list(TRUTHS) + ["group_id", "snap"]:
                if not np.array_equal(d[k], ref[k], equal_nan=True):
                    raise SystemExit(f"{tag}: {k} differs from the reference run")
            if _inner(d) != _inner(ref):
                raise SystemExit(f"{tag}: inner (fold, parent) memberships differ")
            if str(d["label_name"]) != "tsc_tcoll":
                raise SystemExit(f"{tag}: label {d['label_name']} != tsc_tcoll")
            if bool(d.get("provenance_changed_during_run", True)):
                raise SystemExit(f"{tag}: changed-during-run flag missing or true")
            p = json.loads(str(d["provenance"]))
            c = p["config"]
            if p["manifest_sha256"] != expect_manifest:
                raise SystemExit(f"{tag}: manifest {p['manifest_sha256']} != {expect_manifest}")
            if p["sha256"] != rp["sha256"]:
                raise SystemExit(f"{tag}: defining-source hashes differ")
            if not p.get("inputs_sha256") or p["inputs_sha256"] != rp["inputs_sha256"]:
                raise SystemExit(f"{tag}: input hashes missing or different")
            if int(c["seed"]) != s:
                raise SystemExit(f"{tag}: recorded seed {c['seed']} != {s}")
            if c.get("loss") != arm or str(d.get("loss", "")) != arm:
                raise SystemExit(f"{tag}: loss {c.get('loss')}/{d.get('loss')} != declared {arm}")
            if c.get("label") != "tsc_tcoll" or c.get("dataset") != L.MODALITIES["x"]:
                raise SystemExit(f"{tag}: label/dataset {c.get('label')}/{c.get('dataset')}")
            strip = lambda x: {k: v for k, v in x.items() if k not in FREE}
            if strip(c) != strip(rp["config"]):
                diff = sorted(k for k in set(strip(c)) | set(strip(rp["config"]))
                              if strip(c).get(k) != strip(rp["config"]).get(k))
                raise SystemExit(f"{tag}: undeclared configuration differences {diff}")
    return ref


def boot(g, stat, n, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {h: np.where(g == h)[0] for h in ug}
    return np.percentile([stat(np.concatenate([idx[h] for h in rng.choice(ug, len(ug))])) for _ in range(n)],
                         [2.5, 97.5])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-manifest", required=True)
    ap.add_argument("--mode", choices=["production", "smoke"], default="production")
    ap.add_argument("--pred-dir", default="cnn_preds")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--expect-epochs", type=int, default=None)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args(argv)
    seeds = L.parse_seeds(a.seeds, a.mode)
    if a.mode == "production" and a.expect_epochs not in (None, 60):
        raise SystemExit("production requires 60 epochs")
    if a.mode == "smoke" and a.expect_epochs is None:
        raise SystemExit("smoke mode needs --expect-epochs")
    epochs = 60 if a.mode == "production" else a.expect_epochs
    tag = "DIAGNOSTIC (smoke, not a result): " if a.mode == "smoke" else ""
    runs = load(a.pred_dir, seeds)
    ref = check(runs, a.expect_manifest, epochs)
    R, P = L.R_EXPECT, L.P_EXPECT
    ms = ref["sample_snap"] == 99
    rep = lambda v: np.repeat(np.repeat(np.asarray(v), R)[ms][:, None], P, 1).ravel()
    y, g = rep(ref["truth_tcoll_capped"]).astype(np.float64), rep(ref["group_id"])
    raw = rep(ref["truth_tcoll_raw"]).astype(np.float64)
    ens = {k: np.mean([d["oof_single"][ms].astype(np.float64).ravel() for d in v.values()], 0) for k, v in runs.items()}
    per = {k: [d["oof_single"][ms].astype(np.float64).ravel() for d in v.values()] for k, v in runs.items()}
    print(f"{tag}checks passed: huber vs mse on tsc_tcoll, seeds {seeds}; identical identities, truths, folds, inner "
          f"sets, sources and inputs; loss the only declared difference; n={len(y)} snapshot-99 single observations, "
          f"{len(np.unique(g))} parents")
    for k in ARMS:
        print(f"  {k:<6} R2 {r2_score(y, ens[k]):.3f}  MAE {mean_absolute_error(y, ens[k]):.3f}  "
              f"Spearman {stats.spearmanr(y, ens[k])[0]:.3f}")
    b, h = ens["mse"], ens["huber"]
    dr = r2_score(y, b) - r2_score(y, h)
    dm = mean_absolute_error(y, b) - mean_absolute_error(y, h)
    dsp = stats.spearmanr(y, b)[0] - stats.spearmanr(y, h)[0]
    ir = boot(g, lambda s: r2_score(y[s], b[s]) - r2_score(y[s], h[s]), a.boot)
    im = boot(g, lambda s: mean_absolute_error(y[s], b[s]) - mean_absolute_error(y[s], h[s]), a.boot)
    isp = boot(g, lambda s: stats.spearmanr(y[s], b[s])[0] - stats.spearmanr(y[s], h[s])[0], a.boot)
    ds = [r2_score(y, pb) - r2_score(y, ph) for ph, pb in zip(per["huber"], per["mse"])]
    sd = f" ± {np.std(ds, ddof=1):.3f} (s.d.)" if len(ds) > 1 else ""
    print(f"\n{tag}PRIMARY mse - huber (capped T_coll truth): R2 {dr:+.3f} [{ir[0]:+.3f}, {ir[1]:+.3f}]; "
          f"MAE {dm:+.3f} [{im[0]:+.3f}, {im[1]:+.3f}] Gyr; Spearman {dsp:+.3f} [{isp[0]:+.3f}, {isp[1]:+.3f}] "
          f"(paired parent bootstrap, {a.boot} draws; conditional on the fits); per-seed R2 diff {np.mean(ds):+.3f}{sd}")
    print(f"\n{tag}SECONDARY: TSC-band bias / MSE (huber | mse)")
    import diagnose_tsc_hardness as D
    st = np.array([D.tsc_stratum(x) for x in raw])
    for lev in D.TSC_STRATA:
        o = st == lev
        if not o.any():
            print(f"  {lev:<10} empty")
            continue
        print(f"  {lev:<10} obs {int(o.sum()):>5} | bias {np.mean(h[o] - y[o]):+.3f} | {np.mean(b[o] - y[o]):+.3f} | "
              f"MSE {np.mean((h[o] - y[o]) ** 2):.3f} | {np.mean((b[o] - y[o]) ** 2):.3f}")


if __name__ == "__main__":
    main()
