#!/usr/bin/env python3
"""E0 comparison driver (review rounds 4-6): pooled-view vs single-view
training, both selected on single views, evaluated on single observations.

  compare_e0.py --a e0_x_pooled_s{}_preds.npz --b e0_x_single_s{}_preds.npz \
      --expect-manifest <sha256>

Before scoring it requires: all five seeds per arm; identical shapes, then
identical sample identities and order (halo, parent, snapshot, realization;
projection = column of oof_single), labels, outer folds and inner-selection
halos across all runs of both arms; the expected manifest sha256 in every
run's start-of-run provenance; single-view selection in both arms; the
training view as the only declared difference; finite predictions; and, if
mass inputs are present, identical fixed measurements.

Primary score: snapshot-99 single observations (each realization x
projection separately), each predicted by the arm's 5-seed ensemble (mean
of the five seed models' predictions for THAT observation; no averaging of
views or realizations). R2 and MAE per arm, the paired halo-bootstrap
difference B - A (whole parent halos resampled; conditional on the fitted
runs), and the per-seed single-model differences (seed scatter) separately.
Pooled-view scores are reported as a separate diagnostic.
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

SEEDS = list(range(42, 47))
IDENT = ["sample_halo", "sample_group", "sample_snap", "sample_realization",
         "y_sample", "tsc", "fold_of", "sel_groups_flat", "sel_groups_fold"]


def load_arm(pattern, pred_dir):
    runs = {}
    for s in SEEDS:
        p = os.path.join(pred_dir, pattern.format(s))
        if not os.path.exists(p):
            raise SystemExit(f"missing seed {s}: {p}")
        runs[s] = np.load(p, allow_pickle=True)
    return runs


def check(arms, expect_manifest):
    ref = next(iter(arms["A"].values()))
    for name, runs in arms.items():
        for s, d in runs.items():
            for k in IDENT + ["oof_single", "oof_sample"]:
                if k not in d:
                    raise SystemExit(f"{name} seed {s}: missing {k}")
                if d[k].shape != ref[k].shape:
                    raise SystemExit(f"{name} seed {s}: {k} shape {d[k].shape} != {ref[k].shape}")
            for k in IDENT:
                if not np.array_equal(d[k], ref[k]):
                    raise SystemExit(f"{name} seed {s}: {k} differs from arm A seed {SEEDS[0]}")
            for k in ("oof_single", "oof_sample"):
                if not np.isfinite(d[k]).all():
                    raise SystemExit(f"{name} seed {s}: non-finite {k}")
            if str(d["select_view"]) != "single":
                raise SystemExit(f"{name} seed {s}: select_view {d['select_view']} != single")
            prov = json.loads(str(d["provenance"]))
            if prov["manifest_sha256"] != expect_manifest:
                raise SystemExit(f"{name} seed {s}: manifest {prov['manifest_sha256']} "
                                 f"!= expected {expect_manifest}")
            if "provenance_changed_during_run" not in d:
                raise SystemExit(f"{name} seed {s}: no provenance_changed_during_run flag")
            if bool(d["provenance_changed_during_run"]):
                raise SystemExit(f"{name} seed {s}: sources/manifest changed during the run")
            # identical defining-source CONTENT across all runs (HEADs may
            # differ through unrelated commits; hashes may not)
            if prov["sha256"] != json.loads(str(ref["provenance"]))["sha256"]:
                raise SystemExit(f"{name} seed {s}: defining-source hashes differ between runs")
            # the saved seed must be the one the filename claims
            if int(prov["config"]["seed"]) != s:
                raise SystemExit(f"{name} file for seed {s} holds a run with seed "
                                 f"{prov['config']['seed']}")
            if "sim_logm_input" in d and d["sim_logm_input"].size:
                if not np.array_equal(d["sim_logm_input"], ref["sim_logm_input"]):
                    raise SystemExit(f"{name} seed {s}: mass measurements differ")
    va = {str(d["train_view"]) for d in arms["A"].values()}
    vb = {str(d["train_view"]) for d in arms["B"].values()}
    if len(va) != 1 or len(vb) != 1 or va == vb:
        raise SystemExit(f"train_view must be constant within and differ between arms: A {va}, B {vb}")
    # the training view must be the only declared configuration difference
    cfg = lambda d: {k: v for k, v in json.loads(str(d["provenance"]))["config"].items()
                     if k not in ("train_view", "seed", "out_prefix")}
    ca = cfg(next(iter(arms["A"].values())))
    for name, runs in arms.items():
        for s, d in runs.items():
            if cfg(d) != ca:
                diff = {k for k in set(ca) | set(cfg(d)) if ca.get(k) != cfg(d).get(k)}
                raise SystemExit(f"{name} seed {s}: config differs beyond train_view: {diff}")
    return ref, va.pop(), vb.pop()


def single_obs(ref, runs):
    m = ref["sample_snap"] == 99
    P = ref["oof_single"].shape[1]
    y = np.repeat(ref["y_sample"][m][:, None], P, 1).ravel()
    g = np.repeat(ref["sample_group"][m][:, None], P, 1).ravel()
    per_seed = [d["oof_single"][m].ravel() for d in runs.values()]
    return y, g, per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--expect-manifest", required=True)
    ap.add_argument("--pred-dir", default="cnn_preds")
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    arms = {"A": load_arm(a.a, a.pred_dir), "B": load_arm(a.b, a.pred_dir)}
    ref, va, vb = check(arms, a.expect_manifest)
    print(f"checks passed: 5+5 seeds, identical identities/labels/folds/inner sets, "
          f"manifest {a.expect_manifest[:12]}..., single-view selection; "
          f"A trains {va}, B trains {vb}")
    y, g, pa = single_obs(ref, arms["A"])
    _, _, pb = single_obs(ref, arms["B"])
    ea, eb = np.mean(pa, 0), np.mean(pb, 0)
    print(f"\nPRIMARY: snapshot-99 single observations (n={len(y)}), 5-seed ensembles")
    for lab, e in (("A " + va, ea), ("B " + vb, eb)):
        print(f"  {lab:<10} R2 {r2_score(y, e):.3f}  MAE {mean_absolute_error(y, e):.3f} Gyr")
    rng = np.random.default_rng(0)
    ug = np.unique(g)
    idx = {h: np.where(g == h)[0] for h in ug}
    dr, dm = [], []
    for _ in range(a.boot):
        sel = np.concatenate([idx[h] for h in rng.choice(ug, len(ug))])
        dr.append(r2_score(y[sel], eb[sel]) - r2_score(y[sel], ea[sel]))
        dm.append(mean_absolute_error(y[sel], eb[sel]) - mean_absolute_error(y[sel], ea[sel]))
    print(f"  B - A: R2 {r2_score(y, eb) - r2_score(y, ea):+.3f} "
          f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]; "
          f"MAE {mean_absolute_error(y, eb) - mean_absolute_error(y, ea):+.3f} "
          f"[{np.percentile(dm, 2.5):+.3f}, {np.percentile(dm, 97.5):+.3f}] Gyr "
          f"(paired halo bootstrap, {len(ug)} parent halos, {a.boot} draws; "
          f"conditional on the fitted runs)")
    ds = [r2_score(y, b_) - r2_score(y, a_) for a_, b_ in zip(pa, pb)]
    print(f"  seed scatter: per-seed single-model R2 B - A {np.mean(ds):+.3f} ± "
          f"{np.std(ds, ddof=1):.3f} (seeds paired by index)")
    m = ref["sample_snap"] == 99
    ya = ref["y_sample"][m]
    pool = lambda runs: np.mean([d["oof_sample"][m] for d in runs.values()], 0)
    print(f"\nDIAGNOSTIC (pooled views, not the primary estimator): 5-seed ensemble R2 "
          f"A {r2_score(ya, pool(arms['A'])):.3f}, B {r2_score(ya, pool(arms['B'])):.3f}")


if __name__ == "__main__":
    main()
