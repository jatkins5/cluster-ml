#!/usr/bin/env python3
"""E1X comparison driver (review rounds 8-9): X-ray + idealised optical
structure, four matched arms, evaluated on single observations.

  compare_e1x.py --expect-manifest <sha256>

ARM CONTRACT. Every arm is a ViewScalarCNN run with the same five-slot
per-view vector; the ONLY declared configuration switches are
  arm        no_image   view_mask (structure slots zeroed)
  ctrl       True       gap12,off_bcg_x,off_bcg_gal,asym
  ctrlopt    True       (none)
  xctrl      False      gap12,off_bcg_x,off_bcg_gal,asym
  xctrlopt   False      (none)
plus seed and output path. Before scoring it requires, for all runs of all
arms: every seed present; identical shapes, sample identities and order,
labels, outer folds and inner-selection halos; the expected manifest
sha256; pooled training and single-view selection; the start-of-run source
AND input-file hashes (dataset, catalogue, masses, optical features)
identical across runs and unchanged during each run; the saved seed equal
to the filename's; identical global-scalar column order and raw values
(the actual noisy masses and placement redshifts); identical view-column
order and raw view inputs (richness included); the declared view mask; each
fold's fitted preprocessing identical; finite predictions.

Primary contrast: xctrlopt - xctrl (structure beyond X-ray + noisy mass +
placement redshift + richness). Secondary, separately labelled: ctrlopt -
ctrl (structure without images) and xctrl - ctrl (X-ray image beyond the
controls). Score: snapshot-99 single observations, each predicted by the
arm's 5-seed ensemble; paired parent-halo bootstrap R2/MAE intervals
(conditional on the fitted runs); per-seed single-model differences as the
seed scatter.
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from compare_e0 import IDENT

STRUCT = "gap12,off_bcg_x,off_bcg_gal,asym"
VIEW_COLS = ["richness"] + STRUCT.split(",")
ARMS = {  # name: (file pattern, no_image, view_mask config string)
    "ctrl": ("e1x_ctrl_s{}_preds.npz", True, STRUCT),
    "ctrlopt": ("e1x_ctrlopt_s{}_preds.npz", True, ""),
    "xctrl": ("e1x_xctrl_s{}_preds.npz", False, STRUCT),
    "xctrlopt": ("e1x_xctrlopt_s{}_preds.npz", False, ""),
}
CONTRASTS = [("PRIMARY", "xctrlopt", "xctrl", "structure beyond X-ray + controls"),
             ("secondary", "ctrlopt", "ctrl", "structure beyond controls, no image"),
             ("secondary", "xctrl", "ctrl", "X-ray image beyond controls")]
SWITCHES = ("no_image", "view_mask")
FREE = ("seed", "out_prefix")
EQUAL_ARRAYS = ["global_scalar_raw", "view_scalar_raw", "fold_scalar_mu", "fold_scalar_sd",
                "fold_view_median", "fold_view_mu", "fold_view_sd", "sim_logm_input"]
EQUAL_NAMES = ["global_cols", "view_cols"]


def load(pred_dir, seeds, arms=ARMS):
    runs = {}
    for a, (pat, _, _) in arms.items():
        runs[a] = {}
        for s in seeds:
            p = os.path.join(pred_dir, pat.format(s))
            if not os.path.exists(p):
                raise SystemExit(f"{a}: missing seed {s}: {p}")
            runs[a][s] = dict(np.load(p, allow_pickle=True))
    return runs


def check(runs, expect_manifest, arms=ARMS):
    """Raise SystemExit on any contract violation; return the reference run."""
    a0 = next(iter(arms))
    s0 = next(iter(runs[a0]))
    ref = runs[a0][s0]
    rprov = json.loads(str(ref["provenance"]))
    rcfg = rprov["config"]
    for a, (pat, no_image, mask) in arms.items():
        mask_vec = np.array([c in mask.split(",") for c in VIEW_COLS])
        for s, d in runs[a].items():
            tag = f"{a} seed {s}"
            for k in IDENT + EQUAL_ARRAYS + EQUAL_NAMES + [
                    "oof_single", "oof_sample", "view_mask", "provenance",
                    "provenance_changed_during_run"]:
                if k not in d:
                    raise SystemExit(f"{tag}: missing {k}")
            for k in IDENT + ["oof_single", "oof_sample"]:
                if d[k].shape != ref[k].shape:
                    raise SystemExit(f"{tag}: {k} shape {d[k].shape} != {ref[k].shape}")
            for k in IDENT:
                if not np.array_equal(d[k], ref[k]):
                    raise SystemExit(f"{tag}: {k} differs from {a0} seed {s0}")
            for k in ("oof_single", "oof_sample"):
                if not np.isfinite(d[k]).all():
                    raise SystemExit(f"{tag}: non-finite {k}")
            if d["oof_single"].shape[1] != 3 or not np.all(d["fold_of"] >= 0):
                raise SystemExit(f"{tag}: incomplete out-of-fold coverage")
            # common inputs: the actual values, not only the flags
            for k in EQUAL_NAMES:
                if list(d[k]) != list(ref[k]):
                    raise SystemExit(f"{tag}: {k} {list(d[k])} != {list(ref[k])}")
            if list(d["view_cols"]) != VIEW_COLS:
                raise SystemExit(f"{tag}: view_cols {list(d['view_cols'])} != {VIEW_COLS}")
            for k in EQUAL_ARRAYS:
                if d[k].shape != ref[k].shape or not np.array_equal(d[k], ref[k], equal_nan=True):
                    raise SystemExit(f"{tag}: {k} differs from {a0} seed {s0}")
            if not np.array_equal(d["view_mask"], mask_vec):
                raise SystemExit(f"{tag}: view_mask {d['view_mask']} != declared {mask_vec}")
            prov = json.loads(str(d["provenance"]))
            cfg = prov["config"]
            if prov["manifest_sha256"] != expect_manifest:
                raise SystemExit(f"{tag}: manifest {prov['manifest_sha256']} != {expect_manifest}")
            if bool(d["provenance_changed_during_run"]):
                raise SystemExit(f"{tag}: sources/manifest/inputs changed during the run")
            if prov["sha256"] != rprov["sha256"]:
                raise SystemExit(f"{tag}: defining-source hashes differ between runs")
            if not prov.get("inputs_sha256") or prov["inputs_sha256"] != rprov.get("inputs_sha256"):
                raise SystemExit(f"{tag}: input-file hashes missing or different: "
                                 f"{prov.get('inputs_sha256')}")
            if "view_scalars" not in prov["inputs_sha256"]:
                raise SystemExit(f"{tag}: optical feature file not hashed")
            if int(cfg["seed"]) != s:
                raise SystemExit(f"{tag}: file holds a run with seed {cfg['seed']}")
            if cfg["train_view"] != "pooled" or cfg["select_view"] != "single":
                raise SystemExit(f"{tag}: views {cfg['train_view']}/{cfg['select_view']} "
                                 f"!= pooled/single")
            if bool(cfg["no_image"]) != no_image or cfg["view_mask"] != mask:
                raise SystemExit(f"{tag}: switches no_image={cfg['no_image']} "
                                 f"view_mask={cfg['view_mask']!r} != declared "
                                 f"{no_image}/{mask!r}")
            strip = lambda c: {k: v for k, v in c.items() if k not in SWITCHES + FREE}
            if strip(cfg) != strip(rcfg):
                diff = sorted(k for k in set(strip(cfg)) | set(strip(rcfg))
                              if strip(cfg).get(k) != strip(rcfg).get(k))
                raise SystemExit(f"{tag}: undeclared configuration differences {diff}")
            if int(d["n_view"]) != len(VIEW_COLS):
                raise SystemExit(f"{tag}: n_view {int(d['n_view'])} (all arms must use "
                                 f"the {len(VIEW_COLS)}-slot ViewScalarCNN)")
    return ref


def single_obs(ref, arm_runs):
    m = ref["sample_snap"] == 99
    P = ref["oof_single"].shape[1]
    y = np.repeat(ref["y_sample"][m][:, None], P, 1).ravel()
    g = np.repeat(ref["sample_group"][m][:, None], P, 1).ravel()
    per_seed = [d["oof_single"][m].ravel() for d in arm_runs.values()]
    return y, g, per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-manifest", required=True)
    ap.add_argument("--pred-dir", default="cnn_preds")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = load(a.pred_dir, seeds)
    ref = check(runs, a.expect_manifest)
    print(f"checks passed: {len(ARMS)} arms x {len(seeds)} seeds; identical identities, "
          f"folds, inner sets, source and input hashes, raw scalar inputs and fold "
          f"preprocessing; declared switches only; manifest {a.expect_manifest[:12]}...")
    y, g, _ = single_obs(ref, runs["ctrl"])
    ps = {k: single_obs(ref, runs[k])[2] for k in ARMS}
    ens = {k: np.mean(v, 0) for k, v in ps.items()}
    print(f"\nsnapshot-99 single observations (n={len(y)}), {len(seeds)}-seed ensembles")
    for k in ARMS:
        print(f"  {k:<9} R2 {r2_score(y, ens[k]):.3f}  MAE {mean_absolute_error(y, ens[k]):.3f} Gyr")
    ug = np.unique(g)
    idx = {h: np.where(g == h)[0] for h in ug}
    for lab, b, a_, desc in CONTRASTS:
        rng = np.random.default_rng(0)
        dr, dm = [], []
        for _ in range(a.boot):
            sel = np.concatenate([idx[h] for h in rng.choice(ug, len(ug))])
            dr.append(r2_score(y[sel], ens[b][sel]) - r2_score(y[sel], ens[a_][sel]))
            dm.append(mean_absolute_error(y[sel], ens[b][sel])
                      - mean_absolute_error(y[sel], ens[a_][sel]))
        ds = [r2_score(y, pb) - r2_score(y, pa) for pa, pb in zip(ps[a_], ps[b])]
        print(f"\n{lab}: {b} - {a_} ({desc})")
        print(f"  R2 {r2_score(y, ens[b]) - r2_score(y, ens[a_]):+.3f} "
              f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]; MAE "
              f"{mean_absolute_error(y, ens[b]) - mean_absolute_error(y, ens[a_]):+.3f} "
              f"[{np.percentile(dm, 2.5):+.3f}, {np.percentile(dm, 97.5):+.3f}] Gyr "
              f"(paired halo bootstrap, {len(ug)} parent halos, {a.boot} draws; "
              f"conditional on the fitted runs)")
        sd = f" ± {np.std(ds, ddof=1):.3f}" if len(ds) > 1 else ""
        print(f"  seed scatter: per-seed single-model R2 difference {np.mean(ds):+.3f}{sd} "
              f"(seeds paired by index)")


if __name__ == "__main__":
    main()
