#!/usr/bin/env python3
"""Paired parent-halo bootstrap for the clean radio -> radio + X-ray increment
(Results 6.2 TODO; review rounds 62-63). No training: reuses the saved clean
pooled-CNN predictions of both arms.

Arms (train_cnn_pooled, pseudo-TSC, inner selection, seeds 42-46, snapshot 99,
one pooled prediction per parent):
  radio  cnn_preds/pooled_syn_inner_s{seed}.npz        (modality "single")
  joint  cnn_preds/joint_syn_clean_inner_s{seed}.npz   (modality "radio+xray")
Checks before scoring: compare_syn's clean-archive validation; exactly 352
snapshot-99 rows, one per unique halo and group; identical halo/group/snapshot
order and EXACTLY equal labels in all ten archives; declared modality per arm;
select = inner and recorded seed; five finite saved fold R2 per archive.
Partitions are reconstructed from the documented training protocol
(train_cnn_pooled.group_split + KFold(5, shuffle, random_state=seed)) in the
saved row order; agreement with both arms' saved fold R2 is a consistency
check, not recorded original membership. Inner selection is expected to agree
under that protocol but cannot be independently verified from these archives.

Primary (fixed, as the TODO asks): five-seed ensemble increment of POOLED clean
predictions, dR2 = R2(y, mean_s joint_s) - R2(y, mean_s radio_s), both arm
scores reported, 4000-draw paired parent bootstrap (rng seed 0), percentile
interval; the same resampled parents for both arms. Secondary: the mean
per-seed increment under the same draws (it holds the five fitted seeds fixed,
unlike the paired-seed test quoted in the text). Intervals condition on the
fitted CNNs. Undefined statistics stop the full-cohort analysis; invalid
bootstrap draws are counted and make that interval unavailable (null).

  score_clean_joint_bootstrap.py --out <file.json>
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

RADIO, JOINT = "cnn_preds/pooled_syn_inner_s{}.npz", "cnn_preds/joint_syn_clean_inner_s{}.npz"
SEEDS, N_FOLD, N_BOOT, N_PARENTS = [42, 43, 44, 45, 46], 5, 4000, 352
MODALITY = {"radio": "single", "joint": "radio+xray"}
SOURCES = ("score_clean_joint_bootstrap.py", "compare_syn.py", "train_cnn_pooled.py")


class Undefined(Exception):
    pass


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    v = np.sum((y - y.mean()) ** 2)
    if not (np.isfinite(v) and v > 0 and np.isfinite(p).all()):
        raise Undefined("R2 undefined (zero/non-finite target variance or non-finite predictions)")
    return float(1 - np.sum((y - p) ** 2) / v)


def reconstruct_folds(n, seed):
    from sklearn.model_selection import KFold
    from train_cnn_pooled import group_split
    fold = np.full(n, -1)
    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(group_split(np.arange(n), np.arange(n), kf.split)):
        fold[te] = f
    return fold


def paired_bootstrap(y, A, B, n=N_BOOT, seed=0):
    """A, B: (seeds, parents). Ensemble and mean-per-seed increments, each with
    a percentile interval from the SAME parent resamples. A full-cohort
    undefined statistic raises; undefined draws are counted and make the
    interval unavailable."""
    ea, eb = A.mean(0), B.mean(0)
    out = dict(ensemble=dict(radio=r2(y, ea), joint=r2(y, eb)))
    out["ensemble"]["delta"] = out["ensemble"]["joint"] - out["ensemble"]["radio"]
    per = np.array([r2(y, b) - r2(y, a) for a, b in zip(A, B)])
    out["mean_per_seed"] = dict(delta=float(per.mean()), sd=float(per.std(ddof=1)), per_seed=per.tolist())
    rng = np.random.default_rng(seed)
    be, bm, bad = [], [], 0
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        try:
            de = r2(y[i], eb[i]) - r2(y[i], ea[i])
            dm = float(np.mean([r2(y[i], b[i]) - r2(y[i], a[i]) for a, b in zip(A, B)]))
        except Undefined:
            bad += 1
            continue
        be.append(de)
        bm.append(dm)
    for key, v in (("ensemble", be), ("mean_per_seed", bm)):
        ok = bad == 0 and len(v) == n
        out[key].update(ci=[float(x) for x in np.percentile(v, [2.5, 97.5])] if ok else None,
                        interval_status="ok" if ok else f"unavailable: {bad} of {n} draws undefined")
    out["draws"], out["invalid_draws"] = n, bad
    return out


def validate(da, db, n_parents=N_PARENTS, folds_fn=reconstruct_folds):
    """Parent-level contract on loaded archives (lists of dicts, seed order)."""
    ref = da[0]
    y, hid, gid = ref["label"], ref["halo_id"], ref["group_id"]
    if len(y) != n_parents:
        raise SystemExit(f"expected {n_parents} rows, found {len(y)}")
    if len(np.unique(hid)) != len(hid) or len(np.unique(gid)) != len(gid):
        raise SystemExit("parents are not unique (one row per halo and group required)")
    for tag, ds in (("radio", da), ("joint", db)):
        if len(ds) != len(SEEDS):
            raise SystemExit(f"{tag}: expected {len(SEEDS)} seeds")
        for s, d in zip(SEEDS, ds):
            for k in ("halo_id", "group_id", "snap", "label", "oof_pred"):
                if np.shape(d[k]) != (n_parents,):
                    raise SystemExit(f"{tag} s{s}: {k} has shape {np.shape(d[k])}")
            if not (np.array_equal(d["halo_id"], hid) and np.array_equal(d["group_id"], gid)):
                raise SystemExit(f"{tag} s{s}: parent identities differ")
            if not np.array_equal(d["label"], y):
                raise SystemExit(f"{tag} s{s}: labels are not exactly equal")
            if not (d["snap"] == 99).all():
                raise SystemExit(f"{tag} s{s}: rows outside snapshot 99")
            if str(d["select"]) != "inner" or int(d["seed"]) != s:
                raise SystemExit(f"{tag} s{s}: selection/seed metadata")
            if str(d["modality"]) != MODALITY[tag]:
                raise SystemExit(f"{tag} s{s}: modality {d['modality']!r}, expected {MODALITY[tag]!r}")
            fr = np.asarray(d["fold_r2"], float)
            if fr.shape != (N_FOLD,) or not np.isfinite(fr).all() or not np.isfinite(d["oof_pred"]).all():
                raise SystemExit(f"{tag} s{s}: fold R2 or predictions not five finite values / finite")
    yf = np.asarray(y, float)
    resid = {}
    for s, a, b in zip(SEEDS, da, db):
        fold = folds_fn(n_parents, s)
        for tag, d in (("radio", a), ("joint", b)):
            rec = np.array([r2(yf[fold == f], d["oof_pred"][fold == f]) for f in range(N_FOLD)])
            dm = float(np.max(np.abs(rec - np.asarray(d["fold_r2"], float))))
            if not dm < 1e-4:
                raise SystemExit(f"{tag} s{s}: reconstructed partition inconsistent with saved fold R2 ({dm:.2e})")
            resid[f"{tag}_s{s}"] = dm
    A = np.array([d["oof_pred"] for d in da], float)
    B = np.array([d["oof_pred"] for d in db], float)
    return yf, hid, A, B, resid


def main(argv=None):
    import compare_syn as CS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    files = [p.format(s) for p in (RADIO, JOINT) for s in SEEDS]
    fp0 = {p: sha(p) for p in files + list(SOURCES)}
    da, db = CS.load(RADIO), CS.load(JOINT)          # clean-branch validation
    y, hid, A, B, resid = validate(da, db)
    try:
        res = paired_bootstrap(y, A, B)
    except Undefined as e:
        raise SystemExit(f"full-cohort statistic undefined: {e}")
    fp1 = {p: sha(p) for p in fp0}
    changed = sorted(p for p in fp0 if fp0[p] != fp1[p])
    if changed:
        raise SystemExit(f"inputs or sources changed during the analysis: {changed}")
    import sklearn
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--"] + list(SOURCES),
                           capture_output=True, text=True).stdout.strip()
    out = dict(arms=dict(radio=RADIO, joint=JOINT, modality=MODALITY), n_parents=len(y), seeds=SEEDS,
               parent_ids=hid.tolist(), fold_r2_consistency_residuals=resid,
               folds_note=("partitions reconstructed from the documented training protocol in the saved row "
                           "order; agreement with the saved fold R2 values is a consistency check, not recorded "
                           "original membership; inner selection is expected to agree under that protocol but "
                           "cannot be independently verified from these archives"),
               estimands=dict(primary="five-seed ensemble increment of pooled clean predictions (one per parent)",
                              secondary="mean per-seed increment; holds the five fitted seeds fixed, a different "
                                        "uncertainty from the paired-seed test quoted in the text"),
               bootstrap=f"{N_BOOT} paired parent draws, rng seed 0, percentile intervals", result=res,
               provenance=dict(input_sha256={p: fp0[p] for p in files},
                               source_sha256={p: fp0[p] for p in SOURCES}, git_head=head,
                               uncommitted_sources=dirty, unchanged_during_analysis=True,
                               versions=dict(python=platform.python_version(), numpy=np.__version__,
                                             sklearn=sklearn.__version__),
                               note="provenance of this analysis, not recovered training provenance"),
               note="conditional on the fitted CNNs; an interval spanning zero is unresolved, not equivalence")
    with open(a.out, "x") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    e, m = res["ensemble"], res["mean_per_seed"]
    fmt = lambda c: "unavailable" if c is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    print(f"five-seed ensemble (pooled): radio {e['radio']:.3f}, radio + X-ray {e['joint']:.3f}; "
          f"increment {e['delta']:+.3f} {fmt(e['ci'])}")
    print(f"mean per-seed increment {m['delta']:+.3f} +- {m['sd']:.3f} {fmt(m['ci'])}; "
          f"per seed {np.round(m['per_seed'], 3).tolist()}; invalid draws {res['invalid_draws']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
