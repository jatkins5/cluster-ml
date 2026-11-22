#!/usr/bin/env python3
"""Paired parent-halo bootstrap for the clean radio -> radio + X-ray increment
(Results 6.2 TODO; review round 62). No training: reuses the saved clean
pooled-CNN predictions of both arms.

Arms (train_cnn_pooled, pseudo-TSC, inner selection, seeds 42-46, snapshot 99,
one pooled prediction per parent):
  radio  cnn_preds/pooled_syn_inner_s{seed}.npz
  joint  cnn_preds/joint_syn_clean_inner_s{seed}.npz
Checks before scoring: compare_syn's clean-archive validation; identical
parent order and labels in every archive; snapshot 99 only; select = inner;
each seed's outer partition reconstructed with train_cnn_pooled.group_split +
KFold(5, shuffle, random_state=seed) reproduces the five saved fold R2 of BOTH
arms (so the arms share each seed's outer folds).

Primary (as the TODO asks): five-seed ensemble increment
  dR2 = R2(y, mean_s joint_s) - R2(y, mean_s radio_s)
with a 4000-draw paired parent bootstrap (rng seed 0, as compare_syn); the
same resampled parents are used for both arms. Secondary: the mean per-seed
increment (the +0.037 +- 0.032 quoted in the text) under the same draws, and
the per-seed values. Intervals condition on the fitted CNNs.

  score_clean_joint_bootstrap.py --out <file.json>
"""
import argparse
import json
import os
import sys

import numpy as np

import compare_syn as CS

RADIO, JOINT = "cnn_preds/pooled_syn_inner_s{}.npz", "cnn_preds/joint_syn_clean_inner_s{}.npz"
SEEDS, N_FOLD, N_BOOT = CS.SEEDS, 5, 4000


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def reconstruct_folds(n, seed):
    from sklearn.model_selection import KFold
    from train_cnn_pooled import group_split
    fold = np.full(n, -1)
    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(group_split(np.arange(n), np.arange(n), kf.split)):
        fold[te] = f
    return fold


def paired_bootstrap(y, A, B, n=N_BOOT, seed=0):
    """A, B: (seeds, parents). Returns ensemble and mean-per-seed increments and
    their percentile intervals from the SAME parent resamples."""
    rng = np.random.default_rng(seed)
    ea, eb = A.mean(0), B.mean(0)
    ens = r2(y, eb) - r2(y, ea)
    per = np.array([r2(y, b) - r2(y, a) for a, b in zip(A, B)])
    be, bm = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        be.append(r2(y[i], eb[i]) - r2(y[i], ea[i]))
        bm.append(np.mean([r2(y[i], b[i]) - r2(y[i], a[i]) for a, b in zip(A, B)]))
    q = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return dict(ensemble=dict(delta=ens, ci=q(be), radio=r2(y, ea), joint=r2(y, eb)),
                mean_per_seed=dict(delta=float(per.mean()), sd=float(per.std(ddof=1)), ci=q(bm),
                                   per_seed=per.tolist()))


def load_checked():
    da, db = CS.load(RADIO), CS.load(JOINT)          # clean-branch validation
    y, hid = da[0]["label"].astype(float), da[0]["halo_id"]
    n = len(y)
    for tag, ds in (("radio", da), ("joint", db)):
        for s, d in zip(SEEDS, ds):
            if not (np.array_equal(d["halo_id"], hid) and np.allclose(d["label"], y, atol=1e-6)):
                raise SystemExit(f"{tag} s{s}: parent order or labels differ")
            if not ((d["snap"] == 99).all() and str(d["select"]) == "inner" and int(d["seed"]) == s):
                raise SystemExit(f"{tag} s{s}: snapshot/selection/seed metadata")
    fold_check = {}
    for s, a, b in zip(SEEDS, da, db):
        fold = reconstruct_folds(n, s)
        for tag, d in (("radio", a), ("joint", b)):
            rec = np.array([r2(y[fold == f], d["oof_pred"][fold == f]) for f in range(N_FOLD)])
            dm = float(np.max(np.abs(rec - d["fold_r2"])))
            if not dm < 1e-4:
                raise SystemExit(f"{tag} s{s}: reconstructed outer folds do not reproduce saved fold R2 ({dm:.2e})")
            fold_check[f"{tag}_s{s}"] = dm
    A = np.array([d["oof_pred"] for d in da], float)
    B = np.array([d["oof_pred"] for d in db], float)
    return y, hid, A, B, fold_check, [str(d["modality"]) for d in (da[0], db[0])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    y, hid, A, B, fold_check, modality = load_checked()
    res = paired_bootstrap(y, A, B)
    out = dict(arms=dict(radio=RADIO, joint=JOINT, modality=modality), n_parents=len(y), seeds=SEEDS,
               fold_r2_reproduction_maxdiff=fold_check,
               folds_note=("each seed's outer partition reconstructed with train_cnn_pooled.group_split + "
                           "KFold(5, shuffle, random_state=seed) reproduces both arms' saved fold R2; the "
                           "inner-selection split (ShuffleSplit, random_state=seed+fold) is not saved in these "
                           "archives and is shared by construction, not verified"),
               bootstrap=f"{N_BOOT} paired parent draws, rng seed 0", result=res,
               note="conditional on the fitted CNNs; an interval spanning zero is unresolved, not equivalence")
    with open(a.out, "x") as f:
        json.dump(out, f, indent=1)
    e, m = res["ensemble"], res["mean_per_seed"]
    print(f"five-seed ensemble: radio {e['radio']:.3f}, radio + X-ray {e['joint']:.3f}; "
          f"increment {e['delta']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]")
    print(f"mean per-seed increment {m['delta']:+.3f} +- {m['sd']:.3f} "
          f"[{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}]; per seed {np.round(m['per_seed'], 3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
