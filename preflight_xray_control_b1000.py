#!/usr/bin/env python3
"""Preflight for the matched 1 Mpc X-ray control (review round 59).

Runs at the start of the control's GPU allocation, BEFORE the first training
seed, and exits nonzero on any mismatch. It requires the control to reproduce
the 1 Mpc radio + X-ray joint model's protocol exactly:
  * the three frozen defining modules equal the hashes recorded in all five
    joint archives (and the archives are internally consistent and unchanged);
  * the radio, X-ray and catalogue inputs equal the joint's recorded input
    hashes, and the control dataset is the joint's recorded xray_dataset;
  * radio and X-ray mocks agree in row identity and metadata, rows are unique,
    and every parent's rows (both snapshots) share one outer fold;
  * the default outer folds and each seed's inner-selection parents,
    reconstructed with the frozen trainer's own group_folds and the trainer's
    GroupShuffleSplit(test_size=inner_frac, random_state=seed + k), k = 1..5,
    equal those saved in each joint archive, with fit/selection/test parents
    disjoint;
  * no output destination exists.
Writes <run-dir>/preflight/verified.json (assertions, fingerprints, package
versions) and the reconstructed splits. Historical artifacts are only read.
Kept outside the frozen bundle; imports group_folds from it.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import h5py
import numpy as np

FROZEN = Path("runs/followup_20261103T180830Z/code")
MODULES = ("train_cnn_mock.py", "train_cnn_pooled.py", "forward_model_lotss.py")
JOINT = "cnn_preds/syn_phys_b1000_joint_s{}_preds.npz"
XRAY = "xray_fix_s9991.h5"
SEEDS = list(range(42, 47))
META = ("group_id", "halo_id", "snap", "pseudo_tsc", "log_m500", "z")
EXPECT_CFG = dict(folds=5, epochs=60, batch_size=32, huber_delta=0.5, select="inner", inner_frac=0.15,
                  train_view="pooled", select_view="pooled", fold_manifest=None, loss="huber", mass=False,
                  no_image=False, xray_dataset=XRAY)


def fail(msg):
    raise SystemExit(f"PREFLIGHT FAILED: {msg}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    outs = [a.run_dir / "preds" / f"xray_b1000_s{s}_preds.npz" for s in SEEDS]
    if any(p.exists() for p in outs):
        fail(f"output destinations already exist: {[str(p) for p in outs if p.exists()]}")
    pre = a.run_dir / "preflight"
    if pre.exists():
        fail(f"{pre} already exists")
    checks = {}

    # joint archives: internally consistent, unchanged during their runs
    J = [dict(np.load(JOINT.format(s), allow_pickle=True)) for s in SEEDS]
    P = [json.loads(str(d["provenance"])) for d in J]
    for s, d, p in zip(SEEDS, J, P):
        if bool(d["provenance_changed_during_run"]):
            fail(f"joint s{s}: provenance changed during its run")
        if int(p["config"]["seed"]) != s:
            fail(f"joint s{s}: recorded seed {p['config']['seed']}")
        if p["sha256"] != P[0]["sha256"] or p["inputs_sha256"] != P[0]["inputs_sha256"]:
            fail(f"joint s{s}: source/input hashes differ from seed 42")
        strip = lambda c: {k: v for k, v in c.items() if k not in ("seed", "out_prefix")}
        if strip(p["config"]) != strip(P[0]["config"]):
            fail(f"joint s{s}: configuration differs from seed 42")
    cfg = P[0]["config"]
    for k, v in EXPECT_CFG.items():
        if cfg.get(k) != v:
            fail(f"joint config {k} = {cfg.get(k)!r}, expected {v!r}")
    checks["joint_archives"] = "5 seeds consistent, unchanged, expected protocol"

    # frozen defining modules == recorded hashes
    frozen = {m: sha256(FROZEN / m) for m in MODULES}
    if frozen != {m: P[0]["sha256"].get(m) for m in MODULES}:
        fail(f"frozen modules {frozen} differ from the joint's recorded hashes {P[0]['sha256']}")
    checks["frozen_modules"] = "equal to the joint archives' recorded hashes"

    # inputs == recorded input hashes
    radio, catalog = cfg["dataset"], cfg["catalog"]
    inputs = {"dataset": sha256(radio), "catalog": sha256(catalog), "xray_dataset": sha256(XRAY)}
    if inputs != P[0]["inputs_sha256"]:
        fail(f"input hashes {inputs} differ from the joint's recorded {P[0]['inputs_sha256']}")
    checks["inputs"] = "radio, catalogue and X-ray (control dataset = joint xray_dataset) hashes match"

    # metadata and row identity
    with h5py.File(radio) as r, h5py.File(XRAY) as x:
        for k in META:
            if not np.array_equal(r[f"mock/{k}"][:], x[f"mock/{k}"][:]):
                fail(f"radio and X-ray mock/{k} differ")
        shape = r["mock/images"].shape
        if shape != x["mock/images"].shape:
            fail(f"image shapes differ {shape} vs {x['mock/images'].shape}")
        gid, snap = r["mock/group_id"][:], r["mock/snap"][:]
    rows = list(zip(gid.tolist(), snap.tolist()))
    if len(set(rows)) != len(rows):
        fail("duplicate (group_id, snap) rows")
    for k in ("group_id", "snap"):
        if not np.array_equal(J[0][k], {"group_id": gid, "snap": snap}[k]):
            fail(f"joint archive {k} order differs from the datasets")
    N, R = shape[0], shape[1]
    groups = np.repeat(gid, R)
    checks["metadata"] = f"{list(META)} identical; {N} unique rows, {len(np.unique(gid))} parents, R={R}"

    # reconstruct splits with the frozen trainer's implementation
    sys.path.insert(0, str(FROZEN))
    from sklearn.model_selection import GroupShuffleSplit
    from train_cnn_mock import group_folds
    splits = list(group_folds(groups, cfg["folds"]))
    fold_of = np.full(len(groups), -1, dtype=np.int16)
    for k, (_, va) in enumerate(splits, 1):
        fold_of[va] = k - 1
    for g in np.unique(groups):
        if len(np.unique(fold_of[groups == g])) != 1:
            fail(f"parent {g} spans several outer folds")
    pre.mkdir(parents=True)
    per_seed = {}
    for s, d in zip(SEEDS, J):
        if not np.array_equal(d["fold_of"], fold_of):
            fail(f"seed {s}: reconstructed outer folds differ from the joint archive")
        sel_groups, sel_fold = [], []
        for k, (tr, va) in enumerate(splits, 1):
            gss = GroupShuffleSplit(n_splits=1, test_size=cfg["inner_frac"], random_state=s + k)
            rf, rs = next(gss.split(tr, groups=groups[tr]))
            fit, sel = set(groups[tr[rf]]), set(groups[tr[rs]])
            test = set(groups[va])
            if fit & sel or fit & test or sel & test:
                fail(f"seed {s} fold {k}: fit/selection/test parents overlap")
            sg = np.unique(groups[tr[rs]])
            sel_groups.append(sg)
            sel_fold.append(np.full(len(sg), k - 1))
        flat, fl = np.concatenate(sel_groups), np.concatenate(sel_fold)
        saved = set(zip(np.asarray(d["sel_groups_fold"]).astype(int).tolist(),
                        np.asarray(d["sel_groups_flat"]).astype(int).tolist()))
        if set(zip(fl.astype(int).tolist(), flat.astype(int).tolist())) != saved:
            fail(f"seed {s}: reconstructed inner-selection parents differ from the joint archive")
        np.savez(pre / f"splits_s{s}.npz", sample_group=groups, fold_of=fold_of, sel_groups_flat=flat,
                 sel_groups_fold=fl)
        per_seed[s] = dict(n_selection_parents=int(len(flat)))
    checks["splits"] = "outer folds and inner-selection parents reproduce all five joint archives; disjoint"

    import sklearn
    import torch
    rec = dict(status="passed", checks=checks, run_dir=str(a.run_dir), outputs=[str(p) for p in outs],
               joint_archives={JOINT.format(s): sha256(JOINT.format(s)) for s in SEEDS},
               frozen_modules=frozen, inputs={radio: inputs["dataset"], catalog: inputs["catalog"],
                                              XRAY: inputs["xray_dataset"]},
               joint_config=cfg, joint_head=P[0]["head"], shape=list(shape), per_seed=per_seed,
               versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                             torch=torch.__version__, h5py=h5py.__version__),
               host=platform.node())
    (pre / "verified.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("joint_config",)}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
