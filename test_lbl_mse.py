#!/usr/bin/env python3
"""Fixtures for compare_lbl_mse.py -- synthetic artifacts, no data analysis."""
import copy
import json
import os
import tempfile

import numpy as np

import compare_labels as L
import compare_lbl_mse as M

MAN = "f" * 64
N, R, P, F = 10, 3, 3, 5
GRP = np.tile(np.arange(5), 2)
SN = np.r_[np.full(5, 99), np.full(5, 91)]
raw = np.array([0.0, 0.6, np.nan, 2.2, 5.0, 0.25, 0.9, 1.4, 3.5, 0.1])
cap = np.where(np.isnan(raw), 4.0, np.minimum(raw, 4.0))
pseudo = np.array([0.1, 0.5, 4.0, 2.0, 4.0, 0.3, 0.8, 1.5, 3.0, 0.2])


def make(arm, s):
    cfg = dict(L.REQUIRED_CFG, epochs=60, dataset=L.MODALITIES["x"], seed=s, label="tsc_tcoll",
               out_prefix=f"{arm}{s}", tsc_dir="/x")
    cfg["loss"] = arm
    prov = dict(head="h", status="", sha256={"train_cnn_mock.py": "a", "train_cnn_pooled.py": "b",
                                             "forward_model_lotss.py": "c"}, config=cfg, manifest_sha256=MAN,
                inputs_sha256={"dataset": "d", "catalog": "c", "tsc_snap99": "t9", "tsc_snap91": "t1"})
    rg = np.random.default_rng(s + (100 if arm == "mse" else 0))
    return dict(sample_halo=np.repeat(np.arange(N), R), sample_group=np.repeat(GRP, R), sample_snap=np.repeat(SN, R),
                sample_realization=np.tile(np.arange(R), N), group_id=GRP, snap=SN,
                y_sample=np.repeat(cap, R).astype(np.float32), tsc=cap.astype(np.float32),
                fold_of=np.repeat(GRP % F, R).astype(np.int16),
                sel_groups_flat=np.array([(f + 1) % F for f in range(F)]), sel_groups_fold=np.arange(F),
                tsc_cap=np.array(4.0), fold_best_epoch=np.full(F, 30), fold_sel_score=np.full(F, 0.3),
                select_criterion=np.array("inner:single:r2"), truth_pseudo=pseudo, truth_tsc_capped=cap,
                truth_tsc_raw=raw, truth_tcoll_capped=cap, truth_tcoll_raw=raw, label_name=np.array("tsc_tcoll"),
                loss=np.array(arm),
                oof_sample=(np.repeat(cap, R) + rg.normal(0, 0.3, N * R)).astype(np.float32),
                oof_single=(np.repeat(cap, R)[:, None] + rg.normal(0, 0.3, (N * R, P))).astype(np.float32),
                provenance=np.array(json.dumps(prov)), provenance_changed_during_run=np.array(False))


runs = {arm: {s: make(arm, s) for s in (42, 43)} for arm in M.ARMS}
M.check(runs, MAN, 60)
d_ = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "."))
for arm, pat in M.ARMS.items():
    for s, d in runs[arm].items():
        np.savez(os.path.join(d_, pat.format(s)), **d)
M.main(["--expect-manifest", MAN, "--mode", "smoke", "--seeds", "42,43", "--expect-epochs", "60",
        "--pred-dir", d_, "--boot", "50"])
print("    valid synthetic matrix accepted and scored end to end")


def expect(what, f):
    r = copy.deepcopy(runs)
    f(r)
    try:
        M.check(r, MAN, 60)
    except SystemExit as e:
        print(f"    rejected ({what}): {e}")
        return
    raise SystemExit(f"NOT rejected: {what}")


def cfg(arm, s, k, v):
    def f(r):
        p = json.loads(str(r[arm][s]["provenance"])); p["config"][k] = v
        r[arm][s]["provenance"] = np.array(json.dumps(p))
    return f


def both_huber(r):
    for s in r["mse"]:
        cfg("mse", s, "loss", "huber")(r); r["mse"][s]["loss"] = np.array("huber")


expect("both arms Huber", both_huber)
expect("mse arm trained on pseudo", lambda r: r["mse"][43].__setitem__("label_name", np.array("pseudo")))
expect("undeclared epochs change", cfg("mse", 43, "epochs", 30))
expect("outer folds differ", lambda r: r["mse"][43].__setitem__("fold_of", np.repeat((GRP + 1) % F, R).astype(np.int16)))
expect("inner folds reassigned", lambda r: r["mse"][42].__setitem__("sel_groups_fold", np.array([1, 0, 2, 3, 4])))
expect("truth differs", lambda r: r["mse"][43].__setitem__("truth_pseudo", pseudo * 0.9))
expect("wrong manifest", lambda r: r["mse"][42].__setitem__(
    "provenance", np.array(json.dumps(dict(json.loads(str(r["mse"][42]["provenance"])), manifest_sha256="0" * 64)))))
expect("changed-during-run flag missing", lambda r: r["huber"][42].pop("provenance_changed_during_run"))
expect("defining sources differ", lambda r: r["mse"][43].__setitem__("provenance", np.array(json.dumps(
    dict(json.loads(str(r["mse"][43]["provenance"])), sha256={"train_cnn_mock.py": "z", "train_cnn_pooled.py": "b",
                                                             "forward_model_lotss.py": "c"})))))


def all_runs(f):
    def g(r):
        for arm in r:
            for s in r[arm]:
                f(r[arm][s])
    return g


def set_cfg(k, v):
    def f(d):
        p = json.loads(str(d["provenance"])); p["config"][k] = v; d["provenance"] = np.array(json.dumps(p))
    return f


def epochs30(d):
    set_cfg("epochs", 30)(d); d["fold_best_epoch"] = np.full(F, 30)


def drop_input(d):
    p = json.loads(str(d["provenance"])); p["inputs_sha256"].pop("tsc_snap91"); d["provenance"] = np.array(json.dumps(p))


def empty_sources(d):
    p = json.loads(str(d["provenance"])); p["sha256"] = {}; d["provenance"] = np.array(json.dumps(p))


def wrong_target(d):
    d["tsc"] = pseudo.astype(np.float32); d["y_sample"] = np.repeat(pseudo, R).astype(np.float32)


def bad_expansion(d):
    ys = d["y_sample"].copy(); ys[0] += 0.5; d["y_sample"] = ys


def nan_pred(d):
    o = d["oof_single"].copy(); o[3, 1] = np.nan; d["oof_single"] = o


expect("config epochs 30 in EVERY artifact (checkpoints 30)", all_runs(epochs30))
expect("single-view training in EVERY artifact", all_runs(set_cfg("train_view", "single")))
expect("a required input hash missing in EVERY artifact", all_runs(drop_input))
expect("empty defining-source record in EVERY artifact", all_runs(empty_sources))
expect("consistent but wrong training target in EVERY artifact", all_runs(wrong_target))
expect("inconsistent sample-target expansion in EVERY artifact", all_runs(bad_expansion))
expect("non-finite single-view prediction", lambda r: nan_pred(r["mse"][43]))
try:
    M.load(d_, [42, 43, 44]); raise SystemExit("NOT rejected: missing seed")
except SystemExit as e:
    if str(e).startswith("NOT"):
        raise
    print(f"    rejected (missing seed): {e}")
print("ALL COMPARE_LBL_MSE FIXTURE CHECKS PASSED")
