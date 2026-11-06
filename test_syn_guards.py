#!/usr/bin/env python3
"""Fixtures for the compare_syn.py guards (review R27) -- no training, no
scientific analysis: synthetic seed sets fed to the importable validators.

Rejected: changed inner-fold assignments with unchanged flat halo ids;
missing required matched metadata (inner sets, mass inputs, provenance);
altered protocol, defining source or recorded seed; a joint X-ray input that
is not the control's dataset; a clean arm whose cohort differs consistently
across all seeds; malformed or non-finite single-view / real-target arrays.
Passed: a valid matched mass contrast, a valid matched joint-vs-X-ray
contrast, and a legacy descriptive contrast without provenance (reported as
unavailable).
"""
import copy
import json

import numpy as np

import compare_syn as C

SEEDS = C.SEEDS
N, R, P, F = 10, 2, 3, 5


def expect(f, *a, what="", **k):
    try:
        f(*a, **k)
    except SystemExit as e:
        print(f"    rejected ({what}): {e}")
        return
    raise SystemExit(f"NOT rejected: {what}")


def run(seed, cfg_extra=None, mass=None, prov=True, sha="a", inputs=None, label_shift=0.0):
    rg = np.random.default_rng(seed)
    grp = np.arange(N)
    tsc = np.linspace(0.1, 4.0, N) + label_shift
    d = dict(group_id=grp, halo_id=grp + 1000, snap=np.r_[np.full(5, 99), np.full(5, 91)], tsc=tsc,
             y_sample=np.repeat(tsc, R), sample_halo=np.repeat(grp + 1000, R),
             sample_realization=np.tile(np.arange(R), N), sample_group=np.repeat(grp, R),
             sample_snap=np.repeat(np.r_[np.full(5, 99), np.full(5, 91)], R),
             oof_cluster=rg.uniform(0, 4, N), oof_sample=rg.uniform(0, 4, N * R),
             oof_single=rg.uniform(0, 4, (N * R, P)), fold_of=np.repeat(grp % F, R),
             sel_groups_flat=np.array([1, 2, 3, 4, 0]), sel_groups_fold=np.arange(F),
             sim_logm_input=(mass if mass is not None else np.zeros(0)), sim_logm=np.linspace(14, 15, N),
             obs_name=np.array(["A85", "A119"]), obs_z=np.array([0.05, 0.04]),
             obs_pred=rg.uniform(0, 4, 2), provenance_changed_during_run=np.array(False))
    if prov:
        cfg = dict(seed=seed, out_prefix=f"x{seed}", epochs=60, dataset="injected_syn_phys_b3000_s9991.h5",
                   xray_dataset=None, no_image=False, mass=mass is not None)
        cfg.update(cfg_extra or {})
        d["provenance"] = np.array(json.dumps(dict(sha256={"train_cnn_mock.py": sha}, config=cfg,
                                                   inputs_sha256=inputs or {"dataset": "D"})))
    return d


def arm(**k):
    return [run(s, **k) for s in SEEDS]


MASS = np.linspace(14.1, 15.1, N)
print("== matched mass contrast")
mo, im = arm(mass=MASS, cfg_extra={"no_image": True}), arm(mass=MASS)
for x in (mo, im):
    C.validate_set(x, "fixture", R=R, P=P)
st = C.split_status(mo, im)
C.check_matched(mo, im, st, "mass", "mass pair")
print(f"    valid: {C.check_contract(mo, im, ('no_image',), 'mass pair', True)}; {C.fmt(st)}")


def inner_reassign(ds):
    for d in ds:
        d["sel_groups_fold"] = np.array([1, 0, 2, 3, 4])      # same flat ids, swapped folds
    return ds


im2 = inner_reassign(copy.deepcopy(im))
expect(C.check_matched, mo, im2, C.split_status(mo, im2), "mass", "x", what="inner folds reassigned, flat ids unchanged")
im3 = copy.deepcopy(im)
for d in im3:
    del d["sel_groups_fold"]
expect(C.check_matched, mo, im3, C.split_status(mo, im3), "mass", "x", what="inner membership unavailable")
im4 = arm(mass=MASS + 0.1)
expect(C.check_matched, mo, im4, C.split_status(mo, im4), "mass", "x", what="different noisy mass inputs")
im5 = copy.deepcopy(im)
im5[2]["sim_logm_input"] = MASS + 0.2
mo5 = copy.deepcopy(mo); mo5[2]["sim_logm_input"] = MASS + 0.2
expect(C.check_matched, mo5, im5, C.split_status(mo5, im5), "mass", "x", what="mass inputs differ across seeds of an arm")
expect(C.check_contract, mo, arm(mass=MASS, cfg_extra={"epochs": 30}), ("no_image",), "x", True,
       what="undeclared protocol change (epochs)")
expect(C.check_contract, mo, arm(mass=MASS, sha="b"), ("no_image",), "x", True, what="different defining source")
expect(C.check_contract, mo, [run(s, mass=MASS, prov=False) for s in SEEDS], ("no_image",), "x", True,
       what="provenance missing in a matched contrast")
bad_seed = copy.deepcopy(im)
p_ = json.loads(str(bad_seed[1]["provenance"])); p_["config"]["seed"] = 99
bad_seed[1]["provenance"] = np.array(json.dumps(p_))
expect(C.validate_set, bad_seed, "fixture", R=R, P=P, what="recorded seed differs from the file's seed")

print("== matched joint vs X-ray control")
ctrl = arm(cfg_extra={"dataset": "xray_fix_b3000_s9991.h5"}, inputs={"dataset": "XH"})
joint = arm(cfg_extra={"xray_dataset": "xray_fix_b3000_s9991.h5"}, inputs={"dataset": "D", "xray_dataset": "XH"})
st = C.split_status(ctrl, joint)
C.check_matched(ctrl, joint, st, "plain", "joint pair")
print(f"    valid: {C.check_contract(ctrl, joint, ('dataset', 'xray_dataset'), 'joint pair', True, 'b_is_joint')}; {C.fmt(st)}")
jbad = arm(cfg_extra={"xray_dataset": "xray_fix_b3000_s9991.h5"}, inputs={"dataset": "D", "xray_dataset": "OTHER"})
expect(C.check_contract, ctrl, jbad, ("dataset", "xray_dataset"), "x", True, "b_is_joint",
       what="joint X-ray input hash != control dataset hash")

print("== legacy descriptive contrast (no provenance)")
old = [run(s, prov=False) for s in SEEDS]
for d in old:
    del d["sel_groups_fold"], d["sel_groups_flat"]
st = C.split_status(old, joint)
print(f"    passes descriptively: {C.check_contract(old, joint, ('dataset', 'xray_dataset'), 'old vs new', False)}; "
      f"{C.fmt(st)}")
expect(C.check_matched, old, joint, st, "plain", "x", what="legacy archive labelled matched")

print("== clean cohort and per-artifact validation")


def clean(seed, shift=0.0):
    lab = np.linspace(0.1, 4, N) + shift
    return dict(halo_id=np.arange(N), label=lab, oof_pred=lab + 0.1, group_id=np.arange(N), snap=np.full(N, 99))


sets = {"A": [clean(s) for s in SEEDS], "B": [clean(s, 0.05) for s in SEEDS]}
loader = lambda pat, seeds=SEEDS: (C.validate_set(sets[pat], pat, seeds, R=R, P=P), sets[pat])[1]
expect(C.clean_pair, "A", "B", loader=loader, what="clean arm cohort differs consistently across all seeds")
sets["B"] = [clean(s) for s in SEEDS]
a_, b_ = C.clean_pair("A", "B", loader=loader)
print(f"    matching clean cohorts pass ({len(a_)} seeds each): {C.CLEAN_STATUS[('A', 'B')]}")
sets["L"] = [{k: v for k, v in clean(s).items() if k not in ("group_id", "snap")} for s in SEEDS]
C.clean_pair("L", "A", loader=loader)
print(f"    legacy clean archive without group_id/snap: {C.CLEAN_STATUS[('L', 'A')]}")
sets["G"] = [dict(clean(s), group_id=np.arange(N)[::-1].copy()) for s in SEEDS]
expect(C.clean_pair, "G", "A", loader=loader, what="clean cohorts with different group_id")
bad = arm()
bad[0]["oof_single"] = bad[0]["oof_single"][:, :1]
expect(C.validate_set, bad, "fixture", R=R, P=P, what="single-view array with one projection in one seed")
bad = arm()
bad[3]["oof_single"] = bad[3]["oof_single"].copy(); bad[3]["oof_single"][0, 0] = np.nan
expect(C.validate_set, bad, "fixture", R=R, P=P, what="non-finite single-view prediction")
bad = arm()
bad[2]["obs_pred"] = np.array([1.0, np.inf])
expect(C.validate_set, bad, "fixture", R=R, P=P, what="non-finite real-target prediction")
bad = arm()
for d in bad:
    d["obs_pred"] = np.array([1.0])
expect(C.validate_set, bad, "fixture", R=R, P=P, what="real-target prediction length != obs_name")
bad = arm()
for d in bad:
    d["sample_snap"] = d["sample_snap"][::-1].copy()
expect(C.validate_set, bad, "fixture", R=R, P=P, what="sample_snap inconsistent with the row expansion")
print("== R28: input hashes for every run, declared dimensions")
jb = arm(cfg_extra={"xray_dataset": "xray_fix_b3000_s9991.h5"}, inputs={"dataset": "D", "xray_dataset": "XH"})
p_ = json.loads(str(jb[1]["provenance"])); p_["inputs_sha256"]["xray_dataset"] = "OTHER"
jb[1]["provenance"] = np.array(json.dumps(p_))
expect(C.check_contract, ctrl, jb, ("dataset", "xray_dataset"), "x", True, "b_is_joint",
       what="joint X-ray hash differs in a non-first seed")
im_d = arm(mass=MASS, inputs={"dataset": "D2"})
expect(C.check_contract, mo, im_d, ("no_image",), "x", True, what="mass arms with different dataset hashes")
nf = arm()
del nf[2]["provenance_changed_during_run"]
expect(C.validate_set, nf, "fixture", R=R, P=P, what="changed-during-run flag missing")
one = arm()
for d in one:
    d["oof_single"] = d["oof_single"][:, :1]
expect(C.validate_set, one, "fixture", R=R, P=P, what="one projection in EVERY seed (declared P=3)")
expect(C.validate_set, arm(), "fixture", R=3, P=P, what="realization count differs from the declared R")
short = np.linspace(14.1, 15.1, N - 1)
expect(C.validate_set, arm(mass=short), "fixture", R=R, P=P, what="wrong-length mass vector")
ui = arm()
for d in ui:
    d["sel_groups_fold"] = np.arange(F - 1)
expect(C.split_status, ui, arm(), what="unequal inner-array lengths")
sh = arm()
for d in sh:
    d["sample_halo"] = d["sample_halo"][::-1].copy()
expect(C.validate_set, sh, "fixture", R=R, P=P, what="sample_halo inconsistent with the row expansion")
cl = [{"halo_id": np.arange(N), "label": np.full((N, 1), 1.0), "oof_pred": np.full((N, 1), 1.0)} for s in SEEDS]
expect(C.validate_set, cl, "fixture", R=R, P=P, what="2-D clean labels")
print("ALL COMPARE_SYN GUARD CHECKS PASSED")
