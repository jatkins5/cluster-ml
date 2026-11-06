#!/usr/bin/env python3
"""Fixture checks for the catalogue-TSC label experiment -- no training.

1. load_catalogue_tsc on constructed files: NaN -> 4 Gyr, > 4 -> 4, values
   kept below the cap, both fields; rejects duplicate file keys, duplicate
   requested parents, a missing parent, a negative value and a snap_now
   mismatch.
2. the real join on xray_fix_s9991.h5 rows (counts as in Chuiyang's files).
3. compare_labels.check: a valid synthetic matrix passes; rejects a training
   target that is not the declared label, truths differing between runs, an
   undeclared config change, a wrong label name, a missing TSC-file hash, an
   unequal capped/raw pair; the production seed contract.
"""
import copy
import json
import os
import tempfile

import h5py
import numpy as np

import compare_labels as L
from train_cnn_mock import TSC_CAP, load_catalogue_tsc


def expect(exc, f, *a, what=""):
    try:
        f(*a)
    except exc as e:
        print(f"    rejected ({what}): {e}")
        return
    raise SystemExit(f"NOT rejected: {what}")


tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "."))


def write(d, sn, keys, a, b, snap_now=None):
    os.makedirs(d, exist_ok=True)
    with h5py.File(os.path.join(d, f"TSC_eachhalo_snap{sn}.hdf5"), "w") as f:
        f["FOF_Halo_ID_At99"] = np.asarray(keys)
        f["tsc_gyr"], f["tsc_gyr_from_Tcoll"] = np.asarray(a, float), np.asarray(b, float)
        f.attrs["snap_now"] = sn if snap_now is None else snap_now


print("== 1. constructed join")
good = os.path.join(tmp, "good")
write(good, 99, [10, 11, 12, 13], [np.nan, 5.0, 1.5, 0.0], [np.nan, 5.1, 1.55, 0.02])
write(good, 91, [10, 11, 12, 13], [2.0, np.nan, 0.3, 4.0], [2.05, np.nan, 0.31, 4.0])
snap = np.array([99, 99, 99, 99, 91, 91, 91])
grp = np.array([10, 11, 12, 13, 10, 11, 12])
c = load_catalogue_tsc(good, snap, grp)
assert np.allclose(c["tsc_gyr"][0], [4, 4, 1.5, 0.0, 2.0, 4, 0.3])
assert np.allclose(c["tsc_gyr_from_Tcoll"][0], [4, 4, 1.55, 0.02, 2.05, 4, 0.31])
assert np.isnan(c["tsc_gyr"][1][[0, 5]]).all() and c["tsc_gyr"][1][1] == 5.0
print(f"    capped: {c['tsc_gyr_from_Tcoll'][0].tolist()} (NaN and > {TSC_CAP} -> {TSC_CAP}; raw kept)")
bad = os.path.join(tmp, "dup"); write(bad, 99, [10, 10, 12, 13], [1, 1, 1, 1], [1, 1, 1, 1]); write(bad, 91, [10, 11, 12, 13], [1] * 4, [1] * 4)
expect(ValueError, load_catalogue_tsc, bad, snap, grp, what="duplicate file keys")
expect(ValueError, load_catalogue_tsc, good, snap, np.array([10, 10, 12, 13, 10, 11, 12]),
       what="duplicate requested parents")
expect(ValueError, load_catalogue_tsc, good, snap, np.array([10, 11, 12, 99, 10, 11, 12]),
       what="missing parent")
bad = os.path.join(tmp, "neg"); write(bad, 99, [10, 11, 12, 13], [1, -1, 1, 1], [1, 1, 1, 1]); write(bad, 91, [10, 11, 12, 13], [1] * 4, [1] * 4)
expect(ValueError, load_catalogue_tsc, bad, snap, grp, what="negative value")
bad = os.path.join(tmp, "snapnow"); write(bad, 99, [10, 11, 12, 13], [1] * 4, [1] * 4, snap_now=91); write(bad, 91, [10, 11, 12, 13], [1] * 4, [1] * 4)
expect(ValueError, load_catalogue_tsc, bad, snap, grp, what="snap_now mismatch")

print("== 2. real join (xray_fix_s9991.h5 rows)")
with h5py.File("xray_fix_s9991.h5") as f:
    xs, xg = f["mock/snap"][:], f["mock/group_id"][:]
r = load_catalogue_tsc("/oscar/data/idellant/Chuiyang/TSC_new", xs, xg)
for sn in (99, 91):
    raw = r["tsc_gyr_from_Tcoll"][1][xs == sn]
    print(f"    snapshot {sn}: {len(raw)} rows; NaN {int(np.isnan(raw).sum())}, > 4 Gyr "
          f"{int((raw > 4).sum())}, capped total {int(((~np.isfinite(raw)) | (raw > 4)).sum())}")
assert int(np.isnan(r["tsc_gyr_from_Tcoll"][1][xs == 99]).sum()) == 6
assert int(np.isnan(r["tsc_gyr_from_Tcoll"][1][xs == 91]).sum()) == 15

print("== 3. compare_labels.check on a synthetic matrix")
MAN = "f" * 64
N, R, P = 6, 3, 3
pseudo = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 0.3])
tsc_raw = np.array([0.0, 0.6, np.nan, 2.2, 5.0, 0.25])
tco_raw = tsc_raw + np.where(np.isfinite(tsc_raw), 0.01, 0)
capf = lambda v: np.where(np.isfinite(v), np.minimum(v, 4.0), 4.0)
target = {"pseudo": pseudo, "tsc": capf(tsc_raw), "tsc_tcoll": capf(tco_raw)}


def make(m, lab, s):
    cfg = dict(L.REQUIRED_CFG, epochs=60, dataset=L.MODALITIES[m], seed=s, label=lab,
               out_prefix=f"x{s}", tsc_dir="/oscar/data/idellant/Chuiyang/TSC_new")
    prov = dict(head="h", status="", sha256={"train_cnn_mock.py": "a"}, config=cfg, manifest_sha256=MAN,
                inputs_sha256={"dataset": "d", "catalog": "c", "tsc_snap99": "t9", "tsc_snap91": "t1"})
    y = target[lab]
    rg = np.random.default_rng(s)
    return dict(sample_halo=np.repeat(np.arange(N), R), sample_group=np.repeat(np.arange(N), R),
                sample_snap=np.repeat([99, 99, 99, 91, 91, 91], R), sample_realization=np.tile(np.arange(R), N),
                y_sample=np.repeat(y, R).astype(np.float32), tsc=y.astype(np.float32),
                fold_of=np.repeat(np.arange(N) % 5, R), sel_groups_flat=np.zeros(0), sel_groups_fold=np.zeros(0),
                truth_pseudo=pseudo, truth_tsc_capped=capf(tsc_raw), truth_tsc_raw=tsc_raw,
                truth_tcoll_capped=capf(tco_raw), truth_tcoll_raw=tco_raw, label_name=np.array(lab),
                oof_sample=rg.uniform(0, 4, N * R), oof_single=rg.uniform(0, 4, (N * R, P)),
                provenance=np.array(json.dumps(prov)), provenance_changed_during_run=np.array(False))


for m in L.MODALITIES:
    runs = {lab: {s: make(m, lab, s) for s in (42, 43)} for lab in L.LABELS}
    L.check(runs, m, MAN, 60)
print("    valid synthetic matrix accepted (both modalities)")
runs = {lab: {s: make("x", lab, s) for s in (42, 43)} for lab in L.LABELS}


def bad_case(what, f, lab="tsc_tcoll", seed=43):
    r_ = copy.deepcopy(runs)
    f(r_[lab][seed])
    expect(SystemExit, L.check, r_, "x", MAN, 60, what=what)


def cfg(k, v):
    def f(d):
        p = json.loads(str(d["provenance"])); p["config"][k] = v; d["provenance"] = np.array(json.dumps(p))
    return f


def no_hash(d):
    p = json.loads(str(d["provenance"])); p["inputs_sha256"].pop("tsc_snap91"); d["provenance"] = np.array(json.dumps(p))


bad_case("training target is not the declared label",
         lambda d: (d.__setitem__("tsc", pseudo.astype(np.float32)),
                    d.__setitem__("y_sample", np.repeat(pseudo, R).astype(np.float32))))
bad_case("truth differs between runs", lambda d: d.__setitem__("truth_pseudo", pseudo + 0.1))
bad_case("undeclared config change", cfg("huber_delta", 1.0))
bad_case("wrong label name", lambda d: d.__setitem__("label_name", np.array("tsc")))
bad_case("missing TSC-file hash", no_hash)
bad_case("capped != capped raw", lambda d: d.__setitem__("truth_tcoll_capped",
                                                         np.minimum(d["truth_tcoll_capped"], 3.0)))
bad_case("wrong dataset for the modality", cfg("dataset", "injected_syn_phys_b3000_s9991.h5"))
for txt in ("42,43", "42,42,43,44,45"):
    expect(SystemExit, L.parse_seeds, txt, "production", what=f"production seeds {txt!r}")
print("ALL LABEL FIXTURE CHECKS PASSED")
