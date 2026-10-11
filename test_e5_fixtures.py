#!/usr/bin/env python3
"""E5 fixture checks (review round 16) -- no training. Run via smoke_e5_unit.sh.

  1. legacy default: run_fold's default loss is huber; make_criterion maps
     huber/mse/class7 to HuberLoss(delta)/MSELoss/CrossEntropyLoss
  2. label join on a constructed file: no crossing (C6), first-tau (C0),
     exact edges 0.25 and 0.5 (C1, C2: right-closed), a crossing first at
     tau = 4.0 (C5, not C6); rejects duplicate/missing keys, group and
     pseudo-TSC mismatches, non-finite curves and a bad tau grid; then the
     real join (xray_fix_s9991.h5 x dataset_nh4_144_s9991_128.h5)
  3. probabilities normalise; decoded points lie in [0.1, 4.0]; with common
     representatives, decoding averaged probabilities == averaging decoded
  4. representatives/climatology use fitting rows only, count each
     (snapshot, halo) row once, fix C0 = 0.1 and C6 = 4.0, fail on an empty
     class
  5. compare_e5.check accepts a valid five-fold, four-arm synthetic matrix
     and rejects per-run disagreements (rows, undeclared config, wrong
     loss/contrastive switch, probability normalisation/shape/decoder,
     representatives across seeds, label-curve hash, pretraining partition
     and update counts) AND malformed metadata applied to every run alike
     (class -1, two-fold statistics, climatology, representative outside
     its class, nocross, checkpoint record); production seed contract
     (partial, duplicate, empty, wrong) and missing/duplicate seeds in load
  6. a zero true-class probability gives an infinite, counted log score
     with an unavailable interval, while RPS/Brier/coverage are still emitted
  7. E4: the parent-interleaved sampler visits every unit once, never
     repeats a parent in a batch, merges a single trailing unit, rejects a
     layer without negatives; NT-Xent prefers the designated positives and
     masks self-similarity; one CPU pretraining epoch runs; a frozen encoder
     keeps all parameters AND BatchNorm buffers through a head step
"""
import copy
import inspect
import json
import os
import tempfile

import h5py
import numpy as np
import torch.nn as nn

import compare_e5 as C
from train_cnn_mock import (CLASS_EDGES, N_CLASS, class_reps, classify, load_label_classes,
                            make_criterion, pseudo_tsc_from_curve, run_fold, softmax_np)

tau = np.round(np.arange(1, 41) * 0.1, 10)


def expect(exc, f, *a, what=""):
    try:
        f(*a)
    except exc as e:
        print(f"    rejected ({what}): {e}")
        return
    raise SystemExit(f"NOT rejected: {what}")


print("== 1. legacy default and loss factory")
assert inspect.signature(run_fold).parameters["loss"].default == "huber"
c = make_criterion("huber", 0.5)
assert isinstance(c, nn.HuberLoss) and c.delta == 0.5
assert isinstance(make_criterion("mse", 0.5), nn.MSELoss)
assert isinstance(make_criterion("class7", 0.5), nn.CrossEntropyLoss)
print("    run_fold default loss=huber; huber(0.5)/mse/class7 criteria as declared")

print("== 2. label join: constructed curves")


def curve_cross(t_cross_idx, s0=0.4, s1=0.6):
    cv = np.full(40, 0.2)
    cv[t_cross_idx - 1] = s0
    cv[t_cross_idx:] = s1
    return cv


curves = {
    "nocross": np.full(40, 0.3),
    "first": np.full(40, 0.7),
    "edge025": curve_cross(2),                                   # 0.2 + 0.1 * 0.1/0.2 = 0.25
    "edge05": np.r_[np.zeros(4), 0.5, np.full(35, 0.6)],          # 0.4 + 0.1 * 0.5/0.5 = 0.5
    "at4": np.r_[np.full(39, 0.3), 0.5],                          # first >= 0.5 at tau = 4.0
}
expect_cls = {"nocross": 6, "first": 0, "edge025": 1, "edge05": 2, "at4": 5}
for k, cv in curves.items():
    y = pseudo_tsc_from_curve(cv, tau)
    got = classify(cv, y)
    assert got == expect_cls[k], (k, y, got)
    print(f"    {k:<8} y = {y:.4f} -> C{got}")
assert np.isclose(pseudo_tsc_from_curve(curves["edge025"], tau), 0.25)
assert np.isclose(pseudo_tsc_from_curve(curves["edge05"], tau), 0.5)
assert np.isclose(pseudo_tsc_from_curve(curves["at4"], tau), 4.0)

tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "."))


def write_fixture(path, snap, halo, group, s_all, s_pre, ps, tau_=tau):
    with h5py.File(path, "w") as f:
        f["labels/tau_gyr"] = tau_
        f["labels/label_score_all"] = s_all
        f["labels/label_score_pre"] = s_pre
        f["labels/pseudo_tsc"] = ps
        f["meta/snap"], f["meta/halo_id"], f["meta/group_id"] = snap, halo, group


names = list(curves)
n = len(names)
snap = np.array([99, 91, 99, 91, 99])
halo = np.arange(100, 100 + n)
group = np.arange(10, 10 + n)
cv = np.stack([curves[k] for k in names])
junk = np.full_like(cv, 0.9)                     # the curve NOT selected for that snapshot
s_all = np.where((snap == 99)[:, None], cv, junk)
s_pre = np.where((snap == 99)[:, None], junk, cv)
ps = np.array([pseudo_tsc_from_curve(c_, tau) for c_ in cv], dtype=np.float32)
good = os.path.join(tmp, "good.h5")
write_fixture(good, snap, halo, group, s_all, s_pre, ps)
cls, noc = load_label_classes(good, snap, halo, group, ps)
assert cls.tolist() == [expect_cls[k] for k in names] and noc.tolist() == [k == "nocross" for k in names]
print(f"    join selects 'all' at snapshot 99 and 'pre' otherwise: classes {cls.tolist()}")

bad = os.path.join(tmp, "dup.h5")
write_fixture(bad, np.r_[snap[:-1], snap[0]], np.r_[halo[:-1], halo[0]], group, s_all, s_pre, ps)
expect(ValueError, load_label_classes, bad, snap, halo, group, ps, what="duplicate key")
expect(ValueError, load_label_classes, good, snap, np.r_[halo[:-1], 999], group, ps,
       what="missing key")
expect(ValueError, load_label_classes, good, snap, halo, np.r_[group[:-1], 77], ps,
       what="group mismatch")
expect(ValueError, load_label_classes, good, snap, halo, group, ps + np.float32(0.01),
       what="pseudo-TSC mismatch")
nan = s_all.copy(); nan[0, 5] = np.nan
bad = os.path.join(tmp, "nan.h5"); write_fixture(bad, snap, halo, group, nan, s_pre, ps)
expect(ValueError, load_label_classes, bad, snap, halo, group, ps, what="non-finite curve")
bad = os.path.join(tmp, "tau.h5"); write_fixture(bad, snap, halo, group, s_all, s_pre, ps,
                                                 tau_=tau[::-1].copy())
expect(ValueError, load_label_classes, bad, snap, halo, group, ps, what="bad tau grid")

print("== 2b. real join")
with h5py.File("xray_fix_s9991.h5") as f:
    xs, xh, xg, xp = (f["mock/snap"][:], f["mock/halo_id"][:], f["mock/group_id"][:],
                      f["mock/pseudo_tsc"][:])
cls_r, noc_r = load_label_classes("dataset_nh4_144_s9991_128.h5", xs, xh, xg, xp)
cnt = np.bincount(cls_r, minlength=N_CLASS)
assert noc_r.sum() == 81 and cnt[0] == 71 and (cls_r == 6).sum() == (xp == 4.0).sum()
print(f"    {len(xs)} rows joined; classes C0..C6 {cnt.tolist()}; "
      f"snapshot 99: {np.bincount(cls_r[xs == 99], minlength=N_CLASS).tolist()}")

print("== 3. probabilities and decoder")
rng = np.random.default_rng(0)
logits = rng.normal(size=(5, 50, N_CLASS)) * 3
P = softmax_np(logits)
assert np.allclose(P.sum(-1), 1) and (P >= 0).all()
reps = np.array([0.1, 0.18, 0.37, 0.74, 1.45, 2.8, 4.0])
dec = P @ reps
assert (dec >= 0.1 - 1e-12).all() and (dec <= 4.0 + 1e-12).all()
assert np.allclose(P.mean(0) @ reps, dec.mean(0))
print("    rows sum to 1; points in [0.1, 4.0]; decode(mean p) == mean decode(p) with common reps")

print("== 4. representatives and climatology")
y_row = np.array([0.1, 0.2, 0.3, 0.7, 1.5, 3.0, 4.0, 0.1, 0.22, 0.45, 0.9, 1.9, 2.5, 4.0])
c_row = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6])
assert all(c_row[i] == (6 if v == 4.0 else 0 if v == 0.1 else classify(curve_cross(5), v))
           for i, v in enumerate(y_row))
fit = np.arange(10)
r1, n1, cl1 = class_reps(y_row, c_row, fit)
y2 = y_row.copy(); y2[10:] += 0.05                          # outer rows changed
r2, n2, cl2 = class_reps(y2, c_row, fit)
assert np.array_equal(r1, r2) and np.array_equal(n1, n2)
R = 3
fit_samples = np.repeat(fit, R) * R + np.tile(np.arange(R), len(fit))
r3, n3, _ = class_reps(y_row, c_row, np.unique(fit_samples // R))
assert np.array_equal(r1, r3) and np.array_equal(n1, n3)
assert r1[0] == 0.1 and r1[6] == 4.0 and np.isclose(cl1.sum(), 1)
expect(ValueError, class_reps, y_row, c_row, np.arange(3), what="empty fitting class")
print(f"    fit-only, once per row (R=3 replication identical); reps {np.round(r1, 3).tolist()}")

print("== 5. compare_e5.check on a synthetic five-fold matrix (four arms)")
MAN = "f" * 64
N, Rr, Pp, F = 10, 3, 3, 5
row_fold = np.arange(N) % F
fold_of = np.repeat(row_fold, Rr).astype(np.int16)
inner_of = {f: (f + 1) % F for f in range(F)}          # one inner parent per fold
sel_flat = np.array([inner_of[f] for f in range(F)]).astype(float)
sel_fold = np.arange(F).astype(float)
yr, cr = y_row[:N], c_row[:N]
reps_f = np.tile(reps, (F, 1)) + np.linspace(0, 0.004, F)[:, None]
reps_f[:, 0], reps_f[:, 6] = 0.1, 4.0
counts_f = np.tile(np.arange(1, 8), (F, 1)).astype(np.int64)
clim_f = counts_f / counts_f.sum(1, keepdims=True)


def fit_parents(f):
    return sorted(set(np.where(row_fold != f)[0].tolist()) - {inner_of[f]})


def make_run(arm, seed):
    _, loss, cl = C.ARMS[arm]
    rg = np.random.default_rng(seed)
    cfg = dict(C.REQUIRED_CFG, seed=seed, out_prefix=f"x{seed}", loss=loss, epochs=60,
               cl_epochs=300, contrastive_pretrain=cl,
               label_curves="dataset_nh4_144_s9991_128.h5", fold_manifest="folds_e0.npz",
               view_cols="richness,gap12,off_bcg_x,off_bcg_gal,asym")
    prov = dict(head="h", status="", sha256={"train_cnn_mock.py": "a"}, config=cfg,
                manifest_sha256=MAN,
                inputs_sha256={"dataset": "d", "catalog": "c", "label_curves": "l"})
    d = dict(sample_halo=np.repeat(np.arange(N), Rr), sample_group=np.repeat(np.arange(N), Rr),
             sample_snap=np.repeat(np.r_[np.full(N // 2, 99), np.full(N - N // 2, 91)], Rr),
             sample_realization=np.tile(np.arange(Rr), N), y_sample=np.repeat(yr, Rr),
             tsc=yr, fold_of=fold_of, sel_groups_flat=sel_flat, sel_groups_fold=sel_fold,
             class_of=cr.astype(np.int64), nocross=cr == 6, class_edges=CLASS_EDGES,
             fold_class_reps=reps_f, fold_class_counts=counts_f, fold_climatology=clim_f,
             fold_best_epoch=np.array([10, 20, 30, 40, 50]), fold_sel_score=np.full(F, 0.4),
             provenance=np.array(json.dumps(prov)),
             provenance_changed_during_run=np.array(False), loss=np.array(loss),
             contrastive_pretrain=np.array(cl))
    if loss == "class7":
        pr = softmax_np(rg.normal(size=(N * Rr, N_CLASS)))
        ps_ = softmax_np(rg.normal(size=(N * Rr, Pp, N_CLASS)))
        d["oof_prob"], d["oof_single_prob"] = pr, ps_
        d["oof_sample"] = np.einsum("ik,ik->i", pr, reps_f[fold_of]).astype(np.float32)
        d["oof_single"] = np.einsum("ipk,ik->ip", ps_, reps_f[fold_of]).astype(np.float32)
    else:
        d["oof_prob"], d["oof_single_prob"] = np.zeros((0, 7)), np.zeros((0, Pp, 7))
        d["oof_sample"] = rg.uniform(0.1, 4, N * Rr).astype(np.float32)
        d["oof_single"] = rg.uniform(0.1, 4, (N * Rr, Pp)).astype(np.float32)
    if cl:
        cg = [fit_parents(f) for f in range(F)]
        d.update(cl_config=np.array(json.dumps(dict(epochs=300))),
                 fold_cl_updates=np.full(F, 36 * 300), fold_cl_pairs=np.full(F, 4000),
                 cl_loss_curve_fold=np.ones((F, 300)),
                 cl_groups_flat=np.concatenate([np.array(g_, float) for g_ in cg]),
                 cl_groups_fold=np.concatenate([np.full(len(g_), f, float)
                                                for f, g_ in enumerate(cg)]))
    else:
        d.update(cl_groups_flat=np.zeros(0), cl_groups_fold=np.zeros(0))
    return d


runs = {a: {s: make_run(a, s) for s in (42, 43)} for a in C.ARMS}
C.check(runs, MAN)
print("    valid synthetic matrix accepted (5 folds, 4 arms, 2 seeds)")


def bad_case(what, f, arm="d", seed=43, all_runs=False):
    r = copy.deepcopy(runs)
    for a in r:
        for s in r[a]:
            if all_runs or (a == arm and s == seed):
                f(r[a][s])
    expect(SystemExit, C.check, r, MAN, what=what + (" [ALL runs]" if all_runs else ""))


def cfg_edit(k, v):
    def f(d):
        p = json.loads(str(d["provenance"])); p["config"][k] = v
        d["provenance"] = np.array(json.dumps(p))
    return f


def drop_hash(d):
    p = json.loads(str(d["provenance"])); p["inputs_sha256"].pop("label_curves")
    d["provenance"] = np.array(json.dumps(p))


def neg_class(d):
    c_ = d["class_of"].copy(); c_[0] = -1; d["class_of"] = c_


def two_folds(d):
    for k in ("fold_class_reps", "fold_class_counts", "fold_climatology"):
        d[k] = d[k][:2]


def cl_inner(d):
    if "cl_groups_flat" in d and d["cl_groups_flat"].size:
        gf, gi = d["cl_groups_flat"], d["cl_groups_fold"]
        d["cl_groups_flat"] = np.concatenate([gf, [float(inner_of[0])]])
        d["cl_groups_fold"] = np.concatenate([gi, [0.0]])


bad_case("mismatched rows", lambda d: d.__setitem__("sample_halo", d["sample_halo"][::-1].copy()))
bad_case("undeclared config change", cfg_edit("huber_delta", 1.0), arm="m")
bad_case("wrong loss for the arm", cfg_edit("loss", "mse"))
bad_case("contrastive switch on the wrong arm", cfg_edit("contrastive_pretrain", True), arm="h")
bad_case("unnormalised probabilities",
         lambda d: d.__setitem__("oof_single_prob", d["oof_single_prob"] * 1.1))
bad_case("probability shape", lambda d: d.__setitem__("oof_prob", d["oof_prob"][:, :6]))
bad_case("points disagree with decoder",
         lambda d: d.__setitem__("oof_single", d["oof_single"] + 0.1))
bad_case("representatives differ across seeds",
         lambda d: d.__setitem__("fold_class_reps", d["fold_class_reps"] + 0.001))
bad_case("label-curve hash missing", drop_hash, arm="h")
bad_case("class index -1", neg_class, all_runs=True)
bad_case("two-fold statistics under a five-fold protocol", two_folds, all_runs=True)
bad_case("climatology != normalised counts",
         lambda d: d.__setitem__("fold_climatology", np.full((F, 7), 1 / 7)), all_runs=True)
def rep_outside(d):
    r_ = d["fold_class_reps"].copy(); r_[:, 3] = 1.5     # C3 is (0.5, 1.0]
    d["fold_class_reps"] = r_


bad_case("representative outside its class", rep_outside, all_runs=True)
bad_case("nocross inconsistent with C6",
         lambda d: d.__setitem__("nocross", np.zeros_like(d["nocross"])), all_runs=True)
bad_case("checkpoint record missing", lambda d: d.__setitem__("fold_best_epoch", np.zeros(5, int)),
         all_runs=True)
bad_case("pretraining includes an inner-selection parent", cl_inner, arm="c", seed=42)
bad_case("pretraining update counts differ across seeds",
         lambda d: d.__setitem__("fold_cl_updates", d["fold_cl_updates"] + 1), arm="c")
for txt in ("42,43", "42,42,43,44,45", "", "42,43,44,45,47"):
    expect(SystemExit, C.parse_seeds, txt, "production", what=f"production seeds {txt!r}")
expect(SystemExit, C.parse_seeds, "42,42", "smoke", what="duplicate smoke seeds")
assert C.parse_seeds("42,43", "smoke") == [42, 43]
d_ = os.path.join(tmp, "preds"); os.makedirs(d_)
for a, spec in C.ARMS.items():
    for s in (42, 43):
        if not (a == "m" and s == 43):
            np.savez(os.path.join(d_, spec[0].format(s)), **runs[a][s])
expect(SystemExit, C.load, d_, [42, 43], what="missing seed")
expect(SystemExit, C.load, d_, [42, 42], what="duplicate seeds in load")

print("== 6. zero true-class probability")
import contextlib
import io
p0 = np.array([[0.0, 1.0, 0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0, 0, 0]])
lg, rp, br = C.prob_scores(p0, np.array([0, 1]))
assert np.isinf(lg[0]) and np.isfinite(lg[1]) and np.isfinite(rp).all() and np.isfinite(br).all()
rz = copy.deepcopy(runs)
for s in rz["d"]:
    dd = rz["d"][s]
    oh = np.zeros_like(dd["oof_single_prob"]); oh[..., 6] = 1.0       # all mass on C6
    dd["oof_single_prob"] = oh
    dd["oof_single"] = np.einsum("ipk,ik->ip", oh, reps_f[fold_of]).astype(np.float32)
_, g_, _ = C.single_obs(rz["h"][42], rz["h"])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    C.prob_block(rz["h"][42], rz, g_, 50, "")
out = buf.getvalue()
assert "interval unavailable" in out and "RPS (/6)" in out and "nan" not in out.lower()
print("    zero true-class probability: log score infinite and counted, interval "
      "unavailable; RPS/Brier/coverage still emitted")

print("== 7. E4 sampler, loss and frozen encoder")
import torch
from train_cnn_mock import contrastive_batches, nt_xent, pretrain_encoder, PooledCNN
par = np.repeat(np.arange(20), 6)
par = np.r_[par, np.full(3, 20)]                         # one parent with fewer units
bt = contrastive_batches(par, 8, np.random.default_rng(0))
flat = np.concatenate(bt)
assert sorted(flat.tolist()) == list(range(len(par)))   # every unit once per epoch
assert all(len(set(par[b])) == len(b) for b in bt) and min(len(b) for b in bt) >= 2
print(f"    {len(bt)} batches; every unit once; no batch repeats a parent; min batch {min(len(b) for b in bt)}")
bt1 = contrastive_batches(np.r_[np.arange(9), np.arange(9)], 8, np.random.default_rng(1))
assert all(len(b) >= 2 for b in bt1)                     # 9 = 8 + 1 -> merged
expect(ValueError, contrastive_batches, np.r_[0, 0], 8, np.random.default_rng(0),
       what="layer with one unit (no negatives)")
B = 4
e = torch.eye(8)[:B]
z = torch.cat([e, e])                                      # views identical, units orthogonal
lo_ = float(nt_xent(z, 0.2))
zp = torch.cat([e, e[[1, 0, 3, 2]]])                       # positives swapped
assert lo_ < float(nt_xent(zp, 0.2))
zu = torch.nn.functional.normalize(torch.ones(2 * B, 8), dim=1)
assert np.isclose(float(nt_xent(zu, 0.2)), np.log(2 * B - 1), atol=1e-5)   # self masked
print("    NT-Xent: designated positives score best; self-similarity masked (uniform -> log(2B-1))")
torch.manual_seed(0)
ux = np.random.default_rng(0).random((24, 128, 128)).astype(np.float32)
st, curve, nu, npairs = pretrain_encoder(ux, np.repeat(np.arange(12), 2), 1, 0.2, 8, "cpu",
                                         np.random.default_rng(0))
assert nu > 0 and npairs == 24 and len(curve) == 1 and np.isfinite(curve).all()
tx = np.random.default_rng(1).random((34, 3, 128, 128)).astype(np.float32)
ty = np.random.default_rng(2).random(34).astype(np.float32)
enc0 = {k: v.clone() for k, v in st.items()}
m_, _, _ = run_fold(tx, ty, tx[:4], ty[:4], 1, 32, "cpu", 0.5, encoder_state=st)
after = m_.encoder.state_dict()
assert all(torch.equal(enc0[k], after[k].cpu()) for k in enc0), "encoder parameters/BN buffers changed"
assert any("running_mean" in k for k in enc0)
assert all(not q.requires_grad for q in m_.encoder.parameters())
assert all(q.requires_grad for q in m_.head.parameters())
print(f"    pretraining 1 epoch: {nu} updates, {npairs} pairs; frozen encoder: all "
      f"{len(enc0)} parameters and BatchNorm buffers unchanged after a head step")
print("ALL E5/E4 FIXTURE CHECKS PASSED")
