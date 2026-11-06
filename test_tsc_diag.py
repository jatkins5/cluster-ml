#!/usr/bin/env python3
"""Fixtures for diagnose_tsc_hardness.py (review rounds 33-34) -- synthetic data
only. Unit checks plus a complete synthetic input -> saved-output run of the
real entry point, and its rejection/availability cases."""
import copy
import json
import os
import tempfile

import h5py
import numpy as np
from astropy.cosmology import Planck15
from sklearn.metrics import r2_score

import compare_labels as L
import diagnose_tsc_hardness as D

tau = np.round(np.arange(1, 41) * 0.1, 10)
TMP = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "."))


def expect(f, *a, what="", **k):
    try:
        f(*a, **k)
    except SystemExit as e:
        print(f"    rejected ({what}): {e}")
        return
    raise SystemExit(f"NOT rejected: {what}")


# ================================================================ units
print("== crossing classes and the stored rule")
cases = {"floor": np.full(40, 0.7), "interior": np.r_[np.zeros(4), 0.5, np.full(35, 0.6)],
         "final": np.r_[np.full(39, 0.3), 0.6], "final_exact": np.r_[np.full(39, 0.3), 0.5],
         "none": np.full(40, 0.3)}
for k, cv in cases.items():
    c, i = D.crossing_class(cv)
    v = D.pseudo_from_curve(cv, tau)
    assert c == ("final" if k.startswith("final") else k), (k, c)
    print(f"    {k:<11} -> class {c:<8} index {i:>3}, pseudo {v:.4f}")
assert D.pseudo_from_curve(cases["final"], tau) < 4.0
assert D.pseudo_from_curve(cases["final_exact"], tau) == 4.0 and D.crossing_class(cases["final_exact"])[0] == "final"
assert D.pseudo_from_curve(cases["none"], tau) == 4.0 and D.crossing_class(cases["none"])[0] == "none"
print("    final score exactly 0.5 -> pseudo exactly 4 WITH a crossing; no crossing -> 4 without")
near = np.r_[np.zeros(4), np.float32(0.49999997), np.full(35, 0.6)].astype(np.float32)
assert D.crossing_class(near) == ("interior", 5)
print("    float32 0.49999997 does not cross at that sample")
expect(D.validate_curves, tau[::-1], np.zeros((2, 40)), np.zeros((2, 40)), what="decreasing tau grid")
expect(D.validate_curves, tau, np.zeros((2, 39)), np.zeros((2, 39)), what="curve width != tau length")

print("== past-event convention near the epoch")
t_now = 13.8
rows, dt = D.past_events(np.array([0, 1, 2]), np.array([13.8 + 5e-10, 13.8 + 1e-6, 12.8]), t_now)
assert rows.tolist() == [0, 2] and dt[0] == 0.0 and np.isclose(dt[1], 1.0)
print("    T_coll = t_now + 5e-10 admitted with elapsed 0; t_now + 1e-6 excluded")
D.verify_recent(dt, 0.0, "ok")
expect(D.verify_recent, dt, 0.5, "x", what="most-recent elapsed time disagrees with the saved truth")
expect(D.verify_recent, np.zeros(0), 0.3, "x", what="no past event but finite saved truth")

print("== setting-event tie rule and membership")
T = np.array([12.0, 13.0, 13.5, 13.5]); MR = np.array([0.2, 0.3, 0.05, 0.4])
assert D.setting_event(np.array([0, 2, 3]), T, MR)[:4] == (3, 2, 0.4, True)
assert D.setting_event(np.array([3, 0, 2]), T, MR)[:3] == (3, 2, 0.4)
MR2 = MR.copy(); MR2[2] = 0.4
assert D.setting_event(np.array([0, 2, 3]), T, MR2)[0] == 2
MR3 = MR.copy(); MR3[2], MR3[3] = -1, np.nan
s3 = D.setting_event(np.array([0, 2, 3]), T, MR3)
assert s3[0] == 2 and not s3[3]
print("    larger valid ratio wins; equal ratios -> lowest row; all-invalid -> lowest row, invalid; order-free")
assert D.events_for(5, {1, 5}, {}).size == 0
expect(D.events_for, 7, {1, 5}, {}, what="requested parent absent from the catalogue")
print("    catalogue member without events retained")

print("== identities and float32 inputs")
rg = np.random.default_rng(0)
y = rg.uniform(0, 4, 90)
e32 = (y + rg.normal(0, 0.3, 90)).astype(np.float32)
D.affine_check(y, e32)
D.mse_identity(e32, y + 0.1, y)
print("    affine invariance and accounting identity pass with float32 predictions (float64 path)")


def comp(share, mse_c, mse_p, by, bo):
    c = {s: dict(sse_cat=mse_c * 10, mse_cat=mse_c, mse_pse=mse_p, n_obs=10, bias_cat=0.0) for s in D.RANGE_STRATA}
    c["[0,0.5]"] = dict(sse_cat=0.0, mse_cat=0, mse_pse=0, n_obs=10, bias_cat=by)
    c["finite>4"]["bias_cat"] = bo
    c["sse_cat_total"] = 3 * mse_c * 10 / share
    return c


assert D.nominate(comp(0.50, 4.0, 2.0, 0, 0))[0].startswith("range")
assert D.nominate(comp(0.4999, 4.0, 2.0, 0.5, -0.5))[0].startswith("loss")
assert D.nominate(comp(0.60, 1.0, 2.0, 0.3, -0.5))[0] == "none"
assert D.nominate(comp(0.60, 1.0, 2.0, 0.31, -0.3))[0] == "none"
assert D.nominate(comp(0.60, 1.0, 2.0, 0.31, -0.31))[0].startswith("loss")
print("    nomination boundaries as frozen")

# ================================================================ end to end
print("== complete synthetic input -> saved output")
t99, t91 = float(Planck15.age(0.0).value), float(Planck15.age(0.0994018026302219).value)
PAR = [10, 11, 12, 13, 14]
CAT = PAR + [15]                                   # 15: catalogue member, never requested
EV = [  # parent, T_coll, Snap_coll, mass ratio
    (11, t99 - 0.3, 99, 0.2), (11, t99 - 3.5, 80, 0.1),
    (12, t99 - 1.0, 97, 0.1), (12, t99 - 1.0, 97, 0.3), (12, t99 - 6.0, 60, 0.05),   # tie; snap 60 not in table
    (13, t91 - 0.2, 91, -1.0), (13, t99 - 5.0, 75, 0.2),
    (14, t99, 99, 0.5), (14, t91 + 0.5, 95, 0.02)]                                 # event at t99; future at 91
ROWS = [(p, 99) for p in PAR] + [(p, 91) for p in PAR]
TNOW = {99: t99, 91: t91}


def write_world(d, ev=EV, cat=CAT, nan_curve=False, groupcat=True):
    os.makedirs(d, exist_ok=True)
    with h5py.File(f"{d}/cat.h5", "w") as f:
        f["haloID"] = np.array(cat)
    pos = {p: i for i, p in enumerate(cat)}
    with h5py.File(f"{d}/mergers.h5", "w") as f:
        f["HaloID"] = np.array([pos[e[0]] for e in ev])
        f["T_coll"] = np.array([e[1] for e in ev]); f["Snap_coll"] = np.array([e[2] for e in ev])
        f["Mass_ratio"] = np.array([e[3] for e in ev])
    gc = f"{d}/groupcat"
    os.makedirs(gc, exist_ok=True)
    if groupcat:
        for sn, z in ((75, 0.6), (80, 0.45), (91, 0.0994018026302219), (95, 0.05), (97, 0.02), (99, 0.0)):
            os.makedirs(f"{gc}/groupcat_{sn:03d}", exist_ok=True)
            with h5py.File(f"{gc}/groupcat_{sn:03d}/fof_subhalo_tab_{sn:03d}.0.hdf5", "w") as f:
                f.create_group("Header").attrs["Redshift"] = z
    raw, curves = [], []
    T_ = np.array([e[1] for e in ev])
    for p, s in ROWS:
        r_ = np.array([i for i, e in enumerate(ev) if e[0] == p], dtype=int)
        _, dt = D.past_events(r_, T_, TNOW[s])
        raw.append(np.min(dt) if len(dt) else np.nan)
        curves.append(D.reconstruct(dt, tau))
    curves = np.array(curves, dtype=np.float32)
    if nan_curve:
        curves[2, 5] = np.nan
    with h5py.File(f"{d}/curves.h5", "w") as f:
        f["labels/tau_gyr"] = tau
        f["labels/label_score_all"] = curves
        f["labels/label_score_pre"] = curves
        f["meta/group_id"] = np.array([p for p, _ in ROWS]); f["meta/snap"] = np.array([s for _, s in ROWS])
    pseudo = np.array([D.pseudo_from_curve(c, tau) for c in curves.astype(np.float64)]) if not nan_curve else \
        np.array([D.pseudo_from_curve(np.nan_to_num(c), tau) for c in curves.astype(np.float64)])
    return np.array(raw), pseudo


MAN = "f" * 64
R, P, F = 3, 3, 5


def artifacts(pdir, raw, pseudo, permute_radio=False):
    os.makedirs(pdir, exist_ok=True)
    capf = lambda v: np.where(np.isnan(v), 4.0, np.minimum(v, 4.0))
    for m in L.MODALITIES:
        order = np.arange(len(ROWS))
        if m == "r" and permute_radio:
            order = order[::-1]
        grp = np.array([ROWS[i][0] for i in order]); sn = np.array([ROWS[i][1] for i in order])
        rw, ps = raw[order], pseudo[order]
        n = len(order)
        for lab in L.LABELS:
            y = ps if lab == "pseudo" else capf(rw)
            for s in (42, 43):
                rg_ = np.random.default_rng(s)
                cfg = dict(L.REQUIRED_CFG, epochs=60, dataset=L.MODALITIES[m], seed=s, label=lab,
                           out_prefix=f"x{s}", tsc_dir="/x")
                prov = dict(head="h", status="", sha256={"train_cnn_mock.py": "a"}, config=cfg, manifest_sha256=MAN,
                            inputs_sha256={"dataset": m, "catalog": "c", "tsc_snap99": "t9", "tsc_snap91": "t1"})
                d = dict(sample_halo=np.repeat(1000 + order, R), sample_group=np.repeat(grp, R),
                         sample_snap=np.repeat(sn, R), sample_realization=np.tile(np.arange(R), n),
                         group_id=grp, snap=sn, y_sample=np.repeat(y, R).astype(np.float32), tsc=y.astype(np.float32),
                         fold_of=np.repeat(grp % F, R).astype(np.int16),
                         sel_groups_flat=np.array([PAR[(f + 1) % F] for f in range(F)]), sel_groups_fold=np.arange(F),
                         tsc_cap=np.array(4.0), fold_best_epoch=np.full(F, 30), fold_sel_score=np.full(F, 0.3),
                         select_criterion=np.array("inner:single:r2"),
                         truth_pseudo=ps, truth_tsc_capped=capf(rw), truth_tsc_raw=rw,
                         truth_tcoll_capped=capf(rw), truth_tcoll_raw=rw, label_name=np.array(lab),
                         oof_sample=(np.repeat(y, R) + rg_.normal(0, 0.2, n * R)).astype(np.float32),
                         oof_single=(np.repeat(y, R)[:, None] + rg_.normal(0, 0.2, (n * R, P))).astype(np.float32),
                         provenance=np.array(json.dumps(prov)), provenance_changed_during_run=np.array(False))
                np.savez(os.path.join(pdir, L.pattern(m, lab).format(s)), **d)


def argv(d, pdir, out, extra=()):
    return ["--expect-manifest", MAN, "--pred-dir", pdir, "--out", out, "--seeds", "42,43", "--boot", "50",
            "--curves", f"{d}/curves.h5", "--mergers", f"{d}/mergers.h5", "--catalog", f"{d}/cat.h5",
            "--groupcat", f"{d}/groupcat", *extra]


w = f"{TMP}/world"
raw, pseudo = write_world(w)
artifacts(f"{w}/preds", raw, pseudo)
out = D.run(argv(w, f"{w}/preds", f"{w}/diag.npz"))
saved = np.load(f"{w}/diag.npz", allow_pickle=True)
assert saved["row_group"].tolist() == [p for p, _ in ROWS] and saved["row_snap"].tolist() == [s for _, s in ROWS]
assert saved["setting_row"].tolist()[0] == -1                        # parent 10: member with no events, retained
assert saved["setting_row"].tolist()[2] == 3                         # parent 12 tie -> larger ratio row
assert bool(saved["a0_tcoll_reproduces_99"]) and bool(saved["a0_tcoll_reproduces_91"])
assert not saved["avail_snapclock"][2] and saved["avail_snapclock"][1]   # parent 12 uses snap 60 (no header)
assert saved["xtab"].sum() == len(PAR)
nom = json.loads(str(saved["nomination"]))
assert set(nom) >= {"candidate", "share", "finite_old_share", "missing_share", "bias_young", "bias_old"}
prov = json.loads(str(saved["provenance"]))
assert len(prov["predictions"]) == 8 and "snapshot_table" in prov and prov["snapshot_table"]["99"]["path"]
print("    end-to-end run saved all row keys, setting rows (missing member retained, tie rule), A0 per-row results,"
      " partial snapshot-clock availability, cross-tab, nomination components and provenance")
expect(D.run, argv(w, f"{w}/preds", f"{w}/diag.npz"), what="existing output (no overwrite)")

w2 = f"{TMP}/nan"; r2_, p2_ = write_world(w2, nan_curve=True); artifacts(f"{w2}/preds", r2_, p2_)
expect(D.run, argv(w2, f"{w2}/preds", f"{w2}/o.npz"), what="NaN in a stored curve")
w3 = f"{TMP}/miss"; r3_, p3_ = write_world(w3, cat=[10, 11, 12, 14, 15], ev=[e for e in EV if e[0] != 13])
artifacts(f"{w3}/preds", r3_, p3_)
expect(D.run, argv(w3, f"{w3}/preds", f"{w3}/o.npz"), what="requested parent missing from the catalogue")
w4 = f"{TMP}/perm"; r4_, p4_ = write_world(w4); artifacts(f"{w4}/preds", r4_, p4_, permute_radio=True)
expect(D.run, argv(w4, f"{w4}/preds", f"{w4}/o.npz"), what="radio rows consistently permuted")
w5 = f"{TMP}/nogc"; r5_, p5_ = write_world(w5, groupcat=False); artifacts(f"{w5}/preds", r5_, p5_)
o5 = D.run(argv(w5, f"{w5}/preds", f"{w5}/o.npz"))
assert not o5["avail_snapclock"].any() and o5["a0_tcoll_reproduces_99"]
print("    empty snapshot table: snapshot clock unavailable everywhere; T_coll and stored-curve diagnostics still run")
print("ALL TSC-DIAGNOSTIC FIXTURE CHECKS PASSED")
