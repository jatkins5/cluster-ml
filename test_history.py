#!/usr/bin/env python3
"""Fixtures for diagnose_history.py (review rounds 42-43) -- synthetic only."""
import json
import os
import tempfile

import h5py
import numpy as np
from astropy.cosmology import Planck15

import compare_labels as L
import diagnose_history as H
import diagnose_tsc_hardness as D

TMP = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "."))


def expect(f, *a, what="", **k):
    try:
        f(*a, **k)
    except SystemExit as e:
        print(f"    rejected ({what}): {e}")
        return
    raise SystemExit(f"NOT rejected: {what}")


print("== strict-previous features and invalid weights")
t_now = 13.8
T = np.array([13.5, 13.5, 12.0, 10.0, 13.9])          # rows 0,1 tied latest; row 4 is future
MR = np.array([0.2, 0.3, 0.1, 0.05, 0.4])
fo, dt = H.event_features(np.arange(5), T, MR, t_now)
assert fo["latest_rows"] == [0, 1] and fo["prev_rows"] == [2, 3] and fo["n_latest"] == 2
assert np.isclose(fo["latest_strength"], 0.5)
assert np.isclose(fo["M_prev_1.0"], 0.1 * np.exp(-1.8) + 0.05 * np.exp(-3.8))
assert np.isclose(fo["S_prev_1.0"], np.exp(-1.8) + np.exp(-3.8))
print("    whole latest tie group excluded from prev; latest strength = summed tied ratios")
MR2 = MR.copy(); MR2[3] = -1.0
fo2, _ = H.event_features(np.arange(5), T, MR2, t_now)
assert np.isnan(fo2["M_prev_1.0"]) and np.isnan(fo2["M_all_1.0"]) and fo2["prev_invalid"] == 1
assert np.isclose(fo2["M_prev_incomplete_1.0"], 0.1 * np.exp(-1.8)) and np.isfinite(fo2["S_prev_1.0"])
print("    invalid prev ratio -> M_prev and M_all unavailable (incomplete diagnostic kept)")
MR3 = MR.copy(); MR3[1] = np.nan
fo3, _ = H.event_features(np.arange(5), T, MR3, t_now)
assert np.isnan(fo3["latest_strength"]) and np.isnan(fo3["M_all_1.0"]) and np.isfinite(fo3["M_prev_1.0"])
print("    invalid latest ratio -> latest strength and M_all unavailable, M_prev unaffected")
fo4, _ = H.event_features(np.array([0]), np.array([13.0]), np.array([0.3]), t_now)
assert fo4["n_prev"] == 0 and fo4["M_prev_1.0"] == 0.0 and fo4["S_prev_1.0"] == 0.0
fo5, _ = H.event_features(np.array([], int), T, MR, t_now)
assert np.isnan(fo5["T_last"])
print("    no prev events -> genuine 0; no events at all -> T_last NaN")

print("== designs")
Tg = np.linspace(0, 4, 50)
assert H.pl_basis(Tg)[0].shape[1] == 7 and H.ncs_basis(Tg)[0].shape[1] == 6
assert H.fold_indicators(np.arange(50) % 5)[0].shape[1] == 4
X, _ = H.ncs_basis(Tg)
d2 = np.diff(X[:, 2:], 2, axis=0)
assert np.allclose(d2[Tg[1:-1] > 4.0 - 1e-9], 0) if (Tg > 4).any() else True
print("    piecewise linear 7 columns (with intercept), natural cubic spline 6, fold indicators 4")
c, ok, _ = H.ols(np.column_stack([np.ones(10), np.arange(10), np.arange(10)]), np.arange(10.0))
assert not ok
print("    duplicate column -> rank-deficient")

rng = np.random.default_rng(0)


def world(n=250, beta=0.0, rng=rng, hetero=True, nonlinear=True, young=False):
    Tl = 0.02 + 3.98 * rng.beta(0.8, 1.5, n) if young else rng.uniform(0.05, 4.0, n)
    fold = np.arange(n) % 5
    M = rng.gamma(2.0, 0.1, n) * np.exp(-Tl / 2) + rng.uniform(0, 0.05, n)   # age-dependent activity
    base = 1.2 * np.log1p(2 * Tl) + 0.3 * np.sin(2 * Tl) if nonlinear else 0.8 * Tl
    sd = 0.2 + 0.15 * Tl if hetero else 0.3
    y = base + beta * M + rng.normal(0, sd)
    return y, Tl, fold, M


print("== B1 behaviour")
y, Tl, fold, M = world(beta=0.0)
chk = H.b1_fit(np.minimum(Tl, 4), Tl, fold, M, boot=0)
assert abs(chk["beta"]) < 1e-8
print(f"    algebra: response = T gives beta {chk['beta']:.1e}")
yp, Tp, fp, Mp = world(beta=-3.0)
r = H.b1_fit(yp, Tp, fp, Mp, boot=500, seed=1)
assert r["interval_status"] == "ok" and r["lo"] < -3.0 < r["hi"] < 0
print(f"    planted beta -3: estimate {r['beta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
cover, reps = 0, 60
for i in range(reps):
    yy, tt, ff, mm = world(beta=0.0, rng=np.random.default_rng(100 + i))
    rr = H.b1_fit(yy, tt, ff, mm, boot=200, seed=i)
    cover += int(rr["lo"] <= 0 <= rr["hi"])
cv = cover / reps
print(f"    synthetic null (nonlinear age-only response, age-dependent activity, heteroscedastic errors; "
      f"{reps} replicates, 200 draws): empirical 95% coverage {cv:.2f} +- {np.sqrt(cv * (1 - cv) / reps):.2f} (MC s.e.); "
      f"reported, not tuned")
assert cv > 0.5, "gross failure of the null fixture"
r_rd = H.b1_fit(y, Tl, fold, np.maximum(Tl - 1, 0), boot=0)
assert r_rd["status"] == "rank-deficient" and np.isnan(r_rd["beta"])
print(f"    feature exactly in the nuisance span (a knot column) -> '{r_rd['status']}', no coefficient")
fw = np.maximum(Tl - 1, 0) + np.random.default_rng(3).normal(0, 0.01 * np.std(np.maximum(Tl - 1, 0)), len(Tl))
r_ws = H.b1_fit(y, Tl, fold, fw, boot=0)
assert r_ws["status"].startswith("insufficient support") and np.isfinite(r_ws["beta"])
assert H.gate_part(dict(r_ws, interval_status="ok", lo=-9, hi=-8, beta=-8.5)) == "unavailable"
print(f"    feature nearly in the span -> '{r_ws['status']}' (support {r_ws['support']:.3f}); gate unavailable")
Fsparse = np.zeros(len(y)); Fsparse[[3, 7]] = [1.0, 2.0]
r_bad = H.b1_fit(y, Tl, fold, Fsparse, boot=300, seed=2)
assert r_bad["interval_status"].startswith("unavailable") and len(r_bad["failed_draws"]) > 0
print(f"    resamples dropping all non-zero feature values -> {r_bad['interval_status']}")
assert H.gate_part(r_bad) == "unavailable" and H.gate_part(r) == "pass"

print("== B2 splits and coverage")
par = list(range(100, 350))
pf = {p: p % 5 for p in par}
s1 = H.make_splits(par, pf)
s2 = H.make_splits(par[::-1], pf)
assert all(np.array_equal(s1[f][0], s2[f][0]) and np.array_equal(s1[f][1], s2[f][1]) for f in s1)
H.check_splits(s1, pf)
print("    splits reproducible independently of input order; halves disjoint, one original fold each")
bad = {f: (a.copy(), b.copy()) for f, (a, b) in s1.items()}
bad[0] = (np.r_[bad[0][0], 101], bad[0][1])
expect(H.check_splits, bad, pf, what="cross-original-fold calibration parent")
yb, Tb, fb, Mb = world(n=250, beta=-3.0, rng=np.random.default_rng(7), young=True)
rk = H.b2_run(np.array(par), yb, Tb, Mb, pf, s1, boot=50, knots=H.KNOTS_PL)
nrd = sum(v == "unavailable: rank-deficient fit" for v in rk["status"].values())
assert nrd > 0 and not rk["full_coverage"]
print(f"    full 5-knot basis on ~25-parent halves: {nrd}/10 fits rank-deficient (empty segment) -> "
      f"coverage {rk['covered']}/{rk['n']}, primary unavailable")
res = H.b2_run(np.array(par), yb, Tb, Mb, pf, s1, boot=300)
assert res["full_coverage"] and res["interval_status"] == "ok", (res["status"], res["interval_status"])
print(f"    proposed B2 basis {H.B2_KNOTS}: full coverage {res['covered']}/{res['n']}: dR2 {res['dr2']:+.4f} [{res['lo']:+.4f}, {res['hi']:+.4f}] "
      f"(planted effect)")
res2 = H.b2_run(np.array(par), yb, Tb, Mb, pf, s1, boot=100, min_half=26)
assert not res2["full_coverage"] and res2["interval_status"].startswith("unavailable")
print(f"    halves below the minimum -> partial coverage {res2['covered']}/{res2['n']}; primary interval unavailable")

print("== end to end (synthetic 250-parent world, two seeds, reduced draws)")
t99 = float(Planck15.age(0).value)
NP = 250
PAR = list(range(1000, 1000 + NP))
CAT = PAR + [9999]
erng = np.random.default_rng(11)
EV = []
for p in PAR[:-3]:
    for _ in range(erng.integers(1, 5)):
        EV.append((p, t99 - erng.uniform(0.0, 3.9), erng.uniform(0.01, 0.6)))
EV.append((PAR[0], t99 - 2.0, -1.0))                   # invalid prev ratio for one parent
for p in PAR[10:30]:                                    # tied latest collisions for 20 parents
    last = max(e[1] for e in EV if e[0] == p)
    EV.append((p, last, erng.uniform(0.01, 0.6)))
d = f"{TMP}/w"
os.makedirs(d)
with h5py.File(f"{d}/cat.h5", "w") as f:
    f["haloID"] = np.array(CAT)
pos = {p: i for i, p in enumerate(CAT)}
with h5py.File(f"{d}/m.h5", "w") as f:
    f["HaloID"] = np.array([pos[e[0]] for e in EV]); f["T_coll"] = np.array([e[1] for e in EV])
    f["Snap_coll"] = np.full(len(EV), 99); f["Mass_ratio"] = np.array([e[2] for e in EV])
Tc = np.array([e[1] for e in EV])
raw = []
for p in PAR:
    rws = np.array([i for i, e in enumerate(EV) if e[0] == p], int)
    _, dt = D.past_events(rws, Tc, t99)
    raw.append(np.min(dt) if len(dt) else np.nan)
raw = np.array(raw)
capf = lambda v: np.where(np.isnan(v), 4.0, np.minimum(v, 4.0))
cap = capf(raw)
pse = np.minimum(np.where(np.isnan(raw), 4.0, raw / np.log(2)), 4.0)
R, P, F = 3, 3, 5
grp = np.array(PAR)
for m in L.MODALITIES:
    for lab in L.LABELS:
        yv = pse if lab == "pseudo" else cap
        for s in (42, 43):
            rg_ = np.random.default_rng(s)
            cfg = dict(L.REQUIRED_CFG, epochs=60, dataset=L.MODALITIES[m], seed=s, label=lab, out_prefix="x", tsc_dir="/x")
            prov = dict(head="h", status="", sha256={"train_cnn_mock.py": "a"}, config=cfg, manifest_sha256="f" * 64,
                        inputs_sha256={"dataset": m, "catalog": "c", "tsc_snap99": "t9", "tsc_snap91": "t1"})
            dd = dict(sample_halo=np.repeat(grp, R), sample_group=np.repeat(grp, R), sample_snap=np.full(NP * R, 99),
                      sample_realization=np.tile(np.arange(R), NP), group_id=grp, snap=np.full(NP, 99),
                      y_sample=np.repeat(yv, R).astype(np.float32), tsc=yv.astype(np.float32),
                      fold_of=np.repeat(grp % F, R).astype(np.int16),
                      sel_groups_flat=np.array([PAR[k] for f in range(F) for k in range(NP) if PAR[k] % F == (f + 1) % F][:0]
                                               + [PAR[((f + 1) % F) + 5 * j] for f in range(F) for j in range(2)]),
                      sel_groups_fold=np.repeat(np.arange(F), 2), tsc_cap=np.array(4.0),
                      fold_best_epoch=np.full(F, 30), fold_sel_score=np.full(F, 0.3),
                      select_criterion=np.array("inner:single:r2"), truth_pseudo=pse, truth_tsc_capped=cap,
                      truth_tsc_raw=raw, truth_tcoll_capped=cap, truth_tcoll_raw=raw, label_name=np.array(lab),
                      sim_logm=np.linspace(14.5, 15.3, NP),
                      oof_sample=(np.repeat(yv, R) + rg_.normal(0, 0.3, NP * R)).astype(np.float32),
                      oof_single=(np.repeat(yv, R)[:, None] + rg_.normal(0, 0.3, (NP * R, P))).astype(np.float32),
                      provenance=np.array(json.dumps(prov)), provenance_changed_during_run=np.array(False))
            os.makedirs(f"{d}/p", exist_ok=True)
            np.savez(f"{d}/p/" + L.pattern(m, lab).format(s), **dd)
argv = ["--expect-manifest", "f" * 64, "--pred-dir", f"{d}/p", "--out", f"{d}/o.npz", "--seeds", "42,43",
        "--boot", "100", "--mergers", f"{d}/m.h5", "--catalog", f"{d}/cat.h5"]
out = H.run(argv)
sv = np.load(f"{d}/o.npz", allow_pickle=True)
g = json.loads(str(sv["gate"]))
assert set(g) == {"i", "ii", "iii", "overall"} and all(v in ("pass", "fail", "unavailable") for k, v in g.items() if k != "overall")
assert str(sv["exclusion"][0]) == "invalid prev ratio" and not sv["primary"][0]
assert (sv["exclusion"] == "no recorded event").sum() == 3
prov = json.loads(str(sv["provenance"]))
assert len(prov["predictions"]) == 8 and prov["constants"]["tau0"] == 1.0
adjr = json.loads(str(sv["b1"]))["adjusted"]
if True:
    assert adjr["status"] == "ok" and np.isfinite(adjr["beta"]), adjr["status"]
    print(f"    adjusted model with 20 tied parents: status ok, beta {adjr['beta']:+.4f}")
print(f"    saved gate {g}; exclusions recorded; provenance with 8 prediction records")
expect(H.run, argv, what="existing output (no overwrite)")
print("ALL HISTORY-DIAGNOSTIC FIXTURE CHECKS PASSED")
