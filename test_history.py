#!/usr/bin/env python3
"""Fixtures for diagnose_history.py (review rounds 42-44) -- synthetic only."""
import json
import os
import tempfile
import time

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
Tg = np.linspace(0, 4, 81)
assert H.pl_basis(Tg)[0].shape[1] == 7 and H.pl_basis(Tg, H.B2_KNOTS)[0].shape[1] == 5
assert H.ncs_basis(Tg)[0].shape[1] == 6 and H.fold_indicators(np.arange(50) % 5)[0].shape[1] == 4
Xin = H.ncs_basis(Tg)[0][:, 2:]
Tt = np.linspace(4.0, 6.0, 41)
Xt = H.ncs_basis(Tt)[0][:, 2:]
d2_in, d2_tail = np.abs(np.diff(Xin, 2, axis=0)).max(), np.abs(np.diff(Xt, 2, axis=0)).max()
assert d2_in > 1e-4 and d2_tail < 1e-9 * max(1.0, np.abs(Xt).max())
Xl = H.ncs_basis(np.linspace(-2.0, 0.0, 21))[0][:, 2:]
assert np.abs(Xl).max() == 0.0
print(f"    pl 7 columns (B2 amended: 5), ncs 6, fold indicators 4; ncs curvature inside [0,4] "
      f"(max |d2| {d2_in:.2e}), linear beyond the boundary knot 4 (max |d2| {d2_tail:.1e}), zero below 0")
c, ok, _, rk = H.ols(np.column_stack([np.ones(10), np.arange(10), np.arange(10)]), np.arange(10.0))
assert not ok and rk == 2
assert H.ols(np.zeros((0, 3)), np.zeros(0))[1] is False
print("    duplicate column -> rank 2/3, rank-deficient; empty design -> not identifiable")

rng = np.random.default_rng(0)


def world(n=250, beta=0.0, rng=rng, hetero=True, nonlinear=True, young=False):
    Tl = 0.02 + 3.98 * rng.beta(0.8, 1.5, n) if young else rng.uniform(0.05, 4.0, n)
    fold = np.arange(n) % 5
    M = rng.gamma(2.0, 0.1, n) * np.exp(-Tl / 2) + rng.uniform(0, 0.05, n)   # age-dependent activity
    base = 1.2 * np.log1p(2 * Tl) + 0.3 * np.sin(2 * Tl) if nonlinear else 0.8 * Tl
    sd = 0.2 + 0.15 * Tl if hetero else 0.3
    y = base + beta * M + rng.normal(0, sd)
    return y, Tl, fold, M


def coverage(label, beta_true, nonlinear, seed0, reps=60, draws=200):
    valid = cover = 0
    for i in range(reps):
        yy, tt, ff, mm = world(beta=beta_true, rng=np.random.default_rng(seed0 + i), nonlinear=nonlinear)
        rr = H.b1_fit(yy, tt, ff, mm, boot=draws, seed=i)
        if rr["interval_status"] != "ok":
            continue
        valid += 1
        cover += int(rr["lo"] <= beta_true <= rr["hi"])
    se = lambda p, n: np.sqrt(p * (1 - p) / n) if n else np.nan
    cv, un = (cover / valid if valid else np.nan), cover / reps
    print(f"    {label}: {reps} replicates x {draws} draws; valid intervals {valid}, unavailable {reps - valid}; "
          f"valid-interval coverage of beta={beta_true:g}: {cv:.3f} +- {se(cv, valid):.3f}; "
          f"unconditional inclusion {un:.3f} +- {se(un, reps):.3f} (MC s.e.)")
    return cv


print("== B1 behaviour")
y, Tl, fold, M = world(beta=0.0)
chk = H.b1_fit(np.minimum(Tl, 4), Tl, fold, M, boot=0)
assert abs(chk["beta"]) < 1e-8 and chk["interval_status"] == "not requested"
print(f"    algebra: response = T gives beta {chk['beta']:.1e}")
yp, Tp, fp, Mp = world(beta=-3.0, nonlinear=False, rng=np.random.default_rng(1))
r = H.b1_fit(yp, Tp, fp, Mp, boot=500, seed=1)
assert r["interval_status"] == "ok" and r["lo"] < -3.0 < r["hi"] < 0
assert len(r["coef"]) == len(r["names"]) == r["n_columns"] == r["rank"] == 12 and len(r["residual_feature"]) == 250
print(f"    correctly specified planted beta -3: estimate {r['beta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]; "
      f"coef vector, rank {r['rank']}/{r['n_columns']} and residual feature saved")
print("    -- correctly specified mean (0.8 T, inside the piecewise-linear span), heteroscedastic errors:")
cv_s = coverage("null", 0.0, False, 300)
cv_p = coverage("planted -3", -3.0, False, 500)
print("    -- misspecification stress (unchanged: nonlinear age mean outside the span, age-dependent M); "
      "the target is a projection coefficient, so inclusion of 0 mixes misspecification, interval behaviour and MC noise:")
cv_x = coverage("stress null", 0.0, True, 100)
assert min(cv_s, cv_p) > 0.5, "gross failure (not a validation of nominal coverage)"
r_rd = H.b1_fit(y, Tl, fold, np.maximum(Tl - 1, 0), boot=50)
assert r_rd["status"] == "rank-deficient" and np.isnan(r_rd["beta"]) and r_rd["coef"] is None
assert r_rd["interval_status"].startswith("unavailable") and set(r_rd) == set(r)
print(f"    feature exactly in the nuisance span -> '{r_rd['status']}', rank {r_rd['rank']}/{r_rd['n_columns']}, "
      f"no coefficient; same result keys as a successful fit")
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
r_e = H.b1_fit(np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0), boot=10)
assert r_e["status"] == "unavailable: empty cohort" and set(r_e) == set(r) and H.gate_part(r_e) == "unavailable"
r_c = H.b1_fit(np.full(250, 1.5), Tl, fold, M, boot=0)
assert r_c["partial_r2_status"] == "undefined: zero response variance" and np.isnan(r_c["partial_r2"])
print("    empty cohort -> 'unavailable: empty cohort'; constant response -> partial R2 undefined (explicit)")

print("== B2 splits and coverage")
par = list(range(100, 350))
pf = {p: p % 5 for p in par}
s1 = H.make_splits(par, pf)
s2 = H.make_splits(par[::-1], pf)
assert all(np.array_equal(s1[f][0], s2[f][0]) and np.array_equal(s1[f][1], s2[f][1]) for f in s1)
H.check_splits(s1, pf, expected=par)
print("    splits reproducible independently of input order; halves disjoint, one original fold each, exact cover")
bad = {f: (a.copy(), b.copy()) for f, (a, b) in s1.items()}
bad[0] = (np.r_[bad[0][0], 101], bad[0][1])
expect(H.check_splits, bad, pf, what="cross-original-fold calibration parent")
dup = {f: (a.copy(), b.copy()) for f, (a, b) in s1.items()}
dup[2] = (np.r_[dup[2][0], dup[2][0][0]], dup[2][1])
expect(H.check_splits, dup, pf, what="duplicate id within a half")
expect(H.check_splits, s1, pf, expected=par[:-1], what="halves cover parents outside the analysis cohort")
miss = {f: (a.copy(), b.copy()) for f, (a, b) in s1.items()}
miss[1] = (miss[1][0][1:], miss[1][1])
expect(H.check_splits, miss, pf, expected=par, what="an analysis parent missing from every half")
yb, Tb, fb, Mb = world(n=250, beta=-3.0, rng=np.random.default_rng(7), young=True)
rk = H.b2_run(np.array(par), yb, Tb, Mb, pf, s1, boot=50, knots=H.KNOTS_PL)
nrd = sum(r_["status"] == "unavailable: rank-deficient fit" for r_ in rk["directions"])
assert nrd > 0 and not rk["full_coverage"]
print(f"    B1 5-knot basis on ~25-parent halves: {nrd}/10 fits rank-deficient (empty segment) -> "
      f"coverage {rk['covered']}/{rk['n']}, primary unavailable")
res = H.b2_run(np.array(par), yb, Tb, Mb, pf, s1, boot=300)
assert res["full_coverage"] and res["interval_status"] == "ok" and len(res["directions"]) == 10
assert all(len(r_["coef0"]) == 5 and len(r_["coef1"]) == 6 and r_["rank0"] == 5 and r_["rank1"] == 6
           for r_ in res["directions"])
assert sorted(sum((r_["eval_ids"] for r_ in res["directions"]), [])) == par and (res["n_predictions"] == 1).all()
assert len(res["boot_dr2"]) == res["draws_valid"] == 300
r0 = res["directions"][3]
ei = np.array([par.index(p) for p in r0["eval_ids"]])
assert np.allclose(res["pred0"][ei], H.pl_basis(Tb[ei], H.B2_KNOTS)[0] @ np.array(r0["coef0"]))
print(f"    amended B2 basis {H.B2_KNOTS}: full coverage {res['covered']}/{res['n']}, 10 directions with 5/6 "
      f"coefficients, ranks, segment counts, ids, draws; dR2 {res['dr2']:+.4f} [{res['lo']:+.4f}, {res['hi']:+.4f}] "
      f"(planted effect); saved predictions reproduce from saved coefficients")
res2 = H.b2_run(np.array(par), yb, Tb, Mb, pf, s1, boot=100, min_half=26)
assert not res2["full_coverage"] and res2["interval_status"].startswith("unavailable")
print(f"    halves below the minimum -> coverage {res2['covered']}/{res2['n']}; primary unavailable")
par4 = [p for p in par if pf[p] != 4]
s4 = H.make_splits(par4, pf)
i4 = np.array([par.index(p) for p in par4])
res4 = H.b2_run(np.array(par4), yb[i4], Tb[i4], Mb[i4], pf, s4, boot=50)
assert res4["covered"] == res4["n"] == len(par4) and not res4["full_coverage"] and not res4["all_directions_ok"]
assert res4["interval_status"].startswith("unavailable") and "ten fold directions" in res4["interval_status"]
print(f"    empty original fold 4: every retained row predicted ({res4['covered']}/{res4['n']}) but primary "
      f"unavailable ({res4['interval_status']})")
resc = H.b2_run(np.array(par), np.full(250, 2.0), Tb, Mb, pf, s1, boot=20)
assert resc["interval_status"] == "unavailable: held-out R2 undefined (zero response variance)"
res_e = H.b2_run(np.array([], int), np.zeros(0), np.zeros(0), np.zeros(0), pf, H.make_splits([], pf), boot=10)
assert not res_e["full_coverage"] and "empty cohort" in res_e["interval_status"]
iso_t = H.b2_isotonic(np.array(par), yb, {"T": (Tb, True)}, pf, s1)
assert iso_t["T"]["coverage"] == "full" and all(r_["x_thresholds"] for r_ in iso_t["T"]["directions"])
iso_c = H.b2_isotonic(np.array(par), np.full(250, 2.0), {"T": (Tb, True)}, pf, s1)
assert np.isnan(iso_c["T"]["r2"]) and iso_c["T"]["r2_status"].startswith("undefined")
iso_e = H.b2_isotonic(np.array(par4), yb[i4], {"T": (Tb[i4], True)}, pf, s4)
assert iso_e["T"]["coverage"].startswith("partial")
print("    zero-variance response -> B2 interval and isotonic R2 explicitly undefined; empty cohort -> unavailable; "
      "isotonic thresholds saved, partial coverage labelled descriptive")

print("== end to end (synthetic 250-parent worlds)")
t99 = float(Planck15.age(0).value)
NP = 250
PAR = list(range(1000, 1000 + NP))


def build(d, EV, seeds):
    CAT = PAR + [9999]
    os.makedirs(f"{d}/p")
    with h5py.File(f"{d}/cat.h5", "w") as f:
        f["haloID"] = np.array(CAT)
    pos = {p: i for i, p in enumerate(CAT)}
    with h5py.File(f"{d}/m.h5", "w") as f:
        f["HaloID"] = np.array([pos[e[0]] for e in EV], dtype=int)
        f["T_coll"] = np.array([e[1] for e in EV], dtype=float)
        f["Snap_coll"] = np.full(len(EV), 99)
        f["Mass_ratio"] = np.array([e[2] for e in EV], dtype=float)
    Tc = np.array([e[1] for e in EV], dtype=float)
    raw = []
    for p in PAR:
        rws = np.array([i for i, e in enumerate(EV) if e[0] == p], int)
        _, dt_ = D.past_events(rws, Tc, t99)
        raw.append(np.min(dt_) if len(dt_) else np.nan)
    raw = np.array(raw)
    cap = np.where(np.isnan(raw), 4.0, np.minimum(raw, 4.0))
    pse = np.minimum(np.where(np.isnan(raw), 4.0, raw / np.log(2)), 4.0)
    R, P, F = 3, 3, 5
    grp = np.array(PAR)
    for m in L.MODALITIES:
        for lab in L.LABELS:
            yv = pse if lab == "pseudo" else cap
            for s in seeds:
                rg_ = np.random.default_rng(s)
                cfg = dict(L.REQUIRED_CFG, epochs=60, dataset=L.MODALITIES[m], seed=s, label=lab, out_prefix="x",
                           tsc_dir="/x")
                prov = dict(head="h", status="", sha256={"train_cnn_mock.py": "a"}, config=cfg,
                            manifest_sha256="f" * 64,
                            inputs_sha256={"dataset": m, "catalog": "c", "tsc_snap99": "t9", "tsc_snap91": "t1"})
                dd = dict(sample_halo=np.repeat(grp, R), sample_group=np.repeat(grp, R),
                          sample_snap=np.full(NP * R, 99), sample_realization=np.tile(np.arange(R), NP),
                          group_id=grp, snap=np.full(NP, 99), y_sample=np.repeat(yv, R).astype(np.float32),
                          tsc=yv.astype(np.float32), fold_of=np.repeat(grp % F, R).astype(np.int16),
                          sel_groups_flat=np.array([PAR[((f + 1) % F) + 5 * j] for f in range(F) for j in range(2)]),
                          sel_groups_fold=np.repeat(np.arange(F), 2), tsc_cap=np.array(4.0),
                          fold_best_epoch=np.full(F, 30), fold_sel_score=np.full(F, 0.3),
                          select_criterion=np.array("inner:single:r2"), truth_pseudo=pse, truth_tsc_capped=cap,
                          truth_tsc_raw=raw, truth_tcoll_capped=cap, truth_tcoll_raw=raw, label_name=np.array(lab),
                          sim_logm=np.linspace(14.5, 15.3, NP),
                          oof_sample=(np.repeat(yv, R) + rg_.normal(0, 0.3, NP * R)).astype(np.float32),
                          oof_single=(np.repeat(yv, R)[:, None] + rg_.normal(0, 0.3, (NP * R, P))).astype(np.float32),
                          provenance=np.array(json.dumps(prov)), provenance_changed_during_run=np.array(False))
                np.savez(f"{d}/p/" + L.pattern(m, lab).format(s), **dd)
    return ["--expect-manifest", "f" * 64, "--pred-dir", f"{d}/p", "--mergers", f"{d}/m.h5", "--catalog", f"{d}/cat.h5"]


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
base = build(d, EV, L.PRODUCTION_SEEDS)

print("  -- production mode (5 seeds, 4000 draws, min half 20)")
t0 = time.time()
H.run(base + ["--out", f"{d}/prod.npz"])
print(f"    production-mode synthetic run wall time {time.time() - t0:.1f} s")
sv = np.load(f"{d}/prod.npz", allow_pickle=True)
g = json.loads(str(sv["gate"]))
assert str(sv["mode"]) == "production" and g["mode"] == "production"
assert g["overall"] in ("discuss B3", "stop (no association resolved / unavailable)")
assert str(sv["exclusion"][0]) == "invalid prev ratio" and not sv["primary"][0]
assert (sv["exclusion"] == "no recorded event").sum() == 3
prov = json.loads(str(sv["provenance"]))
assert len(prov["predictions"]) == 20 and prov["constants"]["b2_knots"] == [0.5, 1.0, 2.0]
assert "round 44" in prov["constants"]["b2_amendment"] and prov["mode"] == "production"
parv, prim = sv["parent"], sv["primary"]
b1 = json.loads(str(sv["b1"]))
for k_, r_ in b1.items():
    assert set(r_) == set(r), k_
pr = b1["primary"]
assert pr["ids"] == parv[prim].tolist() and len(pr["coef"]) == len(pr["names"]) == pr["rank"] == 12
assert len(pr["residual_feature"]) == pr["n"] and len(pr["boot_betas"]) == pr["draws_valid"] == 4000
assert b1["adjusted"]["status"] == "ok" and b1["adjusted"]["ids"] == parv[sv["adj_mask"]].tolist()
ck = json.loads(str(sv["checks"]))
assert ck["algebra"]["status"] == "ok" and ck["residual_identity"]["status"] == "ok" and "label_context" in ck
b2s = json.loads(str(sv["b2"]))
assert b2s["parents"] == parv[prim].tolist() and len(b2s["pred0"]) == int(prim.sum()) and len(b2s["directions"]) == 10
spl = json.loads(str(sv["splits"]))
ref_spl = H.make_splits(parv[prim].tolist(), {p_: p_ % 5 for p_ in PAR})
assert all(spl[str(f)][0] == ref_spl[f][0].tolist() and spl[str(f)][1] == ref_spl[f][1].tolist() for f in range(5))
Tsv = sv["T_last"][prim]
r3 = b2s["directions"][5]
if r3["status"] == "ok":
    ix = np.array([b2s["parents"].index(p_) for p_ in r3["eval_ids"]])
    mp_sv = np.array([f_["M_prev_1.0"] for f_ in json.loads(str(sv["feats"]))])[prim]
    assert np.allclose(np.array(b2s["pred1"])[ix],
                       np.column_stack([H.pl_basis(Tsv[ix], H.B2_KNOTS)[0], mp_sv[ix]]) @ np.array(r3["coef1"]))
iso = json.loads(str(sv["b2_iso"]))
assert all(v["parents"] == parv[sv["iso_mask"]].tolist() and len(v["pred"]) == v["n"] and len(v["directions"]) == 10
           for v in iso.values())
print(f"    saved gate {g}; B1 coefficient vectors/ranks/residual support/draws, checks, B2 per-direction fits and "
      f"draws, isotonic predictions and thresholds all present and aligned with the saved parent order")

print("  -- smoke mode and argument guards")
H.run(base + ["--out", f"{d}/smoke.npz", "--mode", "smoke", "--seeds", "42,43", "--boot", "100"])
ss = np.load(f"{d}/smoke.npz", allow_pickle=True)
gs = json.loads(str(ss["gate"]))
assert str(ss["mode"]) == "smoke" and gs["overall"] == "smoke run: components exercised only; no B3 nomination"
print(f"    smoke artifact classified: mode {str(ss['mode'])}, overall '{gs['overall']}'")
for extra_, what in ((["--seeds", "42,43"], "production with two seeds"),
                     (["--boot", "100"], "production with 100 draws"),
                     (["--min-half", "10"], "production with min half 10"),
                     (["--mode", "smoke", "--seeds", "42,42"], "duplicate seeds"),
                     (["--mode", "smoke", "--seeds", ""], "empty seed list"),
                     (["--mode", "smoke", "--seeds", "42,x"], "non-integer seed"),
                     (["--mode", "smoke", "--seeds", "42", "--boot", "-1"], "negative draws"),
                     (["--mode", "smoke", "--seeds", "42", "--min-half", "1"], "min half 1")):
    expect(H.run, base + ["--out", f"{d}/never.npz"] + extra_, what=what)
expect(H.run, base + ["--out", f"{d}/prod"], what="--out without .npz (numpy would append the suffix)")
expect(H.run, base + ["--out", f"{d}/prod.npz"], what="existing output (no overwrite)")
assert not os.path.exists(f"{d}/never.npz")

print("  -- history-collinear primary feature (one event per parent -> M_prev identically 0)")
EVc = [(p, t99 - erng.uniform(0.0, 3.9), erng.uniform(0.01, 0.6)) for p in PAR]
dc = f"{TMP}/c"
bc = build(dc, EVc, [42])
H.run(bc + ["--out", f"{dc}/o.npz", "--mode", "smoke", "--seeds", "42", "--boot", "20"])
sc = np.load(f"{dc}/o.npz", allow_pickle=True)
b1c, ckc, gc = (json.loads(str(sc[k])) for k in ("b1", "checks", "gate"))
assert b1c["primary"]["status"] == "rank-deficient" and ckc["algebra"]["status"].startswith("unavailable")
assert ckc["residual_identity"]["status"].startswith("unavailable") and gc["i"] == gc["ii"] == gc["iii"] == "unavailable"
print(f"    artifact saved; primary '{b1c['primary']['status']}'; checks '{ckc['algebra']['status']}'; gate {gc}")

print("  -- empty eligible cohort (no recorded events)")
de = f"{TMP}/e"
be = build(de, [(9999, t99 - 1.0, 0.2)], [42])
H.run(be + ["--out", f"{de}/o.npz", "--mode", "smoke", "--seeds", "42", "--boot", "20"])
se_ = np.load(f"{de}/o.npz", allow_pickle=True)
b1e, b2e, ge = (json.loads(str(se_[k])) for k in ("b1", "b2", "gate"))
assert not se_["primary"].any() and b1e["primary"]["status"] == "unavailable: empty cohort"
assert "empty cohort" in b2e["interval_status"] and ge["i"] == ge["ii"] == ge["iii"] == "unavailable"
print(f"    artifact saved; primary '{b1e['primary']['status']}'; B2 '{b2e['interval_status']}'; gate {ge}")
print("ALL HISTORY-DIAGNOSTIC FIXTURE CHECKS PASSED")
