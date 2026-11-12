#!/usr/bin/env python3
"""Synthetic fixtures for paper/scripts/regen_stale.py and extract_phase_syn.py
(review rounds 48-49). No project data are read."""
import importlib.util
import os
import tempfile

import numpy as np

spec = importlib.util.spec_from_file_location("regen_stale", "paper/scripts/regen_stale.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
import build_radio_cells as brc
import extract_phase_syn as E
from forward_model_lotss import rebin_flux

TMP = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "."))


def raises(exc, f, *a, what="", **k):
    try:
        f(*a, **k)
    except exc as e:
        print(f"    rejected ({what}): {str(e)[:110]}")
        return
    raise SystemExit(f"NOT rejected: {what}")


print("== W1 partitions and calibration")
from sklearn.model_selection import KFold
n = 352
for s in (42, 46):
    fold = R.reconstruct_folds(n, s)
    ref = np.full(n, -1)
    for f, (_, te) in enumerate(KFold(5, shuffle=True, random_state=s).split(np.arange(n))):
        ref[te] = f
    assert np.array_equal(fold, ref) and (fold >= 0).all()
print("    group_split reconstruction equals the plain KFold row split (one row per parent)")
ids = np.arange(1000, 1000 + n)
fold = R.reconstruct_folds(n, 42)
A, B = R.calib_halves(ids[fold == 2], 42, 2)
A2, B2 = R.calib_halves(ids[fold == 2][::-1], 42, 2)
assert np.array_equal(A, A2) and np.array_equal(B, B2) and not set(A) & set(B)
assert sorted(np.r_[A, B]) == sorted(ids[fold == 2]) and abs(len(A) - len(B)) <= 1
print(f"    halves of one original fold: order-independent, disjoint, exact cover ({len(A)}/{len(B)})")
rng = np.random.default_rng(0)
X5 = rng.normal(size=(n, 5))
y = X5 @ np.array([0.5, -0.3, 0.2, 0.0, 0.1]) + rng.normal(0, 1.0, n)
good = y + rng.normal(0, 0.3, n)
noise = rng.normal(size=n)
pred, full, dirs = R.within_fold_calibration(ids, y, fold, {"s": X5, "s+good": np.column_stack([X5, good]),
                                                            "s+noise": np.column_stack([X5, noise])}, 42)
assert all(full.values()) and len(dirs) == 10
for d in dirs:
    assert set(d["fit_ids"]).isdisjoint(d["eval_ids"])
    assert len({int(fold[ids == i][0]) for i in d["fit_ids"] + d["eval_ids"]}) == 1
inc_g, inc_n = R.r2(y, pred["s+good"]) - R.r2(y, pred["s"]), R.r2(y, pred["s+noise"]) - R.r2(y, pred["s"])
assert inc_g > 0.3 and abs(inc_n) < 0.06
print(f"    planted informative output: increment {inc_g:+.3f}; pure-noise output: {inc_n:+.3f}; complete "
      f"coverage; every fit/eval pair inside one original fold")
p2, full2, d2 = R.within_fold_calibration(ids, y, fold, {"s": X5}, 42, min_n=40)
assert not full2["s"] and all(d["status"].startswith("unavailable") for d in d2)
print("    halves below the declared minimum -> arm unavailable")
p_a, al_a, m_a = R.ridge_fit_predict(X5[:50], y[:50], X5[50:60], 1)
p_b, al_b, m_b = R.ridge_fit_predict(X5[:50], y[:50], X5[60:90] * 7, 1)
assert al_a == al_b and np.allclose(m_a["coef"], m_b["coef"])
print("    ridge alpha/coefficients depend only on the fitting rows")

print("== descriptive statistics")
c = rng.normal(size=4000)
a_ = c + rng.normal(size=4000)
b_ = c + rng.normal(size=4000)
ps = R.partial_spearman(a_, b_, c)
assert abs(ps) < 0.06 and abs(R.partial_spearman(a_, a_, c) - 1) < 1e-12
print(f"    partial rank correlation: conditionally independent {ps:+.3f}; identical 1")
m1 = rng.random((16, 16))
m2 = np.zeros((16, 16)); m2[0, 0] = 1
m3 = np.zeros((16, 16)); m3[5, 5] = 2
assert R.tv_distance(m1, m1 * 37) < 1e-12 and abs(R.tv_distance(m2, m3) - 1) < 1e-12
assert np.isnan(R.tv_distance(m1, np.zeros((16, 16))))
print("    TV distance: scale-invariant 0, disjoint 1, zero-total map unavailable")
one = np.ones((512, 512))
assert abs(R.box_sum(one, 100.3) - 200.6 ** 2) < 1e-9
img = rng.random((512, 512))
for h in (37.25, 100.3, 211.9):
    assert abs(R.box_sum(img, h) - rebin_flux(img, h).sum()) < 1e-8 * img.sum()
print("    fractional-overlap box sum: exact for a constant map; agrees with the forward model's rebin_flux")
lp = np.repeat(rng.normal(size=(5, 1, 3)), 3, axis=1)
assert R.collapse_realisations(lp)[0].shape == (5, 3)
lp2 = lp.copy(); lp2[2, 1, 0] += 1e-3
raises(R.Fail, R.collapse_realisations, lp2, what="box power differing across realisations")
yb = np.array([0.1, 0.2, 0.6, 3.5, 4.0, 4.0])
bins = R.residual_bins(yb, yb + 1)
assert bins[-1]["n"] == 3 and bins[2]["status"].startswith("unavailable") and bins[0]["n"] == 2
raises(R.Fail, R.residual_bins, np.array([4.5]), np.array([4.0]), what="label above the 4 Gyr top edge")
print("    residual bins: 4 Gyr in the closed last bin; empty bin unavailable")
dm = np.array([[1.0, np.nan, 3.0], [np.nan] * 3, [2.0, 2.0, 2.0]])
hm, nw = R.halo_summary(dm)
assert hm[0] == 2.0 and np.isnan(hm[1]) and hm[2] == 2.0 and list(nw) == [2, 0, 3]
bs = R.boot_spearman(np.arange(30.0), np.arange(30.0) ** 2, n=200)
assert np.isclose(bs["rho"], 1.0) and np.isclose(bs["lo"], 1.0) and bs["interval_status"] == "ok"
bc = R.boot_spearman(np.ones(30), np.arange(30.0), n=50)
assert bc["interval_status"].startswith("unavailable") and bc["invalid_draws"] == 50
print("    halo summary excludes zero-peak halos; bootstrap counts invalid draws -> interval unavailable")
assert R.norm_key(b"RXC J1217.6+0339") == R.norm_key("RXCJ1217.6+0339")

print("== W3 per-parent cell statistics (extract_phase_syn.parent_stats)")
w = np.array([1.0, 0.0, 2.0, 4.0, 3.0, np.nan, 5.0])
nH = np.array([1e-5, 2e-5, 1e-4, 1e-3, 1e-6, 1e-5, 1e-9])   # 1e-4 kept; 1e-9 outside the n_H histogram range
T = np.array([1.0, 2.0, 0.05, 0.01, 3.0, 1.0, 1.0])
M = np.array([2.0, 3.0, 4.0, 30.0, 2.5, 2.0, 5.0])
st = E.parent_stats(w, T, nH, M)
assert st["n_invalid"] == 1 and st["n_valid"] == 6 and st["n_zero_w"] == 1 and st["n_out_of_range"] == 1
assert abs(st["H_n"].sum() + st["n_out_of_range"] / st["n_valid"] - 1) < 1e-12
assert abs(st["H_w"].sum() + st["w_out_of_range"] / st["w_total"] - 1) < 1e-12
assert abs(st["frac_above"] - 4.0 / 15.0) < 1e-12 and abs(st["frac_cold"] - 6.0 / 15.0) < 1e-12
assert abs(st["top_cut"] - 5.0 / 11.0) < 1e-12 and st["cut_status"] == "ok"
v = np.array([1.0, 0.0, 2.0, 4.0, 3.0, 5.0]); Mv = np.array([2.0, 3.0, 4.0, 30.0, 2.5, 5.0])
keep = np.array([1, 1, 1, 0, 1, 1], bool)
w14 = v * (1.4 / 0.144) ** (-0.5 * brc.s_of_M(Mv))
assert abs(st["ratio_cut"] - v[keep].sum() / w14[keep].sum()) < 1e-12
print("    zero-weight cells counted, invalid excluded, out-of-range accounted (histograms + outside = 1), "
      "n_H = 1e-4 kept, cold fraction, brightest-cell share and frequency ratio match hand values")
st2 = E.parent_stats(np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([1e-3, 1e-2]), np.array([3.0, 4.0]))
assert st2["cut_status"] == "empty cut" and np.isnan(st2["top_cut"]) and np.isnan(st2["ratio_cut"])
print("    no cells below the cut -> 'empty cut', statistics unavailable")

print("== run directory and manifest")
d = os.path.join(TMP, "run")
R.TASKS["ok_t"] = lambda run, a: run.record("ok_t", {"x": 1}, {})
R.TASKS["bad_t"] = lambda run, a: (_ for _ in ()).throw(R.Fail("planted"))
rc = R.main(["--run-dir", d, "--tasks", "ok_t,bad_t"])
import json
man = json.load(open(os.path.join(d, "manifest.json")))
assert rc == 1 and not man["complete"] and man["status"]["ok_t"] == "ok" and man["status"]["bad_t"].startswith("failed")
raises(SystemExit, R.main, ["--run-dir", d, "--tasks", "ok_t"], what="existing run directory")
raises(SystemExit, R.main, ["--run-dir", d + "2", "--tasks", "ok_t,ok_t"], what="duplicate task")
rc2 = R.main(["--run-dir", d + "3", "--tasks", "ok_t"])
assert rc2 == 0 and json.load(open(os.path.join(d + "3", "manifest.json")))["complete"]
print("    failed task -> incomplete manifest and nonzero exit; reuse and duplicate tasks rejected")
print("ALL REGEN FIXTURE CHECKS PASSED")
