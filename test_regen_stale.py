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
assert R.collapse_realisations(lp, n=5)[0].shape == (5, 3)
lp2 = lp.copy(); lp2[2, 1, 0] += 1e-3
raises(R.Fail, R.collapse_realisations, lp2, what="box power differing across realisations")
lp3 = lp.copy(); lp3[1, 2, 1] = np.nan
raises(R.Fail, R.collapse_realisations, lp3, what="NaN outside realisation zero")
raises(R.Fail, R.collapse_realisations, lp[:, :2], n=5, what="wrong realisation count")
yc = np.array([4.0, 3.9999, 4.0, 1.0], dtype=np.float32)
cs = R.cap_stats(yc, np.array([3.0, 3.0, 2.0, 1.0]))
assert cs["n"] == 2 and cs["median_residual"] == -1.5
print("    exact-cap statistic counts only stored 4.0 (3.9999 excluded)")
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
R.TASKS["ok_t"] = lambda run, a: run.record("ok_t", {"x": 1})
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
print("== joins, agreement and clean-source validators")
assert np.isnan(R.legacy_join([3, 1], [1, 2, 3], [0.5, 1.0, np.nan], "x")[0])
raises(R.Fail, R.legacy_join, [1, 2], [1, 1, 2], [0.1, 0.2, 0.3], "x", what="duplicate legacy keys")
raises(R.Fail, R.legacy_join, [1, 4], [1, 2, 3], [0.1, 0.2, 0.3], "x", what="parent missing from a legacy source")
print("    stored NaN preserved after a successful join (distinct from a failed join)")
hid4 = np.array([10, 11, 12])
cat = dict(halo_id=hid4, d_max=np.array([[1.0, np.nan, 2.0]] * 3), n_relics=np.ones((3, 3), int),
           merger_tsc=np.array([0.5, np.nan, 1.0]), mass_ratio=np.array([0.2, 0.05, np.nan]))
cat["n_relics"][:, 1] = 0
ag = R.relic_agreement(cat, cat["d_max"].copy(), cat["n_relics"].copy(), cat["merger_tsc"].copy(),
                       cat["mass_ratio"].copy(), cat["mass_ratio"].copy(), hid4)
assert all(ag.values())
dm2 = cat["d_max"].copy(); dm2[0, 0] = 1.1
mr2 = cat["mass_ratio"].copy(); mr2[0] = 0.3
assert not R.relic_agreement(cat, dm2, cat["n_relics"], cat["merger_tsc"], cat["mass_ratio"], cat["mass_ratio"],
                             hid4)["d_max_values"]
assert not R.relic_agreement(cat, cat["d_max"], cat["n_relics"], cat["merger_tsc"], cat["mass_ratio"], mr2,
                             hid4)["legacy_vs_current_mass_ratio"]
print("    relic agreement flags a changed peak distance and a legacy/current mass-ratio mismatch")
yv = np.array([0.5, 1.0, 2.0])
R.check_clean_arrays(hid4, yv, yv.astype(np.float32), np.array([14.5, 14.6, 14.7]),
                     np.array([14.5, 14.6, 14.7], np.float32), [10, 11, 12, 13])
raises(R.Fail, R.check_clean_arrays, hid4, yv, yv + 0.01, np.ones(3), np.ones(3), [10, 11, 12],
       what="shape-dataset labels differ")
raises(R.Fail, R.check_clean_arrays, hid4, yv, yv, np.ones(3), np.ones(3) + 1e-3, [10, 11, 12],
       what="catalogue vs dataset mass mismatch")
raises(R.Fail, R.check_clean_arrays, hid4, yv, yv, np.ones(3), np.ones(3), [10, 11, 12, 12],
       what="duplicate catalogue ids")

print("== synthetic production world (4 parents): validators and real task paths")
import h5py
import json
import shutil
import detect_relics_radio as DR
W = os.path.join(TMP, "world")
os.makedirs(W)
NP_ = 4
R.PROTO["n_clusters"] = NP_
HID = np.array([101, 205, 307, 409])
Y = np.array([0.3, 1.2, 2.5, 4.0], np.float32)


def write_maps(path, img, cells_dir, size):
    with h5py.File(path, "w") as f:
        f["images"] = img.astype(np.float32)
        f["meta/halo_id"] = HID
        f["meta/halo_id_snap"] = HID
        f["meta/r500c_kpc"] = np.full(NP_, 1000.0, np.float32)
        f["meta/log_m500"] = np.array([14.5, 14.7, 14.9, 15.1], np.float32)
        f["meta/mass_ratio"] = np.array([0.2, 0.05, 0.5, 0.3], np.float32)
        f["labels/pseudo_tsc"] = Y
        for k, v in dict(R.PROTO, img_size=size, cells_dir=cells_dir).items():
            f.attrs[k] = np.array(v, dtype=object) if isinstance(v, list) else v


raises(R.Fail, lambda: R.validate_maps(h5py.File(os.path.join(W, "bad.h5"), "w"), 128), what="map file without protocol attributes")
mp = lambda n: os.path.join(W, n)
R.IN.update(ds128=mp("d128.h5"), ds1400=mp("d1400.h5"), ratio_table=mp("ratio.npz"), cells144=mp("c144"),
            cells1400=mp("c1400"), ds512=mp("d512.h5"), relic_cat=mp("relic.h5"), tsc_legacy=mp("tsc.h5"),
            ds_legacy=mp("dl.h5"))
img = rng.random((NP_, 3, 128, 128))
write_maps(R.IN["ds128"], np.arcsinh(img), R.IN["cells144"], 128)
write_maps(R.IN["ds1400"], np.arcsinh(img * rng.uniform(0.8, 1.2, img.shape)), R.IN["cells1400"], 128)
with h5py.File(R.IN["ds128"]) as f:
    R.validate_maps(f, 128)
    raises(R.Fail, R.validate_maps, f, 512, what="wrong declared image size")
np.savez(R.IN["ratio_table"], x=np.arange(3))
os.makedirs(R.IN["cells144"]); os.makedirs(R.IN["cells1400"])
ratio = np.array([12.0, 13.0, 11.0, 12.5])
for h, rc in zip(HID, ratio):
    w144 = rng.random(50) + 0.1
    np.savez(os.path.join(R.IN["cells144"], f"cells_FOF{h}_sub{h}.npz"), w=w144)
    w14 = w144 * (w144.sum() / (rc * 16.48 / 206.0)) / w144.sum()       # sums give ratio rc * 16.48/206
    np.savez(os.path.join(R.IN["cells1400"], f"cells_FOF{h}_sub{h}.npz"),
             w=w144 / (rc * 16.48 / 206.0))


def write_phase(path, ids=HID, prod=True, limit=0, nu_hi=1.4, table=None, manifest=True, rat=ratio):
    n = len(ids)
    cfg = dict(R.PHASE_CONFIG, nu_hi_ghz=nu_hi)
    pv = dict(config=cfg, production_complete=prod, limit=limit, n_check_failures=0, n_parents=n,
              ratio_table=table or PU.sha(R.IN["ratio_table"]), cells_cut={})
    arr = {k: rng.random(n) for k in ("frac_above", "frac_cold", "top_all", "top_cut", "mach_all", "mach_cut",
                                      "ratio_all", "w_out_of_range")}
    np.savez(path, halo_id=np.array(ids), H_n=rng.random((n, 70, 50)) / 3500, H_w=rng.random((n, 70, 50)) / 3500,
             nh_edges=np.linspace(-7, 0, 71), t_edges=np.linspace(-3.5, 1.5, 51), ratio_cut=np.array(rat[:n]),
             n_cells=np.full(n, 50), n_invalid=np.zeros(n, int), n_zero_w=np.zeros(n, int),
             n_out_of_range=np.zeros(n, int), w_total=np.ones(n), cut_status=np.array(["ok"] * n),
             provenance=np.array(json.dumps(pv)), **arr)
    if manifest:
        json.dump(dict(status="complete", artifact=path, artifact_sha256=PU.sha(path)), open(path + ".manifest.json", "w"))
    return path


def write_bman(nu=1.4, sha=None):
    p = R.IN["ds1400"] + ".manifest.json"
    if os.path.exists(p):
        os.remove(p)
    json.dump(dict(status="complete", kind="cells_1400", prescription=dict(R.B_PRESCRIPTION, nu_ghz=nu),
                   dataset=dict(sha256=sha or PU.sha(R.IN["ds1400"]))), open(p, "w"))


import prov_utils as PU
good = write_phase(mp("phase_good.npz"))
d, pv, _ = R.validate_phase(good, HID[::-1], R.IN["ratio_table"])
assert list(d["halo_id"]) == list(HID[::-1])
print("    complete job-A product accepted and reindexed to the adopted parent order")
for nm, kw, what in (("p1", dict(ids=HID[:3]), "partial parent set"),
                     ("p2", dict(ids=np.array([101, 101, 307, 409])), "duplicate parents"),
                     ("p3", dict(ids=np.array([101, 205, 307, 999])), "wrong parent"),
                     ("p4", dict(prod=False), "not production-complete"),
                     ("p5", dict(limit=3), "limited smoke artifact"),
                     ("p6", dict(nu_hi=1.5), "configuration mismatch"),
                     ("p7", dict(table="0" * 64), "different ratio table"),
                     ("p8", dict(manifest=False), "no success manifest")):
    raises(R.Fail, R.validate_phase, write_phase(mp(nm + ".npz"), **kw), HID, R.IN["ratio_table"], what=what)
tam = write_phase(mp("p9.npz"))
with open(tam, "ab") as f:
    f.write(b"x")
raises(R.Fail, R.validate_phase, tam, HID, R.IN["ratio_table"], what="artifact changed after its manifest")
raises(R.Fail, R.validate_b_manifest, R.IN["ds1400"], what="job B manifest missing")
write_bman(nu=0.144)
raises(R.Fail, R.validate_b_manifest, R.IN["ds1400"], what="job B prescription mismatch")
write_bman(sha="0" * 64)
raises(R.Fail, R.validate_b_manifest, R.IN["ds1400"], what="job B dataset hash mismatch")
write_bman()
R.validate_b_manifest(R.IN["ds1400"])
st = lambda d_: json.load(open(os.path.join(d_, "manifest.json")))["status"]
R.main(["--run-dir", mp("run_w3_ok"), "--tasks", "w3", "--phase-npz", good])
assert st(mp("run_w3_ok"))["w3"] == "ok", st(mp("run_w3_ok"))
assert os.path.exists(mp("run_w3_ok/w3_arrays.npz")) and os.path.exists(mp("run_w3_ok/fig_phase.pdf"))
R.main(["--run-dir", mp("run_w3_partial"), "--tasks", "w3", "--phase-npz", mp("p1.npz")])
assert st(mp("run_w3_partial"))["w3"].startswith("failed check")
bad_r = write_phase(mp("p10.npz"), rat=ratio * 1.01)
R.main(["--run-dir", mp("run_w3_b"), "--tasks", "w3", "--phase-npz", bad_r])
assert "do not reproduce" in st(mp("run_w3_b"))["w3"]
print("    real task w3: ok on the consistent world; failed status for a partial job-A product and for job-B "
      "cells that do not reproduce job A's frequency ratio")
# W9 world: blobs at 1 R500 so peaks exist; catalogue produced by the same detector
im512 = np.zeros((NP_, 3, 512, 512))
yy, xx = np.mgrid[:512, :512]
for i in range(NP_):
    for p in range(3):
        cx = 256 + 64 * np.cos(i + p)
        cy = 256 + 64 * np.sin(i + p)
        im512[i, p] = 3 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 50) + 5 * np.exp(-((xx - 256) ** 2 + (yy - 256) ** 2) / 20)
write_maps(R.IN["ds512"], im512, R.IN["cells144"], 512)
MT = np.array([0.4, 1.1, np.nan, 1.8])


def write_w9(tsc_ids=HID, alter=False):
    for pth in (R.IN["relic_cat"], R.IN["tsc_legacy"], R.IN["ds_legacy"]):
        if os.path.exists(pth):
            os.remove(pth)
    dm, nr = np.full((NP_, 3), np.nan), np.zeros((NP_, 3), int)
    for i in range(NP_):
        for p in range(3):
            pk = DR.detect_in_image(im512[i, p].astype(np.float32), 1000.0, 4000.0)
            if pk:
                nr[i, p], dm[i, p] = len(pk), max(r_ for r_, _ in pk)
    if alter:
        dm[0, 0] *= 1.1
    with h5py.File(R.IN["relic_cat"], "w") as f:
        for k, v in dict(halo_id=HID, d_max=dm, n_relics=nr, merger_tsc=MT,
                         mass_ratio=np.array([0.2, 0.05, 0.5, 0.3], np.float32)).items():
            f[k] = v
        f.attrs.update(dict(source=R.IN["ds512"], r_min_frac=0.3, r_max_frac=2.0, peak_thresh_frac=0.25,
                            min_sep_kpc=600.0))
    with h5py.File(R.IN["tsc_legacy"], "w") as f:
        f["halo_id"] = np.array(tsc_ids)
        f["tsc_gyr"] = np.r_[MT, np.zeros(len(tsc_ids) - NP_)]
    with h5py.File(R.IN["ds_legacy"], "w") as f:
        f["meta/halo_id"] = HID
        f["meta/mass_ratio"] = np.array([0.2, 0.05, 0.5, 0.3], np.float32)


write_w9()
R.main(["--run-dir", mp("run_w9_ok"), "--tasks", "w9"])
assert st(mp("run_w9_ok"))["w9"] == "ok", st(mp("run_w9_ok"))
write_w9(alter=True)
R.main(["--run-dir", mp("run_w9_alt"), "--tasks", "w9"])
assert st(mp("run_w9_alt"))["w9"].startswith("failed check: review required")
assert os.path.exists(mp("run_w9_alt/w9_arrays.npz")) and os.path.exists(mp("run_w9_alt/w9.json"))
write_w9(tsc_ids=np.array([101, 205, 307, 307, 409]))
R.main(["--run-dir", mp("run_w9_dup"), "--tasks", "w9"])
assert "duplicate keys" in st(mp("run_w9_dup"))["w9"]
write_w9(tsc_ids=np.array([101, 205, 307, 999]))
R.main(["--run-dir", mp("run_w9_miss"), "--tasks", "w9"])
assert "missing from the source" in st(mp("run_w9_miss"))["w9"]
print("    real task w9: ok when redetection reproduces the catalogue; 'review required' (diagnostics saved) on a "
      "changed peak; failed status for duplicate or missing legacy join keys")
print("ALL REGEN FIXTURE CHECKS PASSED")
