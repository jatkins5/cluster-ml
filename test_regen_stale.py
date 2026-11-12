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
R.Run.git_dirty = lambda self: ""          # fixture-only: treat helpers as committed
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
                             hid4)["d_max"]
assert not R.relic_agreement(cat, cat["d_max"], cat["n_relics"], cat["merger_tsc"], cat["mass_ratio"], mr2,
                             hid4)["legacy_vs_current_mass_ratio"]
dm3 = cat["d_max"].copy(); dm3[1, 0] = np.inf
ag3 = R.relic_agreement(cat, cat["d_max"], cat["n_relics"], cat["merger_tsc"], cat["mass_ratio"], cat["mass_ratio"], hid4)
ag4 = R.relic_agreement(dict(cat, d_max=dm3), cat["d_max"], cat["n_relics"], cat["merger_tsc"], cat["mass_ratio"],
                        cat["mass_ratio"], hid4)
assert all(ag3.values()) and not ag4["no_infinities"] and not ag4["d_max"]
assert not R.relic_agreement(cat, cat["d_max"][:2], cat["n_relics"], cat["merger_tsc"], cat["mass_ratio"],
                             cat["mass_ratio"], hid4)["shapes"]
print("    relic agreement flags a changed peak distance, a legacy/current mass-ratio mismatch, an infinite "
      "catalogue distance (finite-mask mismatch) and a shape mismatch")
yv = np.array([0.5, 1.0, 2.0])
R.check_clean_arrays(hid4, yv, yv.astype(np.float32), np.array([14.5, 14.6, 14.7]),
                     np.array([14.5, 14.6, 14.7], np.float32), [10, 11, 12, 13])
raises(R.Fail, R.check_clean_arrays, hid4, yv, yv + 0.01, np.ones(3), np.ones(3), [10, 11, 12],
       what="shape-dataset labels differ")
raises(R.Fail, R.check_clean_arrays, hid4, yv, yv, np.ones(3), np.ones(3) + 1e-3, [10, 11, 12],
       what="catalogue vs dataset mass mismatch")
raises(R.Fail, R.check_clean_arrays, hid4, yv, yv, np.ones(3), np.ones(3), [10, 11, 12, 12],
       what="duplicate catalogue ids")

print("== synthetic production world (4 parents): producers, validators and real task paths")
import h5py
import json
import detect_relics_radio as DR
import manifest_cells_1400 as MB
import prov_utils as PU
W = os.path.join(TMP, "world")
os.makedirs(W)
NP_ = 4
R.PROTO["n_clusters"] = NP_
MB.EXPECT_ATTRS["n_clusters"] = NP_
HID = np.array([101, 205, 307, 409])
Y = np.array([0.3, 1.2, 2.5, 4.0], np.float32)
mp = lambda n: os.path.join(W, n)
st = lambda d_: json.load(open(os.path.join(d_, "manifest.json")))


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


raises(R.Fail, lambda: R.validate_maps(h5py.File(mp("bad.h5"), "w"), 128), what="map file without protocol attributes")
R.IN.update(ds128=mp("d128.h5"), ds1400=mp("d1400.h5"), ratio_table=mp("ratio.npz"), cells144=mp("c144"),
            cells1400=mp("c1400"), ds512=mp("d512.h5"), relic_cat=mp("relic.h5"), tsc_legacy=mp("tsc.h5"),
            ds_legacy=mp("dl.h5"))
img = rng.random((NP_, 3, 128, 128))
write_maps(R.IN["ds128"], np.arcsinh(img), R.IN["cells144"], 128)
with h5py.File(R.IN["ds128"]) as f:
    R.validate_maps(f, 128)
    raises(R.Fail, R.validate_maps, f, 512, what="wrong declared image size")
np.savez(R.IN["ratio_table"], x=np.arange(3))
for dname in ("radio", "cut", "c144", "call", "gc", "srcids"):
    os.makedirs(mp(dname))


def cells_of(h):
    g = np.random.default_rng(int(h))
    n = 60
    w = g.random(n) + 0.1
    w[0] = 0.0
    nH = 10 ** g.uniform(-6, -3, n)
    T = 10 ** g.uniform(-2, 0.5, n)
    M = g.uniform(1.4, 6.0, n)
    return np.zeros((n, 3)), w, np.ones(n), T, nH, M


for h in HID:
    np.savez(mp(f"radio/radio_FOF{h}_sub{h}.npz"), x=0)
    np.savez(mp(f"srcids/radio_FOF{h}_sub{h}.npz"), x=0)
    open(mp(f"cut/cutout_sub{h}_FOF{h}.hdf5"), "wb").write(os.urandom(1000))
    _, w, _, T, nH, M = cells_of(h)
    keep = nH <= 1e-4
    np.savez(mp(f"c144/cells_FOF{h}_sub{h}.npz"), w=w[keep] / 206.0)
    np.savez(mp(f"call/cells_FOF{h}_sub{h}.npz"), w=w / 206.0)
open(mp("gc/fof_subhalo_tab_099.0.hdf5"), "wb").write(os.urandom(500))
for nm in ("pkl.pkl", "cat.h5"):
    open(mp(nm), "wb").write(os.urandom(100))
E.brc.make_ratio = lambda p: None
E.brc.generate = lambda cut, interp, nu_ghz, phase, model: cells_of(int(re.search(r"_FOF(\d+)\.hdf5", cut).group(1)))
import re
A_ARGS = lambda out, extra=(): ["--out", out, "--radio-dir", mp("radio"), "--cutout-dir", mp("cut"),
                                "--ratio-table", R.IN["ratio_table"], "--cells-cut", mp("c144"),
                                "--cells-all", mp("call"), "--cohort", R.IN["ds128"], *extra]
E.main(A_ARGS(mp("phase_good.npz")))
assert os.path.exists(mp("phase_good.npz.manifest.json"))
print("    job A producer (real main, stubbed cell generator): production artifact and success manifest written")
raises(SystemExit, E.main, A_ARGS(mp("phase_lim.npz"), ["--limit", "2"]), what="job A limited (smoke) run")
assert os.path.exists(mp("phase_lim.npz")) and not os.path.exists(mp("phase_lim.npz.manifest.json"))
orig_gen = E.brc.generate
def gen_touch(cut, *a_, **k_):
    np.savez(mp("ratio2.npz"), x=np.arange(4))      # the run's ratio table changes mid-run
    return orig_gen(cut, *a_, **k_)
E.brc.generate = gen_touch
np.savez(mp("ratio2.npz"), x=np.arange(3))
raises(SystemExit, E.main, ["--out", mp("phase_chg.npz"), "--radio-dir", mp("radio"),
       "--cutout-dir", mp("cut"), "--ratio-table", mp("ratio2.npz"), "--cells-cut", mp("c144"), "--cells-all",
       mp("call"), "--cohort", R.IN["ds128"]], what="job A source changed during the run")
assert not os.path.exists(mp("phase_chg.npz.manifest.json"))
E.brc.generate = orig_gen
# job B two-stage producer
B_ARGS = lambda stage, table=R.IN["ratio_table"]: [
    "--stage", stage, "--dataset", R.IN["ds1400"], "--cells-dir", R.IN["cells1400"], "--radio-dir", mp("radio"),
    "--cutout-dir", mp("cut"), "--ratio-table", table, "--pkl", mp("pkl.pkl"), "--catalog", mp("cat.h5"),
    "--groupcat", mp("gc"), "--source-ids", mp("srcids"), "--nu-ghz", "1.4", "--w-scale", "16.48",
    "--max-nh", "1e-4", "--img-size", "128"]


def build_b():                                     # stands in for build_radio_cells + build_dataset
    os.makedirs(R.IN["cells1400"])
    for h in HID:
        _, w, _, T, nH, M = cells_of(h)
        keep = nH <= 1e-4
        fac = (1.4 / 0.144) ** (-0.5 * brc.s_of_M(M[keep]))
        np.savez(os.path.join(R.IN["cells1400"], f"cells_FOF{h}_sub{h}.npz"), w=w[keep] * fac / 16.48)
    write_maps(R.IN["ds1400"], np.arcsinh(img * rng.uniform(0.8, 1.2, img.shape)), R.IN["cells1400"], 128)


MB.main(B_ARGS("begin"))
build_b()
raises(SystemExit, MB.main, B_ARGS("begin"), what="job B begin with existing outputs")
open(mp("pkl.pkl"), "ab").write(b"x")                # a builder input changes mid-job
raises(SystemExit, MB.main, B_ARGS("end"), what="job B source changed between begin and end")
assert not os.path.exists(R.IN["ds1400"] + ".manifest.json")
import shutil
shutil.rmtree(R.IN["cells1400"]); os.remove(R.IN["ds1400"]); os.remove(R.IN["ds1400"] + ".begin.json")
MB.main(B_ARGS("begin"))
build_b()
MB.main(B_ARGS("end"))
assert os.path.exists(R.IN["ds1400"] + ".manifest.json")
print("    job B two-stage producer: success manifest only when sources are unchanged between begin and end")
R.IN["cells144"] = mp("c144")
with h5py.File(R.IN["ds128"], "a") as f:
    f.attrs["cells_dir"] = R.IN["cells144"]
# the cohort file changed (attribute fix) after job A: A must be rerun so its recorded cohort matches
for f_ in ("phase_good.npz", "phase_good.npz.manifest.json", "phase_lim.npz", "phase_chg.npz"):
    if os.path.exists(mp(f_)):
        os.remove(mp(f_))
E.main(A_ARGS(mp("phase_good.npz")))
good = mp("phase_good.npz")
dig = lambda d_: PU.dir_digest(d_)["digest"]
RS = PU.sha(R.IN["ratio_table"])
d, pv, _ = R.validate_phase(good, HID[::-1], RS, dig(R.IN["cells144"]))
assert list(d["halo_id"]) == list(HID[::-1])
R.validate_b_manifest(R.IN["ds1400"], dig(R.IN["cells1400"]), RS)
print("    real job A/B products accepted; phase arrays reindexed to the adopted parent order")


def variant(name, mutate):
    """Copy the good artifact, apply one defect, re-save with a fresh matching
    manifest so that only the planted defect is tested."""
    dd = dict(np.load(good, allow_pickle=True))
    pv_ = json.loads(str(dd["provenance"]))
    mutate(dd, pv_)
    dd["provenance"] = np.array(json.dumps(pv_))
    p_ = mp(name + ".npz")
    np.savez(p_, **dd)
    man_ = json.load(open(good + ".manifest.json"))
    man_["artifact_sha256"] = PU.sha(p_)
    json.dump(man_, open(p_ + ".manifest.json", "w"))
    return p_


def m_trunc(dd, pv_): dd["ratio_cut"] = dd["ratio_cut"][:-1]
def m_trunc2(dd, pv_): dd["frac_above"] = dd["frac_above"][:-1]
def m_edges(dd, pv_): dd["nh_edges"] = dd["nh_edges"] + 0.1
def m_norm(dd, pv_): pv_["config"]["normalisation"] = "per parent by its in-range population"
def m_rule(dd, pv_): pv_["config"]["nh_cut_rule"] = "kept: n_H < nh_cut"
def m_npar(dd, pv_): pv_["n_parents"] = 5
def m_dup(dd, pv_): dd["halo_id"] = np.array([101, 101, 307, 409])
def m_wrong(dd, pv_): dd["halo_id"] = np.array([101, 205, 307, 999])
def m_part(dd, pv_):
    for k in R.PHASE_PER_PARENT + ("H_n", "H_w"):
        dd[k] = dd[k][:3]
    pv_["n_parents"] = 3
def m_nu(dd, pv_): pv_["config"]["nu_hi_ghz"] = 1.5
def m_src(dd, pv_): pv_["sources_verified_unchanged"] = False
def m_cells(dd, pv_): pv_["cells_cut"]["digest"] = "0" * 64


for nm, mut, what in (("v1", m_trunc, "full ids but truncated ratio_cut"), ("v2", m_trunc2, "truncated frac_above"),
                      ("v3", m_edges, "altered histogram edges"), ("v4", m_norm, "altered normalisation"),
                      ("v5", m_rule, "altered density-boundary rule"), ("v6", m_npar, "n_parents inconsistent"),
                      ("v7", m_dup, "duplicate parents"), ("v8", m_wrong, "wrong parent"),
                      ("v9", m_part, "partial parent set"), ("v10", m_nu, "configuration mismatch"),
                      ("v11", m_src, "sources not verified unchanged"), ("v12", m_cells, "different 144 MHz cells")):
    raises(R.Fail, R.validate_phase, variant(nm, mut), HID, RS, dig(R.IN["cells144"]), what=what)
raises(R.Fail, R.validate_phase, mp("phase_lim_missing.npz"), HID, RS, dig(R.IN["cells144"]), what="missing artifact")
raises(R.Fail, R.validate_phase, good, HID, "0" * 64, dig(R.IN["cells144"]), what="different ratio table")
raises(R.Fail, R.validate_b_manifest, R.IN["ds1400"], "0" * 64, RS, what="job B cells differ from its manifest")
bm = json.load(open(R.IN["ds1400"] + ".manifest.json"))
for k, v, what in (("prescription", dict(bm["prescription"], nu_ghz=0.144), "job B prescription mismatch"),
                   ("sources_verified_unchanged", False, "job B not source-verified")):
    bad_m = dict(bm, **{k: v})
    json.dump(bad_m, open(R.IN["ds1400"] + ".manifest.json", "w"))
    raises(R.Fail, R.validate_b_manifest, R.IN["ds1400"], dig(R.IN["cells1400"]), RS, what=what)
json.dump(bm, open(R.IN["ds1400"] + ".manifest.json", "w"))
# real task paths, promotable success and source-change rejection
rc = R.main(["--run-dir", mp("run_w3_ok"), "--tasks", "w3", "--phase-npz", good])
man3 = st(mp("run_w3_ok"))
assert rc == 0 and man3["status"]["w3"] == "ok" and man3["promotable"] and not man3["sources_changed"], man3
assert os.path.exists(mp("run_w3_ok/w3_arrays.npz")) and os.path.exists(mp("run_w3_ok/fig_phase.pdf"))
print("    real task w3 on real producer outputs: status ok, promotable, exit 0")
rc = R.main(["--run-dir", mp("run_dep_fail"), "--tasks", "w2,w3", "--phase-npz", variant("v13", m_trunc)])
manf = st(mp("run_dep_fail"))
assert rc == 1 and all(manf["status"][t].startswith("unavailable: prerequisite failed") for t in ("w2", "w3"))
assert not os.path.exists(mp("run_dep_fail/w2.json")), "W2 must not fit after a failed prerequisite"
print("    D prerequisite gate: invalid job-A product -> w2 and w3 unavailable before any fit")
TMPIN = mp("touched.txt")
open(TMPIN, "w").write("a")
def t_touch(run, a):
    run.use("touch_t", f=TMPIN)
    open(TMPIN, "a").write("b")
    run.record("touch_t", {})
R.TASKS["touch_t"] = t_touch
rc = R.main(["--run-dir", mp("run_touch"), "--tasks", "touch_t"])
mt_ = st(mp("run_touch"))
assert rc == 1 and TMPIN in mt_["sources_changed"] and not mt_["promotable"] and mt_["status"]["touch_t"] == "ok"
print("    an input modified during the run -> sources_changed, not promotable, exit 1")
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




def write_w9_inf():
    write_w9()
    with h5py.File(R.IN["relic_cat"], "a") as f:
        dm = f["d_max"][:]
        i, j = np.argwhere(np.isfinite(dm))[0]
        dm[i, j] = np.inf
        f["d_max"][...] = dm


write_w9()
rc = R.main(["--run-dir", mp("run_w9_ok"), "--tasks", "w9"])
assert st(mp("run_w9_ok"))["status"]["w9"] == "ok" and rc == 0, st(mp("run_w9_ok"))
write_w9(alter=True)
R.main(["--run-dir", mp("run_w9_alt"), "--tasks", "w9"])
assert st(mp("run_w9_alt"))["status"]["w9"].startswith("failed check: review required")
assert os.path.exists(mp("run_w9_alt/w9_arrays.npz")) and os.path.exists(mp("run_w9_alt/w9.json"))
write_w9_inf()
rc = R.main(["--run-dir", mp("run_w9_inf"), "--tasks", "w9"])
mi = st(mp("run_w9_inf"))
assert rc == 1 and mi["status"]["w9"].startswith("failed check: review required") and not mi["promotable"]
write_w9(tsc_ids=np.array([101, 205, 307, 307, 409]))
R.main(["--run-dir", mp("run_w9_dup"), "--tasks", "w9"])
assert "duplicate keys" in st(mp("run_w9_dup"))["status"]["w9"]
write_w9(tsc_ids=np.array([101, 205, 307, 999]))
R.main(["--run-dir", mp("run_w9_miss"), "--tasks", "w9"])
assert "missing from the source" in st(mp("run_w9_miss"))["status"]["w9"]
print("    real task w9: ok/promotable when redetection reproduces the catalogue; 'review required' for a changed "
      "peak and for a +inf catalogue distance (not promotable); failed for duplicate or missing join keys")
print("ALL REGEN FIXTURE CHECKS PASSED")
