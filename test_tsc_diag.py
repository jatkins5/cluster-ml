#!/usr/bin/env python3
"""Fixtures for diagnose_tsc_hardness.py (review R33) -- no data analysis."""
import os
import tempfile

import h5py
import numpy as np
from sklearn.metrics import r2_score

import diagnose_tsc_hardness as D

tau = np.round(np.arange(1, 41) * 0.1, 10)


def expect(f, *a, what="", **k):
    try:
        f(*a, **k)
    except SystemExit as e:
        print(f"    rejected ({what}): {e}")
        return
    raise SystemExit(f"NOT rejected: {what}")


print("== crossing classes and the stored rule")
cases = {"floor": np.full(40, 0.7), "interior": np.r_[np.zeros(4), 0.5, np.full(35, 0.6)],
         "final": np.r_[np.full(39, 0.3), 0.6], "none": np.full(40, 0.3)}
for k, cv in cases.items():
    c, i = D.crossing_class(cv)
    v = D.pseudo_from_curve(cv, tau)
    assert c == k, (k, c)
    print(f"    {k:<8} -> class {c}, index {i}, pseudo {v:.4f}")
assert D.pseudo_from_curve(cases["final"], tau) < 4.0 + 1e-12 and D.crossing_class(cases["final"])[0] != "none"
assert D.pseudo_from_curve(cases["none"], tau) == 4.0
near = np.r_[np.zeros(4), np.float32(0.49999997), np.full(35, 0.6)].astype(np.float32)
c, i = D.crossing_class(near)
v32 = D.pseudo_from_curve(near, tau.astype(np.float32))
print(f"    near-threshold float32 value 0.49999997 -> class {c}, crossing index {i} (not at the 0.5 sample), "
      f"pseudo {v32:.6f}")
assert c == "interior" and i == 5

print("== event join")
tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "."))
cat = os.path.join(tmp, "cat.h5")
with h5py.File(cat, "w") as f:
    f["haloID"] = np.array([100, 200, 300])


def mergers(path, hid, t, s, mr):
    with h5py.File(path, "w") as f:
        f["HaloID"], f["T_coll"], f["Snap_coll"], f["Mass_ratio"] = hid, t, s, mr


good = os.path.join(tmp, "m.h5")
mergers(good, np.array([1, 0, 1, 1]), np.array([12.0, 13.0, 13.5, 13.5]), np.array([95, 98, 99, 99]),
        np.array([0.2, 0.3, 0.05, 0.4]))
ev = D.load_events(good, cat)
assert sorted(set(ev["parent"].tolist())) == [100, 200]
print("    valid join; parent 300 has no events (retained, not an error)")
r = D.setting_event(np.array([], dtype=int), ev["T_coll"], ev["Mass_ratio"])
assert r[0] == -1 and np.isnan(r[2]) and not r[3]
print("    missing-event parent -> no setting event, NaN ratio")
bad = os.path.join(tmp, "b1.h5"); mergers(bad, np.array([1.5, 0, 1, 1]), ev["T_coll"], ev["Snap_coll"], ev["Mass_ratio"])
expect(D.load_events, bad, cat, what="non-integral HaloID")
bad = os.path.join(tmp, "b2.h5"); mergers(bad, np.array([1, 0, 5, 1]), ev["T_coll"], ev["Snap_coll"], ev["Mass_ratio"])
expect(D.load_events, bad, cat, what="HaloID out of bounds")
dcat = os.path.join(tmp, "dcat.h5")
with h5py.File(dcat, "w") as f:
    f["haloID"] = np.array([100, 100, 300])
expect(D.load_events, good, dcat, what="duplicate catalogue parent ids")

print("== setting-event tie rule (attributes from the same row; unsorted-array trap)")
rows200 = np.where(ev["parent"] == 200)[0]           # rows 0, 2, 3 in file order
sel, tied, ratio, valid = D.setting_event(rows200, ev["T_coll"], ev["Mass_ratio"])
assert (sel, tied, ratio, valid) == (3, 2, 0.4, True), (sel, tied, ratio, valid)
print(f"    two events tied at the latest T_coll: chose row {sel} (ratio {ratio}, the larger), tied {tied}")
mr2 = ev["Mass_ratio"].copy(); mr2[2] = 0.4
sel = D.setting_event(rows200, ev["T_coll"], mr2)[0]
assert sel == 2
print("    equal ratios -> lowest row (2)")
mr3 = ev["Mass_ratio"].copy(); mr3[2], mr3[3] = -1.0, np.nan
sel, _, _, valid = D.setting_event(rows200, ev["T_coll"], mr3)
assert sel == 2 and not valid
print("    all tied ratios invalid -> lowest row, invalid category")
shuffled = rows200[::-1]
assert D.setting_event(shuffled, ev["T_coll"], ev["Mass_ratio"])[:3] == D.setting_event(rows200, ev["T_coll"], ev["Mass_ratio"])[:3]
print("    order of the candidate rows does not change the choice")
D.verify_recent(13.8, np.array([12.0, 13.5]), 0.3, "ok")
expect(D.verify_recent, 13.8, np.array([12.0, 13.5]), 0.5, "x", what="most-recent T_coll disagrees with the saved truth")
expect(D.verify_recent, 13.8, np.array([]), 0.3, "x", what="no past event but finite saved truth")
D.verify_recent(13.8, np.array([]), np.nan, "ok")

print("== map-fitting parents, identities, nomination boundaries")
pf = {p: p % 5 for p in range(20)}
fit = D.fitting_parents(pf, {1, 6}, 0)
assert not set(fit) & {p for p in pf if pf[p] == 0} and not set(fit) & {1, 6}
print(f"    fold 0 fitting parents exclude outer fold 0 and inner {{1, 6}}: {len(fit)} parents")
rg = np.random.default_rng(0)
t, p = rg.uniform(0, 4, 50), rg.uniform(0, 4, 50); ph = p + rg.normal(0, 0.3, 50)
lhs, a1, a2, a3 = D.mse_identity(ph, p, t)
assert np.isclose(lhs, a1 + a2 + a3, rtol=0, atol=1e-12)
assert np.isclose(r2_score(2.5 * t - 1.3, 2.5 * ph - 1.3), r2_score(t, ph), rtol=0, atol=1e-12)
print("    accounting identity and affine invariance hold")


def comp(share, mse_c, mse_p, by, bo):
    """Synthetic X-ray strata with internally consistent SSE = MSE x n_obs."""
    c = {s: dict(sse_cat=mse_c * 10, mse_cat=mse_c, mse_pse=mse_p, n_obs=10, bias_cat=0.0) for s in D.RANGE_STRATA}
    sse_r = 3 * mse_c * 10
    c["[0,0.5]"] = dict(sse_cat=0.0, mse_cat=0, mse_pse=0, n_obs=10, bias_cat=by)
    c["finite>4"]["bias_cat"] = bo
    c["sse_cat_total"] = sse_r / share
    return c


assert D.nominate(comp(0.50, 4.0, 2.0, 0, 0))[0].startswith("range")
assert D.nominate(comp(0.4999, 4.0, 2.0, 0.5, -0.5))[0].startswith("loss")
assert D.nominate(comp(0.60, 1.0, 2.0, 0.3, -0.5))[0] == "none"           # bias exactly +0.3: not > 0.3
assert D.nominate(comp(0.60, 1.0, 2.0, 0.31, -0.3))[0] == "none"          # -0.3 exactly: not < -0.3
assert D.nominate(comp(0.60, 1.0, 2.0, 0.31, -0.31))[0].startswith("loss")
print("    nomination: share >= 0.50 with higher catalogue MSE -> range; bias > +0.3 and < -0.3 strictly -> loss")
print("ALL TSC-DIAGNOSTIC FIXTURE CHECKS PASSED")
