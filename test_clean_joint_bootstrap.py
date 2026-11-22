#!/usr/bin/env python3
"""Synthetic fixtures for score_clean_joint_bootstrap (review rounds 62-63)."""
import numpy as np

import score_clean_joint_bootstrap as S


def raises(exc, f, *a, what="", **k):
    try:
        f(*a, **k)
    except exc as e:
        print(f"  rejected ({what}): {str(e)[:100]}")
        return
    raise SystemExit(f"NOT rejected: {what}")


rng = np.random.default_rng(1)
y = rng.uniform(0, 4, 352)
A = y + rng.normal(0, 1.0, (5, 352))
r = S.paired_bootstrap(y, A, A.copy(), n=200)
assert r["ensemble"]["delta"] == 0 and r["ensemble"]["ci"] == [0.0, 0.0] and r["mean_per_seed"]["ci"] == [0.0, 0.0]
print("identical arms -> increment 0, interval [0, 0]")
B = y + rng.normal(0, 0.6, (5, 352))
r = S.paired_bootstrap(y, A, B, n=400)
assert r["ensemble"]["delta"] > 0 and r["ensemble"]["ci"][0] > 0 and r["mean_per_seed"]["ci"][0] > 0
print(f"planted better arm -> ensemble {r['ensemble']['delta']:+.3f} {np.round(r['ensemble']['ci'], 3).tolist()}")
C = A + rng.normal(0, 0.05, (5, 352))
r = S.paired_bootstrap(y, A, C, n=400)
assert r["ensemble"]["ci"][0] < 0 < r["ensemble"]["ci"][1], r["ensemble"]
print(f"near-identical arms -> interval includes 0: {np.round(r['ensemble']['ci'], 4).tolist()}")
assert S.paired_bootstrap(y, A, B, n=100) == S.paired_bootstrap(y, A, B, n=100)
print("fixed seed -> reproducible")
raises(S.Undefined, S.paired_bootstrap, np.full(352, 2.0), A, B, n=10, what="constant target (undefined R2)")
y2 = np.full(352, 1.0); y2[:2] = [3.0, 0.5]      # some resamples (~15%) miss both distinct parents
r = S.paired_bootstrap(y2, A, B, n=200, seed=3)
assert r["invalid_draws"] > 0 and r["ensemble"]["ci"] is None and r["mean_per_seed"]["ci"] is None
assert r["ensemble"]["interval_status"].startswith("unavailable")
print(f"undefined bootstrap draws -> counted ({r['invalid_draws']} of 200), intervals unavailable (null)")

# parent-level contract on synthetic archives with self-consistent fold R2
def arch(pred, s, modality, y=y, hid=None, gid=None):
    hid = np.arange(1000, 1352) if hid is None else hid
    gid = hid if gid is None else gid
    fold = S.reconstruct_folds(352, s)
    fr = np.array([S.r2(y[fold == f], pred[fold == f]) for f in range(5)])
    return dict(label=y.copy(), halo_id=hid, group_id=gid, snap=np.full(352, 99), oof_pred=pred,
                fold_r2=fr, select="inner", seed=s, modality=modality)

da = [arch(A[k], s, "single") for k, s in enumerate(S.SEEDS)]
db = [arch(B[k], s, "radio+xray") for k, s in enumerate(S.SEEDS)]
S.validate(da, db)
print("valid synthetic archives accepted")
dup = np.arange(1000, 1352); dup[5] = dup[4]
raises(SystemExit, S.validate, [arch(A[k], s, "single", hid=dup) for k, s in enumerate(S.SEEDS)],
       [arch(B[k], s, "radio+xray", hid=dup) for k, s in enumerate(S.SEEDS)], what="duplicate parents")
raises(SystemExit, S.validate, da, [arch(B[k], s, "single") for k, s in enumerate(S.SEEDS)], what="wrong modality")
y3 = y.copy(); y3[7] += 1e-7
raises(SystemExit, S.validate, da, [arch(B[k], s, "radio+xray", y=y3) for k, s in enumerate(S.SEEDS)],
       what="labels not exactly equal")
other = np.arange(2000, 2352)
raises(SystemExit, S.validate, da, [arch(B[k], s, "radio+xray", hid=other) for k, s in enumerate(S.SEEDS)],
       what="altered parent identity")
bad = [dict(d) for d in db]; bad[2]["fold_r2"] = bad[2]["fold_r2"] + 0.01
raises(SystemExit, S.validate, da, bad, what="fold R2 inconsistent with the reconstructed partition")
raises(SystemExit, S.validate, da[:4], db, what="missing seed")
print("ALL CLEAN-BOOTSTRAP FIXTURES PASSED")
