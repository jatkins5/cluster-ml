#!/usr/bin/env python3
"""Synthetic fixtures for score_clean_joint_bootstrap.paired_bootstrap."""
import numpy as np
import score_clean_joint_bootstrap as S

rng = np.random.default_rng(1)
y = rng.uniform(0, 4, 352)
A = y + rng.normal(0, 1.0, (5, 352))
r = S.paired_bootstrap(y, A, A.copy(), n=200)
assert r["ensemble"]["delta"] == 0 and r["ensemble"]["ci"] == [0.0, 0.0] and r["mean_per_seed"]["ci"] == [0.0, 0.0]
print("identical arms -> increment 0, interval [0, 0]")
B = y + rng.normal(0, 0.6, (5, 352))
r = S.paired_bootstrap(y, A, B, n=400)
assert r["ensemble"]["delta"] > 0 and r["ensemble"]["ci"][0] > 0 and r["mean_per_seed"]["ci"][0] > 0
print(f"planted better arm -> ensemble {r['ensemble']['delta']:+.3f} {r['ensemble']['ci']}")
C = A + rng.normal(0, 0.05, (5, 352))
r = S.paired_bootstrap(y, A, C, n=400)
assert r["ensemble"]["ci"][0] < 0 < r["ensemble"]["ci"][1] or abs(r["ensemble"]["delta"]) < 0.01
print(f"near-identical arms -> ensemble {r['ensemble']['delta']:+.4f} {np.round(r['ensemble']['ci'], 4).tolist()}")
r1 = S.paired_bootstrap(y, A, B, n=100); r2 = S.paired_bootstrap(y, A, B, n=100)
assert r1 == r2
print("fixed seed -> reproducible")
print("ALL CLEAN-BOOTSTRAP FIXTURES PASSED")
