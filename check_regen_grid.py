"""Summarise the regenerated forward-model grid (beam-correlated noise).

Every KS table and mock->TSC number predating the noise fix was computed with
white noise, which is wrong. This is the replacement, five mock seeds per
configuration, paired across configurations.
"""
import numpy as np
from scipy import stats

rows = []
for line in open("regen_grid_summary.txt"):
    p = line.split()
    rows.append((p[0], p[1], float(p[3]), float(p[4]), float(p[5]), float(p[6])))

cfgs = sorted({(t, m) for t, m, *_ in rows})
d = {c: np.array([r[2:] for r in rows if (r[0], r[1]) == c]) for c in cfgs}

print(f"{'cut':<6}{'map':<6}{'R2':>16}{'ncomp@3s':>16}{'area@8s':>16}"
      f"{'area@13s':>16}")
for c in cfgs:
    a = d[c]
    cols = "".join(f"{a[:, j].mean():>9.3f}+-{a[:, j].std(ddof=1):<6.3f}"
                   for j in range(4))
    print(f"{c[0]:<6}{c[1]:<6}{cols}")

print("\npaired comparisons (same mock seeds):")
for m in ["arc", "lin"]:
    for a_tag, b_tag in [("nh3", "gp"), ("nh4", "gp"), ("nh4", "nh3")]:
        a, b = d[(a_tag, m)], d[(b_tag, m)]
        for j, n in enumerate(["R2", "ncomp@3s", "area@8s", "area@13s"]):
            t = stats.ttest_rel(a[:, j], b[:, j])
            if t.pvalue < 0.05:
                print(f"  {m} {a_tag}-{b_tag} {n:<9} "
                      f"{(a[:, j]-b[:, j]).mean():+.3f}  p={t.pvalue:.3f}")
