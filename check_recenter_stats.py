"""Paired comparison of weight-centred vs GroupPos-centred training.

OOF R2 from 5 seeds x 2 models x 2 centrings (submit_recenter_array.sh and
submit_recenter_seeds.sh). Seeds are matched across centrings -- same init,
same fold assignment -- so the paired test is both correct and much more
sensitive than comparing group means.
"""
import numpy as np
from scipy import stats

SEEDS = [42, 43, 44, 45, 46]
d = {
    ("shallow", "wc"): [0.531, 0.530, 0.540, 0.528, 0.526],
    ("shallow", "gp"): [0.529, 0.541, 0.539, 0.522, 0.529],
    ("pooled",  "wc"): [0.554, 0.529, 0.523, 0.550, 0.524],
    ("pooled",  "gp"): [0.565, 0.554, 0.526, 0.555, 0.545],
}

print(f"{'model':<9}{'centre':<8}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}")
for k, v in d.items():
    v = np.array(v)
    print(f"{k[0]:<9}{k[1]:<8}{v.mean():>8.4f}{v.std(ddof=1):>8.4f}"
          f"{v.min():>8.3f}{v.max():>8.3f}")

print()
for m in ["shallow", "pooled"]:
    wc, gp = np.array(d[(m, "wc")]), np.array(d[(m, "gp")])
    diff = gp - wc
    t = stats.ttest_rel(gp, wc)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"{m}: gp - wc = {diff.mean():+.4f} +/- {se:.4f} (SE)"
          f"   paired t={t.statistic:+.2f}, p={t.pvalue:.3f}")
    print(f"   per-seed deltas: {np.round(diff, 3)}")
