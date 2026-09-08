"""Does the density cut help the CNN, not just the LoTSS forward model?

Pooled CNN, pseudo-TSC, 5-fold OOF, 5 seeds matched across datasets so the
comparison is paired (same init, same fold assignment).
"""
import numpy as np
from scipy import stats

d = {
    "gp  (no cut)":   [0.576, 0.549, 0.528, 0.540, 0.537],
    "n_H < 1e-3":     [0.568, 0.556, 0.550, 0.570, 0.570],
    "n_H < 1e-4":     [0.594, 0.559, 0.549, 0.557, 0.574],
}
print(f"{'dataset':<16}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}")
for k, v in d.items():
    v = np.array(v)
    print(f"{k:<16}{v.mean():>8.4f}{v.std(ddof=1):>8.4f}{v.min():>8.3f}{v.max():>8.3f}")

base = np.array(d["gp  (no cut)"])
print()
for k in ["n_H < 1e-3", "n_H < 1e-4"]:
    v = np.array(d[k])
    diff = v - base
    t = stats.ttest_rel(v, base)
    print(f"{k} - no cut = {diff.mean():+.4f} +/- "
          f"{diff.std(ddof=1)/np.sqrt(len(diff)):.4f} (SE)  "
          f"paired t={t.statistic:+.2f}, p={t.pvalue:.3f}")
    print(f"   per-seed: {np.round(diff, 3)}  ({(diff > 0).sum()}/5 positive)")
