"""How much of the forward-model R2 gap between cuts is mock-realization noise?

The mock realization is random (sim/obs pairing, P150 flux scatter, thermal
noise), so single-seed differences cannot justify a configuration choice.
Seeds are matched across configurations, so the comparisons are paired.
"""
import numpy as np
from scipy import stats

r2 = {
    ("nh3", "arc"): [0.332, 0.367, 0.352, 0.355, 0.313],
    ("nh4", "arc"): [0.285, 0.302, 0.327, 0.328, 0.279],
    ("nh3", "lin"): [0.390, 0.393, 0.398, 0.386, 0.428],
    ("nh4", "lin"): [0.365, 0.329, 0.313, 0.399, 0.336],
}
ks = {
    ("nh3", "arc"): [0.228, 0.254, 0.288, 0.248, 0.237],
    ("nh4", "arc"): [0.211, 0.217, 0.251, 0.211, 0.205],
    ("nh3", "lin"): [0.509, 0.442, 0.455, 0.451, 0.449],
    ("nh4", "lin"): [0.481, 0.447, 0.438, 0.465, 0.469],
}

for name, d in [("mock -> TSC R2 (cluster-mean)", r2),
                ("ncomp@3sigma KS D", ks)]:
    print(f"\n{name}")
    print(f"{'config':<12}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
    for k, v in d.items():
        v = np.array(v)
        print(f"{k[0]+' '+k[1]:<12}{v.mean():>9.4f}{v.std(ddof=1):>9.4f}"
              f"{v.min():>9.3f}{v.max():>9.3f}")
    for mode in ["arc", "lin"]:
        a, b = np.array(d[("nh3", mode)]), np.array(d[("nh4", mode)])
        t = stats.ttest_rel(a, b)
        print(f"  {mode}: nh3 - nh4 = {(a-b).mean():+.4f}, "
              f"paired t={t.statistic:+.2f}, p={t.pvalue:.3f}")

# Do the two cuts actually differ for the CNN? (from check_cut_cnn_stats.py)
cnn = {"nh3": [0.568, 0.556, 0.550, 0.570, 0.570],
       "nh4": [0.594, 0.559, 0.549, 0.557, 0.574]}
a, b = np.array(cnn["nh3"]), np.array(cnn["nh4"])
t = stats.ttest_rel(b, a)
print(f"\nCNN OOF R2: nh4 - nh3 = {(b-a).mean():+.4f} "
      f"+/- {(b-a).std(ddof=1)/np.sqrt(5):.4f} (SE), "
      f"paired t={t.statistic:+.2f}, p={t.pvalue:.3f}")
