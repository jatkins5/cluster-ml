"""Review F7 reruns under the current protocol (inner-split selection),
144 MHz, snapshot 99, paired seeds 42-46.
  1. density cut vs none (the earlier p = 0.010 used pre-fix selection)
  2. clean joint radio + X-ray on 144 MHz radio (was 1.4 GHz: 0.537)
"""
import glob
import os

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score


def r2s(pat):
    out = {}
    for s in range(42, 47):
        p = f"cnn_preds/{pat.format(s)}"
        if os.path.exists(p):
            d = np.load(p)
            out[s] = r2_score(d["label"], d["oof_pred"])
    return out


def paired(lab, a, b):
    s = sorted(set(a) & set(b))
    x, y = np.array([a[k] for k in s]), np.array([b[k] for k in s])
    t = stats.ttest_rel(y, x)
    print(f"{lab:<46} {x.mean():.3f} -> {y.mean():.3f}  diff {np.mean(y - x):+.3f} "
          f"± {np.std(y - x, ddof=1):.3f}  p={t.pvalue:.3f}  ({sum(y > x)}/{len(s)} seeds higher, n={len(s)})")


cut = r2s("pooled_nh4_144_inner_s{}.npz")
paired("no cut -> n_H < 1e-4 cut (144 MHz, inner)", r2s("pooled_nocut144_inner_s{}.npz"), cut)
paired("radio -> clean joint radio + X-ray (144 MHz)", cut, r2s("joint144_clean_inner_s{}.npz"))
old = r2s("joint_scaled_inner_s{}.npz")
paired("clean joint 1.4 GHz -> 144 MHz", old, r2s("joint144_clean_inner_s{}.npz"))
