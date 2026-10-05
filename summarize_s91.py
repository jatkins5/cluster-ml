"""Snapshots 99+91 vs 99 alone, paired by seed, scored on the snapshot-99
clusters only (the same 352 clusters either way); plus how far the real-
cluster predictions move."""
import glob
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score


def r2_99(p, mock):
    d = np.load(p)
    if mock:
        y, yh, s = d["tsc"], d["oof_cluster"], d["snap"] if "snap" in d else None
    else:
        y, yh, s = d["label"], d["oof_pred"], d["snap"] if "snap" in d else None
    m = np.ones(len(y), bool) if s is None else (s == 99)
    return r2_score(y[m], yh[m])


rows = [("clean pooled CNN", "pooled_nh4_144_inner_s{}.npz", "pooled_s9991_inner_s{}.npz", range(42, 47), False),
        ("radio transfer, image", "nu144_sfimg_s{}_preds.npz", "s9991_sfimg_s{}_preds.npz", range(42, 45), True),
        ("radio transfer, image+mass", "nu144_sfimgmassn_s{}_preds.npz", "s9991_sfimgmassn_s{}_preds.npz", range(42, 45), True),
        ("X-ray (real depth)", "x144_s{}_preds.npz", "s9991_x_s{}_preds.npz", range(42, 45), True),
        ("joint radio+X-ray", "joint144_s{}_preds.npz", "s9991_joint_s{}_preds.npz", range(42, 45), True)]
print(f"{'model':<28}{'99 only':>9}{'99+91':>9}{'paired diff':>18}{'real-pred shift':>32}")
for name, a, b, seeds, mock in rows:
    ra = np.array([r2_99(f"cnn_preds/{a.format(s)}", mock) for s in seeds])
    rb = np.array([r2_99(f"cnn_preds/{b.format(s)}", mock) for s in seeds])
    d = rb - ra
    t = stats.ttest_rel(rb, ra)
    msg = ""
    if mock:
        pa = np.mean([np.load(f"cnn_preds/{a.format(s)}")["obs_pred"] for s in seeds], 0)
        pb = np.mean([np.load(f"cnn_preds/{b.format(s)}")["obs_pred"] for s in seeds], 0)
        msg = (f"rho {stats.spearmanr(pa, pb)[0]:+.2f}, mean {np.mean(pb - pa):+.2f} Gyr"
               f" (n={len(pa)})")
    print(f"{name:<28}{ra.mean():>9.3f}{rb.mean():>9.3f}"
          f"{d.mean():>+9.3f} ± {d.std(ddof=1):.3f} p={t.pvalue:.2f}  {msg}")
