"""Radio vs X-ray predictions for the real clusters with both, from the
seed ensembles of each configuration (99 only at 144 MHz; 99+91)."""
import numpy as np
import pandas as pd
from scipy import stats

wl = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")
key = lambda s: "".join(str(s).split()).replace("_", "").upper()


def ens(pat, seeds=(42, 43, 44)):
    ds = [np.load(f"cnn_preds/{pat.format(s)}") for s in seeds]
    names = [key(n) for n in ds[0]["obs_name"]]
    return dict(zip(names, np.mean([d["obs_pred"] for d in ds], 0))), \
        dict(zip(names, ds[0]["obs_z"]))


for lab, rp, xp in [("99 only (144 MHz)", "nu144_sfimg_s{}_preds.npz", "x144_s{}_preds.npz"),
                    ("99 + 91", "s9991_sfimg_s{}_preds.npz", "s9991_x_s{}_preds.npz")]:
    r, rz = ens(rp)
    x, _ = ens(xp)
    both = sorted(set(r) & set(x))
    a, b = np.array([r[k] for k in both]), np.array([x[k] for k in both])
    z = np.array([rz[k] for k in both])
    rho, p = stats.spearmanr(a, b)
    rk = stats.rankdata
    def partial(c):
        A = np.column_stack([rk(c), np.ones(len(c))])
        res = lambda v: rk(v) - A @ np.linalg.lstsq(A, rk(v), rcond=None)[0]
        return stats.pearsonr(res(a), res(b))
    pz = partial(z)
    m = [k in wl.index for k in both]
    am, bm = a[m], b[m]
    lm = np.array([wl.loc[k, "log_m500c"] for k in np.array(both)[m]])
    A = np.column_stack([stats.rankdata(lm), np.ones(len(lm))])
    res = lambda v: stats.rankdata(v) - A @ np.linalg.lstsq(A, stats.rankdata(v), rcond=None)[0]
    pm = stats.pearsonr(res(am), res(bm))
    print(f"{lab}: n={len(both)}  rho={rho:+.2f} (p={p:.3f});  fixed z {pz[0]:+.2f} "
          f"(p={pz[1]:.3f});  fixed WL mass {pm[0]:+.2f} (p={pm[1]:.2f}, n={m.count(True)})")
    print(f"   radio range {a.min():.2f}-{a.max():.2f} Gyr, X-ray {b.min():.2f}-{b.max():.2f}")
