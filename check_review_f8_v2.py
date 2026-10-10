"""F8, corrected per Codex round 1 (CODEX_CLAUDE_REVIEW.md).

Partial rank correlation of radio vs X-ray real predictions given a nuisance
c (redshift or weak-lensing mass). a, b, c are ranked once; Z = [1, rank c];
the statistic is corr(M_Z ra, M_Z rb). Freedman-Lane: permute the reduced-
model residuals M_Z rb, add back the fitted part, and recompute the
statistic WITHOUT reranking. Relic: each point estimator paired with its own
interval (halo means with a halo bootstrap; flattened projections with a
halo-block bootstrap).
"""
import glob

import h5py
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(0)
NPERM = 100000


def fl_partial(a, b, c, n=NPERM):
    ra, rb, rc = (stats.rankdata(v) for v in (a, b, c))
    Z = np.column_stack([np.ones(len(rc)), rc])
    M = np.eye(len(rc)) - Z @ np.linalg.pinv(Z)
    ea, eb = M @ ra, M @ rb
    stat = lambda x, y: np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
    obs = stat(ea, eb)
    fit = rb - eb
    cnt = 0
    for _ in range(n):
        bs = fit + rng.permutation(eb)
        cnt += abs(stat(ea, M @ bs)) >= abs(obs) - 1e-12
    k = len(rc)
    t = obs * np.sqrt((k - 3) / (1 - obs ** 2))
    return obs, (cnt + 1) / (n + 1), 2 * stats.t.sf(abs(t), k - 3)


wl = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")
key = lambda s: "".join(str(s).split()).replace("_", "").upper()


def ens(pat):
    ds = [np.load(p) for p in sorted(glob.glob(pat))]
    names = [key(n) for n in ds[0]["obs_name"]]
    return (dict(zip(names, np.mean([d["obs_pred"] for d in ds], 0))),
            dict(zip(names, ds[0]["obs_z"])))


r, rz = ens("cnn_preds/s9991_sfimg_s*_preds.npz")
x, _ = ens("cnn_preds/s9991_x_s*_preds.npz")
both = sorted(set(r) & set(x))
a = np.array([r[k] for k in both]); b = np.array([x[k] for k in both])
z = np.array([rz[k] for k in both])
print("======== F8a (Freedman-Lane, no reranking, 1e5 permutations)")
for lab, aa, bb, cc in [("| z", a, b, z)] + [(
        "| WL mass", a[m], b[m], np.array([wl.loc[k, "log_m500c"] for k in np.array(both)[m]]))
        for m in [np.array([k in wl.index for k in both])]]:
    o, pp, pt = fl_partial(aa, bb, cc)
    print(f"  partial {lab:<9} {o:+.3f}  n={len(aa)}  permutation p {pp:.4f}  "
          f"t(n-3) p {pt:.4f}")

print("\n======== F8b: relic, each estimator with its own interval")
with h5py.File("relic_catalog_radio_nh4_144.h5") as f:
    d = f["d_max"][:] / 1000.0
    mt = f["merger_tsc"][:]
    mr = f["mass_ratio"][:]
for lab, sel in (("TSC <= 2", np.isfinite(mt) & (mt <= 2)),
                 ("TSC <= 2, ratio >= 0.1", np.isfinite(mt) & (mt <= 2) & (mr >= 0.1))):
    D, T = d[sel], mt[sel]
    dm = np.nanmean(D, axis=1)
    ok = np.isfinite(dm)
    hm = stats.spearmanr(dm[ok], T[ok])[0]
    hb = []
    idx = np.where(ok)[0]
    for _ in range(5000):
        s = rng.choice(idx, len(idx))
        hb.append(stats.spearmanr(dm[s], T[s])[0])
    k = np.isfinite(D)
    fl = stats.spearmanr(D[k], np.repeat(T[:, None], 3, 1)[k])[0]
    fb = []
    allh = np.arange(len(D))
    for _ in range(5000):
        s = rng.choice(allh, len(allh))
        Ds, Ts = D[s], np.repeat(T[s][:, None], 3, 1)
        kk = np.isfinite(Ds)
        fb.append(stats.spearmanr(Ds[kk], Ts[kk])[0])
    print(f"  {lab}: halo-mean rho {hm:+.3f}, halo bootstrap 95% "
          f"[{np.percentile(hb, 2.5):+.3f}, {np.percentile(hb, 97.5):+.3f}] (n={ok.sum()} halos);"
          f" flattened rho {fl:+.3f}, halo-block bootstrap 95% "
          f"[{np.percentile(fb, 2.5):+.3f}, {np.percentile(fb, 97.5):+.3f}]")
