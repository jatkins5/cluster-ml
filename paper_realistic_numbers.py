"""Every realistic-pipeline number the manuscript quotes, from saved
predictions (corrected forward models: F1 physical radio recipe, F5 X-ray
count model; snapshots 99+91; scored on snapshot-99 clusters).

Estimators (review F3):
  pooled  : per-seed cluster-mean R2 (3 projections pooled, realizations
            averaged), mean ± sd over 5 seeds -- a pooled-view benchmark
  1v-5    : single observations (realization x projection), 5-seed ensemble
  1v-1    : single observations, single model (mean over seeds)
Real clusters: 25-model ensembles (5 folds x 5 seeds), labelled as such;
partial correlations by the checked Freedman-Lane permutation (no reranking).
"""
import glob

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

SEEDS = range(42, 47)
key = lambda s: "".join(str(s).split()).replace("_", "").upper()
rng = np.random.default_rng(0)


def load(pat):
    fs = [f"cnn_preds/{pat.format(s)}" for s in SEEDS]
    return [np.load(f, allow_pickle=True) for f in fs]


def scores(ds):
    m = ds[0]["snap"] == 99
    pooled = [r2_score(d["tsc"][m], d["oof_cluster"][m]) for d in ds]
    R = len(ds[0]["oof_sample"]) // len(ds[0]["tsc"])
    ms = np.repeat(ds[0]["snap"], R) == 99
    P = ds[0]["oof_single"].shape[1]
    y = np.repeat(ds[0]["y_sample"][ms][:, None], P, 1).ravel()
    single = [d["oof_single"][ms].ravel() for d in ds]
    return (np.mean(pooled), np.std(pooled, ddof=1), r2_score(y, np.mean(single, 0)),
            np.mean([r2_score(y, q) for q in single]),
            np.mean([stats.spearmanr(d["oof_cluster"][m], np.asarray(d["sim_logm"])[m])[0]
                      for d in ds]))


ROWS = [("radio, physical 1 Mpc", "fix_phys_b1000_sfimg_s{}_preds.npz"),
        ("radio, physical 3 Mpc", "fix_phys_b3000_sfimg_s{}_preds.npz"),
        ("radio, arcsinh-morphology arm 3 Mpc", "fix_asinh_b3000_sfimg_s{}_preds.npz"),
        ("mass only (fixed noisy measurement)", "r4_massonly_s{}_preds.npz"),
        ("radio 3 Mpc + mass (fixed measurement)", "r4_imgmass_b3000_s{}_preds.npz"),
        ("X-ray, real count model", "fix_x_s{}_preds.npz"),
        ("radio 1 Mpc + X-ray", "fix_phys_b1000_joint_s{}_preds.npz")]
print(f"{'model':<40}{'pooled':>16}{'1v-5':>8}{'1v-1':>8}{'rho(pred,M)':>13}")
for lab, pat in ROWS:
    p, ps, s5, s1, rm = scores(load(pat))
    print(f"{lab:<40}{p:>9.3f} ± {ps:.3f}{s5:>8.3f}{s1:>8.3f}{rm:>+13.2f}")


def fl(a, b, c, n=100000):
    ra, rb, rc = (stats.rankdata(v) for v in (a, b, c))
    Z = np.column_stack([np.ones(len(rc)), rc])
    M = np.eye(len(rc)) - Z @ np.linalg.pinv(Z)
    ea, eb = M @ ra, M @ rb
    st = lambda x, y: np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
    obs = st(ea, eb)
    fit = rb - eb
    cnt = sum(abs(st(ea, M @ (fit + rng.permutation(eb)))) >= abs(obs) - 1e-12
              for _ in range(n))
    return obs, (cnt + 1) / (n + 1)


def ens(pat):
    ds = load(pat)
    names = [key(n) for n in ds[0]["obs_name"]]
    return (dict(zip(names, np.mean([d["obs_pred"] for d in ds], 0))),
            dict(zip(names, ds[0]["obs_z"])), dict(zip(names, ds[0]["obs_name"])))


print("\n======== real clusters: radio 3 Mpc vs X-ray (25-model ensembles)")
r, rz, rn = ens("fix_phys_b3000_sfimg_s{}_preds.npz")
x, _, _ = ens("fix_x_s{}_preds.npz")
both = sorted(set(r) & set(x))
a = np.array([r[k] for k in both]); b = np.array([x[k] for k in both])
z = np.array([rz[k] for k in both])
rho, p = stats.spearmanr(a, b)
pz, ppz = fl(a, b, z)
wl = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")
m = np.array([k in wl.index for k in both])
pm, ppm = fl(a[m], b[m], np.array([wl.loc[k, "log_m500c"] for k in np.array(both)[m]]))
d = a - b
print(f"  n={len(both)}: Spearman {rho:+.2f} (p={p:.3f}); partial | z {pz:+.2f} (FL p={ppz:.3f}); "
      f"partial | WL mass {pm:+.2f} (FL p={ppm:.3f}, n={m.sum()})")
print(f"  radio - X-ray: median {np.median(d):+.2f} Gyr, {np.mean(d > 0):.0%} radio older; "
      f"medians radio {np.median(a):.2f}, X-ray {np.median(b):.2f}")
print(f"  targets: {[rn[k] for k in both]}")

print("\n======== A399 / A401 ranks (1 = youngest)")
for lab, pat in (("mass only (fixed measurement)", "r4_massonly_s{}_preds.npz"),
                 ("radio 3 Mpc", "fix_phys_b3000_sfimg_s{}_preds.npz"),
                 ("radio 3 Mpc + mass", "r4_imgmass_b3000_s{}_preds.npz"),
                 ("X-ray", "fix_x_s{}_preds.npz"),
                 ("radio 1 Mpc + X-ray", "fix_phys_b1000_joint_s{}_preds.npz")):
    e, _, _ = ens(pat)
    ks = list(e); v = np.array([e[k] for k in ks]); rk = stats.rankdata(v)
    cell = "  ".join(f"{t} {e[t]:.2f} ({int(rk[ks.index(t)])}/{len(ks)})"
                     for t in ("A399", "A401") if t in e)
    print(f"  {lab:<32} {cell}")
