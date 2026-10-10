"""Results of the review fixes (F1 radio recipe, F5 X-ray count model, F3
single-view scoring), snapshots 99+91, scored on the snapshot-99 clusters.

Per configuration, over seeds 42-46:
  3view   cluster-mean R2 as before (3 projections pooled, realizations
          averaged) -- an information ceiling, not what real data get
  1view   single projection repeated, single realization, each held-out
          sample scored by the 5-seed ensemble of ITS outer fold (all five
          models excluded that halo; GroupKFold folds are seed-independent)
  rho(M)  Spearman of the 3view prediction with M500
Real clusters: radio vs X-ray agreement and offset with equal-size 5-model
ensembles (one outer fold's 5 seed models), summarised over the 5 folds,
and with the full 25-model ensemble, labelled separately.
"""
import glob
import os

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

SEEDS = range(42, 47)
key = lambda s: "".join(str(s).split()).replace("_", "").upper()


def load(pat):
    return {s: np.load(f"cnn_preds/{pat.format(s)}", allow_pickle=True)
            for s in SEEDS if os.path.exists(f"cnn_preds/{pat.format(s)}")}


def scores(ds):
    if not ds:
        return None
    r3, rm = [], []
    for d in ds.values():
        m = d["snap"] == 99
        r3.append(r2_score(d["tsc"][m], d["oof_cluster"][m]))
        rm.append(stats.spearmanr(d["oof_cluster"][m], np.asarray(d["sim_logm"])[m])[0])
    out = dict(n=len(ds), r3=np.mean(r3), r3s=np.std(r3, ddof=1) if len(r3) > 1 else 0,
               rhoM=np.mean(rm))
    d0 = next(iter(ds.values()))
    if "oof_single" in d0:
        R = len(d0["oof_sample"]) // len(d0["tsc"])
        snap_s = np.repeat(d0["snap"], R)
        m = snap_s == 99
        ens = np.mean([d["oof_single"] for d in ds.values()], axis=0)   # (N*R, P)
        y = d0["y_sample"]
        out["r1"] = r2_score(np.repeat(y[m][:, None], ens.shape[1], 1).ravel(),
                             ens[m].ravel())
        per = [r2_score(np.repeat(d["y_sample"][m][:, None], ens.shape[1], 1).ravel(),
                        d["oof_single"][m].ravel()) for d in ds.values()]
        out["r1_single_model"] = np.mean(per)
    return out


def real(ds):
    d0 = next(iter(ds.values()))
    names = [key(n) for n in d0["obs_name"]]
    full = np.mean([d["obs_pred"] for d in ds.values()], axis=0)
    folds = (np.mean([d["obs_pred_folds"] for d in ds.values()], axis=0)
             if "obs_pred_folds" in d0 else None)          # (folds, n_obs)
    return names, full, folds


def agree(rad, xr):
    (rn, rf, rF), (xn, xf, xF) = real(rad), real(xr)
    both = [k for k in rn if k in xn]
    ri = [rn.index(k) for k in both]
    xi = [xn.index(k) for k in both]
    res = dict(n=len(both), rho25=stats.spearmanr(rf[ri], xf[xi])[0],
               off25=np.median(rf[ri] - xf[xi]), older=np.mean(rf[ri] > xf[xi]))
    if rF is not None and xF is not None:
        rh = [stats.spearmanr(rF[k][ri], xF[k][xi])[0] for k in range(len(rF))]
        of = [np.median(rF[k][ri] - xF[k][xi]) for k in range(len(rF))]
        res.update(rho5=np.mean(rh), rho5s=np.std(rh), off5=np.mean(of))
    return res


CONFIGS = [
    ("radio, legacy recipe 1 Mpc (before)", "s9991_sfimg_s{}_preds.npz"),
    ("radio, physical 1 Mpc", "fix_phys_b1000_sfimg_s{}_preds.npz"),
    ("radio, physical 2 Mpc", "fix_phys_b2000_sfimg_s{}_preds.npz"),
    ("radio, physical 3 Mpc", "fix_phys_b3000_sfimg_s{}_preds.npz"),
    ("radio, arcsinh arm 1 Mpc", "fix_asinh_b1000_sfimg_s{}_preds.npz"),
    ("radio, arcsinh arm 3 Mpc", "fix_asinh_b3000_sfimg_s{}_preds.npz"),
    ("radio+mass, legacy 1 Mpc (before)", "s9991_sfimgmassn_s{}_preds.npz"),
    ("radio+mass, physical 1 Mpc", "fix_phys_b1000_sfimgmassn_s{}_preds.npz"),
    ("radio+mass, physical 2 Mpc", "fix_phys_b2000_sfimgmassn_s{}_preds.npz"),
    ("radio+mass, physical 3 Mpc", "fix_phys_b3000_sfimgmassn_s{}_preds.npz"),
    ("X-ray, old count model (before)", "lxcal_x_s{}_preds.npz"),
    ("X-ray, real count model", "fix_x_s{}_preds.npz"),
    ("joint, before (legacy radio, old X-ray)", "lxcal_joint_s{}_preds.npz"),
    ("joint, physical 1 Mpc", "fix_phys_b1000_joint_s{}_preds.npz"),
    ("joint, physical 2 Mpc", "fix_phys_b2000_joint_s{}_preds.npz"),
    ("joint, physical 3 Mpc", "fix_phys_b3000_joint_s{}_preds.npz"),
]
print(f"{'configuration':<42}{'n':>3}{'3view R2':>14}{'1view R2 (5-ens)':>18}"
      f"{'1view (1 model)':>16}{'rho(pred,M)':>13}")
loaded = {}
for lab, pat in CONFIGS:
    ds = load(pat)
    loaded[lab] = ds
    sc = scores(ds)
    if sc is None:
        print(f"{lab:<42}  (no runs)")
        continue
    one = f"{sc['r1']:.3f}" if "r1" in sc else "-"
    one1 = f"{sc['r1_single_model']:.3f}" if "r1_single_model" in sc else "-"
    print(f"{lab:<42}{sc['n']:>3}{sc['r3']:>9.3f} ±{sc['r3s']:.3f}{one:>18}{one1:>16}"
          f"{sc['rhoM']:>+13.2f}")

print("\nreal clusters, radio vs X-ray (X-ray: real count model)")
xr = loaded.get("X-ray, real count model")
for lab in [l for l, _ in CONFIGS if l.startswith("radio, ")]:
    rad = loaded.get(lab)
    if not rad or not xr:
        continue
    a = agree(rad, xr)
    five = (f"5-model rho {a['rho5']:+.2f} ± {a['rho5s']:.2f}, offset {a['off5']:+.2f} Gyr; "
            if "rho5" in a else "")
    print(f"  {lab:<38} n={a['n']}: {five}25-model rho {a['rho25']:+.2f}, offset "
          f"{a['off25']:+.2f} Gyr ({a['older']:.0%} radio older)")
xo = loaded.get("X-ray, old count model (before)")
rl = loaded.get("radio, legacy recipe 1 Mpc (before)")
if xo and rl:
    a = agree(rl, xo)
    print(f"  {'before: legacy radio vs old X-ray':<38} n={a['n']}: 25-model rho "
          f"{a['rho25']:+.2f}, offset {a['off25']:+.2f} Gyr ({a['older']:.0%} radio older)")
