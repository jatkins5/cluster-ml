"""B2: what does the image add once the model is handed halo mass?

Four conditions on matched seeds and identical folds:

  img       image only -- the status quo, where mock brightness is a mass
            proxy by construction (flux anchored to Cuciti P150 ~ M500^3.55)
  imgmass   image + the simulation's true log10 M500c
  imgmassn  image + mass blurred by the real sample's measured error
  massonly  mass alone, images zeroed -- same folds, protocol, architecture

The headline is imgmass minus massonly: what the image contributes once
mass is no longer something it has to infer. The imgmass/imgmassn gap says
whether that contribution survives real mass uncertainty, which is the form
the observational claim would actually take.
"""
import glob
import re

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

ORDER = ["img", "imgmass", "imgmassn", "massonly"]
LABEL = {"img": "image only", "imgmass": "image + mass",
         "imgmassn": "image + blurred mass", "massonly": "mass only"}

runs = {}
for p in sorted(glob.glob("cnn_preds/b2_*_preds.npz")):
    m = re.search(r"b2_([a-z]+)_s(\d+)_preds\.npz$", p)
    runs[(m.group(1), int(m.group(2)))] = np.load(p, allow_pickle=True)
if not runs:
    raise SystemExit("no B2 predictions found")
seeds = sorted({s for _, s in runs})
print(f"conditions {sorted({c for c, _ in runs})}, seeds {seeds}\n")

any_run = next(iter(runs.values()))
tsc, halo = any_run["tsc"], any_run["halo_id"]
logm = any_run["sim_logm"]
q = np.quantile(logm, [1 / 3, 2 / 3])
terciles = [("low", logm <= q[0]), ("mid", (logm > q[0]) & (logm <= q[1])),
            ("high", logm > q[1])]


def partial(pred):
    rk = stats.rankdata
    def resid(a, b):
        A = np.column_stack([b, np.ones_like(b)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return stats.pearsonr(resid(rk(pred), rk(logm)),
                          resid(rk(tsc), rk(logm)))[0]


print("======== mock OOF R2 (cluster-mean)")
print(f"{'condition':<22}" + "".join(f"{f'seed {s}':>10}" for s in seeds)
      + f"{'mean':>9}{'rho(p,M)':>10}{'partial':>9}"
      + "".join(f"{t:>8}" for t, _ in terciles))
mean_r2 = {}
for c in ORDER:
    have = [(c, s) for s in seeds if (c, s) in runs]
    if not have:
        continue
    r2s = [r2_score(tsc, runs[k]["oof_cluster"]) for k in have]
    ens = np.mean([runs[k]["oof_cluster"] for k in have], axis=0)
    mean_r2[c] = np.mean(r2s)
    print(f"{LABEL[c]:<22}" + "".join(f"{v:>10.3f}" for v in r2s)
          + f"{np.mean(r2s):>9.3f}"
          + f"{stats.spearmanr(ens, logm)[0]:>10.3f}"
          + f"{partial(ens):>9.3f}"
          + "".join(f"{r2_score(tsc[b], ens[b]):>8.3f}" for _, b in terciles))

print("\n======== the increment the paper turns on")
for a, b in [("imgmass", "massonly"), ("imgmassn", "massonly"),
             ("imgmass", "img"), ("imgmassn", "imgmass")]:
    if a in mean_r2 and b in mean_r2:
        pa = [r2_score(tsc, runs[(a, s)]["oof_cluster"]) for s in seeds
              if (a, s) in runs and (b, s) in runs]
        pb = [r2_score(tsc, runs[(b, s)]["oof_cluster"]) for s in seeds
              if (a, s) in runs and (b, s) in runs]
        d = np.array(pa) - np.array(pb)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        p = stats.ttest_rel(pa, pb).pvalue if len(d) > 1 else np.nan
        print(f"  {LABEL[a]:<22} - {LABEL[b]:<22} "
              f"{d.mean():+.3f} +- {se:.3f} (SE)  p={p:.4f}  "
              f"{int((d > 0).sum())}/{len(d)} positive")
        print(f"  {'':<22}   {'':<22} per seed "
              f"{', '.join(f'{v:+.3f}' for v in d)}")

print("\n======== real predictions")
print(f"{'condition':<22}{'n':>4}{'median':>9}{'sd':>8}{'min':>8}{'max':>8}"
      f"{'rho(pred,M500)':>16}")
for c in ORDER:
    have = [(c, s) for s in seeds if (c, s) in runs]
    if not have:
        continue
    op = np.mean([runs[k]["obs_pred"] for k in have], axis=0)
    om = runs[have[0]]["obs_logm"]
    r = (stats.spearmanr(op, om)[0] if len(om) == len(op) else np.nan)
    print(f"{LABEL[c]:<22}{len(op):>4}{np.median(op):>9.2f}{op.std():>8.2f}"
          f"{op.min():>8.2f}{op.max():>8.2f}{r:>16.2f}")

print(f"\n{'target':<20}" + "".join(f"{LABEL[c][:11]:>13}" for c in ORDER
                                    if any((c, s) in runs for s in seeds))
      + f"{'log M500c':>11}")
names = [str(s) for s in runs[("imgmass", seeds[0])]["obs_name"]] \
    if ("imgmass", seeds[0]) in runs else []
cols = {}
for c in ORDER:
    have = [(c, s) for s in seeds if (c, s) in runs]
    if have:
        nm = [str(s) for s in runs[have[0]]["obs_name"]]
        cols[c] = dict(zip(nm, np.mean([runs[k]["obs_pred"] for k in have],
                                       axis=0)))
mass_col = dict(zip(names, runs[("imgmass", seeds[0])]["obs_logm"])) \
    if ("imgmass", seeds[0]) in runs else {}
for n in sorted(set().union(*[set(v) for v in cols.values()])):
    row = f"{n:<20}"
    for c in ORDER:
        if c in cols:
            v = cols[c].get(n)
            row += f"{v:>13.2f}" if v is not None else f"{'--':>13}"
    lm = mass_col.get(n)
    row += f"{lm:>11.2f}" if lm is not None else f"{'--':>11}"
    print(row)
