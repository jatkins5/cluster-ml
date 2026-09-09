"""Summarize the A3 re-runs: what honest checkpoint selection costs.

The numbers being replaced were obtained by keeping the epoch with the best
score on the very fold then reported:

  pooled CNN, pseudo-TSC, n_H<1e-4, 128px : OOF R2 0.567 +- 0.018 (5 seeds)
  field-injected transfer, n_H<1e-4       : mock OOF R2 0.426 (2 seeds)

Here "inner" selects on a held-out 15% of each training fold and "final"
takes the last epoch. Seeds are matched across the two protocols, so the
comparison between them is paired.
"""
import glob
import os
import re

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

OLD_BASE = 0.567          # 5-seed mean, best-epoch-on-test-fold
OLD_TRANS = 0.426


def load(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"_(inner|final)_s(\d+)", p)
        if m:
            out[(m.group(1), int(m.group(2)))] = np.load(p, allow_pickle=True)
    return out


print("======== A3a: pooled CNN baseline (dataset_nh4_128, pseudo-TSC)")
base = load("cnn_preds/pooled_nh4_*_s*.npz")
if not base:
    raise SystemExit("no baseline predictions yet")
rows = {}
for (sel, seed), d in sorted(base.items()):
    r2 = r2_score(d["label"], d["oof_pred"])
    rows.setdefault(sel, {})[seed] = r2
    print(f"  {sel:<6} seed {seed}  OOF R2 {r2:+.4f}   "
          f"per-fold {np.mean(d['fold_r2']):+.4f} +- {np.std(d['fold_r2']):.4f}")
for sel, d in rows.items():
    v = np.array(list(d.values()))
    print(f"\n  {sel:<6} mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
          f"n={len(v)}   vs old 0.567: {v.mean() - OLD_BASE:+.4f}")
if len(rows) == 2:
    seeds = sorted(set(rows["inner"]) & set(rows["final"]))
    d = np.array([rows["inner"][s] - rows["final"][s] for s in seeds])
    t, p = stats.ttest_rel([rows["inner"][s] for s in seeds],
                           [rows["final"][s] for s in seeds])
    print(f"\n  paired inner - final over {len(seeds)} seeds: "
          f"{d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f} (SE), "
          f"p={p:.3f}, {np.sum(d > 0)}/{len(d)} positive")

print("\n======== A3b: field-injected transfer (injected_nh4)")
tr = load("cnn_preds/injtrans_v2_*_s*.npz")
for (sel, seed), d in sorted(tr.items()):
    r2 = r2_score(d["tsc"], d["oof_cluster"])
    op = d["obs_pred"]
    print(f"  {sel:<6} seed {seed}  mock OOF R2 {r2:+.4f}  (old {OLD_TRANS:+.3f})"
          f"   real: median {np.median(op):.2f}  sd {op.std():.2f}  "
          f"range {op.min():.2f}-{op.max():.2f}")
if tr:
    names = [str(s) for s in list(tr.values())[0]["obs_name"]]
    keys = sorted(tr)
    M = np.array([tr[k]["obs_pred"] for k in keys])
    if len(keys) > 1:
        rho = [stats.spearmanr(M[i], M[j])[0]
               for i in range(len(keys)) for j in range(i + 1, len(keys))]
        print(f"\n  cross-run Spearman on the 17 real clusters: "
              f"mean {np.mean(rho):.2f}  min {np.min(rho):.2f}")
    # Does the mass/brightness confound survive the protocol fix?
    import h5py
    with h5py.File("injected_nh4.h5") as f:
        obs = f["obs/images"][:]
    tot = np.sinh(obs.astype(np.float64)).sum(axis=(1, 2))
    with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
        mmap = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
    print("\n  confound check (was rho(pred,M500)=-0.87, "
          "rho(real pred,total SNR)=-0.83):")
    for k in keys:
        d = tr[k]
        m = np.array([mmap[h] for h in d["halo_id"]])
        print(f"    {k[0]:<6} seed {k[1]}  rho(oof,M500) "
              f"{stats.spearmanr(d['oof_cluster'], m)[0]:+.2f}   "
              f"rho(oof,TSC) {stats.spearmanr(d['oof_cluster'], d['tsc'])[0]:+.2f}"
              f"   rho(real pred, total SNR) "
              f"{stats.spearmanr(d['obs_pred'], tot)[0]:+.2f}")
