"""Rank-correlation summary of the merger-definition A/B retrains.

R2 depends on the spread of the target, and the major-merger clock is much
wider than the any-ratio one, so the comparison is repeated with Spearman
rho and MAE to rule out a variance artefact.
"""
import glob
import os

import numpy as np
from scipy import stats

RUNS = [("any_on_major", "any ratio (control, same 255 halos)"),
        ("mr010", "mass ratio > 0.10"),
        ("mr020", "mass ratio > 0.20"),
        ("mr033", "mass ratio > 1/3 (major merger)")]


def cluster_mean(y, yhat, halo):
    """Collapse the 3 projections per cluster before scoring."""
    out_y, out_p = [], []
    for h in np.unique(halo):
        m = halo == h
        out_y.append(y[m].mean())
        out_p.append(yhat[m].mean())
    return np.array(out_y), np.array(out_p)


print(f"{'label':<38} {'n':>4} {'spread':>14} {'R2':>8} {'rho':>7} "
      f"{'MAE':>7} {'MAE/std':>8}")
for tag, desc in RUNS:
    path = f"cnn_major_128/oof_{tag}.npz"
    if not os.path.exists(path):
        print(f"{desc:<38} MISSING")
        continue
    d = np.load(path)
    y, yhat, halo = d["y"], d["yhat"].ravel(), d["halo"]
    y, yhat = cluster_mean(y, yhat, halo)
    r2 = 1 - np.sum((yhat - y) ** 2) / np.sum((y - y.mean()) ** 2)
    rho = stats.spearmanr(yhat, y).statistic
    mae = np.mean(np.abs(yhat - y))
    print(f"{desc:<38} {len(y):>4} "
          f"{f'{y.std():.2f} Gyr':>14} {r2:>8.3f} {rho:>7.3f} "
          f"{mae:>7.3f} {mae / y.std():>8.3f}")

base = "cnn_camels_128/oof_tng_only.npz"
if os.path.exists(base):
    d = np.load(base)
    y, yhat = cluster_mean(d["y"], d["yhat"].ravel(), d["halo"])
    r2 = 1 - np.sum((yhat - y) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"\n{'any ratio, all 352 halos (prior baseline)':<38} {len(y):>4} "
          f"{f'{y.std():.2f} Gyr':>14} {r2:>8.3f} "
          f"{stats.spearmanr(yhat, y).statistic:>7.3f}")
