"""Score proxy-label-trained predictions against the TRUE TSC.

The trainer scores each run against whatever label it was trained on, so a
proxy-trained model looks fine on its own target while being useless for the
physical quantity. This rescores the saved OOF predictions against
TSC_eachhalo_snap99, which is the number that decides whether a tree-free
CAMELS label is worth building.

The proxy label itself is included as a row: a model can only beat its own
label if the label's error is independent of what the images show.
"""
import os

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

TRUTH = "TSC_Cutimages/TSC_eachhalo_snap99.hdf5"
RUNS = [("cnn_proxy_128/oof_proxy.npz", "CNN trained on proxy (censored filled)"),
        ("cnn_proxy_128/oof_proxy_unc.npz", "CNN trained on proxy (uncensored only)"),
        ("cnn_camels_128/oof_tng_only.npz", "CNN trained on truth (baseline)")]


def cluster_mean(y, yhat, halo):
    hs = np.unique(halo)
    return (np.array([y[halo == h].mean() for h in hs]),
            np.array([yhat[halo == h].mean() for h in hs]), hs)


def score(name, pred, truth):
    rho = stats.spearmanr(pred, truth).statistic
    r2 = 1 - np.sum((pred - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
    auc = roc_auc_score(truth <= 2.0, -pred)
    print(f"{name:<42} {len(truth):>5} {r2:>8.3f} {rho:>8.3f} {auc:>9.3f}")


def main():
    with h5py.File(TRUTH, "r") as f:
        truth_of = dict(zip(f["halo_id"][:], f["tsc_gyr"][:]))

    print(f"{'model / label':<42} {'n':>5} {'R2':>8} {'rho':>8} "
          f"{'AUC<2Gyr':>9}   (all vs TRUE TSC)")
    for path, desc in RUNS:
        if not os.path.exists(path):
            print(f"{desc:<42} MISSING")
            continue
        d = np.load(path)
        _, pred, hs = cluster_mean(d["y"], d["yhat"].ravel(), d["halo"])
        truth = np.array([truth_of[h] for h in hs])
        ok = np.isfinite(truth) & np.isfinite(pred)
        score(desc, pred[ok], truth[ok])

    with h5py.File("tsc_proxy_snap99.hdf5", "r") as f:
        hid, proxy = f["halo_id"][:], f["tsc_proxy"][:]
    truth = np.array([truth_of[h] for h in hid])
    ok = np.isfinite(truth) & np.isfinite(proxy)
    print()
    score("proxy label itself (ceiling if noise is not", proxy[ok], truth[ok])
    print("  independent of image content)")


if __name__ == "__main__":
    main()
