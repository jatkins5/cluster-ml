#!/usr/bin/env python3
"""
Conditional diffusion NN check, stratified by TSC.

Two diagnostics from the same set of nearest-neighbour computations:

1. Visual memorization panel
     Rows = generation TSC bin. Columns alternate (gen, nearest train).
     For each gen sample we find its nearest training image by
     mean-subtracted L2 (the brightness-corrected metric from the
     memorization-metric upgrade). If gen samples at high TSC look
     suspiciously like the few high-TSC training clusters, it shows here.

2. Condition leakage panel
     For each gen TSC bin, histogram of "what TSC does the nearest train
     neighbour have?" overlaid with the overall training TSC distribution.
     If neighbours concentrate near the gen TSC, the condition is doing
     per-sample selection (healthy). If neighbours are uniform across TSC,
     the condition is only shifting aggregate stats.

Also prints per-bin median gen->train NN distance, mean NN train TSC,
and the train-train NN baseline within the same TSC bin (for context).
"""

import argparse

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_diffusion import load_split


def mean_subtract_flat(imgs: np.ndarray) -> np.ndarray:
    """Flatten each image and remove its mean (brightness-corrected L2 space)."""
    flat = imgs.reshape(len(imgs), -1)
    return flat - flat.mean(1, keepdims=True)


def nn(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row of `query`, distance + index of the nearest row in
    `reference` (flat L2). Returns (distances, indices)."""
    distances = np.empty(len(query))
    indices = np.empty(len(query), dtype=np.int64)
    for i in range(len(query)):
        sq_dist = ((reference - query[i]) ** 2).sum(1)
        indices[i] = int(sq_dist.argmin())
        distances[i] = float(np.sqrt(sq_dist.min()))
    return distances, indices


def nn_self(rows: np.ndarray) -> np.ndarray:
    """For each row, distance to its nearest OTHER row (flat L2)."""
    distances = np.empty(len(rows))
    for i in range(len(rows)):
        sq_dist = ((rows - rows[i]) ** 2).sum(1)
        sq_dist[i] = np.inf
        distances[i] = float(np.sqrt(sq_dist.min()))
    return distances


def main(args: argparse.Namespace) -> None:
    d = np.load(args.samples)
    gen = d["samples"]                                 # (N_gen, 1, S, S)
    cond = d["tsc"]                                    # (N_gen,) in Gyr
    print(f"loaded {len(gen)} gen samples in {sorted(set(cond.tolist()))} TSC bins")

    tr_i, tr_l, _, _, attrs = load_split(
        args.data, seed=args.seed,
        labels_path=args.labels, label_key=args.label_key,
        cond_scale_norm=args.cond_scale_norm,
    )
    tr_l_gyr = tr_l * args.cond_scale_norm             # back to Gyr
    print(f"loaded {len(tr_i)} train projections "
          f"(valid TSC labels: {(~np.isnan(tr_l_gyr)).sum()})")

    gen_ms = mean_subtract_flat(gen)
    train_ms = mean_subtract_flat(tr_i)

    print("computing gen->train NN ...")
    g_nn_d, g_nn_i = nn(gen_ms, train_ms)
    print("computing train-train baseline NN ...")
    t_nn_d = nn_self(train_ms)

    g_nn_tsc = tr_l_gyr[g_nn_i]                        # TSC of the NN training image

    unique = sorted(set(float(t) for t in cond.tolist()))
    n_tsc = len(unique)
    half = (unique[1] - unique[0]) / 2 if n_tsc > 1 else 0.5
    valid_tr = ~np.isnan(tr_l_gyr)

    # quantitative summary
    print(f"\n{'gen TSC':>8} {'n':>4} {'gen->train NN':>14} "
          f"{'<NN train TSC>':>16} {'train-train NN @ same TSC':>26}")
    for t in unique:
        m = np.isclose(cond, t)
        # train-train at this TSC stratum
        tr_at_t = valid_tr & (tr_l_gyr >= t - half) & (tr_l_gyr < t + half)
        baseline = np.median(t_nn_d[tr_at_t]) if tr_at_t.sum() >= 3 else np.nan
        nn_tsc_mean = float(np.nanmean(g_nn_tsc[m]))
        nn_tsc_std = float(np.nanstd(g_nn_tsc[m]))
        print(f"{t:8.2f} {int(m.sum()):4d} {np.median(g_nn_d[m]):14.3f} "
              f"{nn_tsc_mean:8.2f} ± {nn_tsc_std:4.2f}  {baseline:26.3f}")

    # ===== figure 1: visual NN panel stratified by TSC =====
    n_show = min(4, int(sum(np.isclose(cond, unique[0]))))
    fig, ax = plt.subplots(n_tsc, 2 * n_show,
                           figsize=(2.4 * 2 * n_show, 2.5 * n_tsc),
                           squeeze=False)
    for r, t in enumerate(unique):
        idx = np.where(np.isclose(cond, t))[0][:n_show]
        for c, i in enumerate(idx):
            ax_g = ax[r, 2 * c]
            ax_n = ax[r, 2 * c + 1]
            ax_g.imshow(gen[i, 0], cmap="inferno", vmin=-1, vmax=1)
            ax_g.set_xticks([]); ax_g.set_yticks([])
            ax_g.set_title(f"gen TSC={t:.1f}\nNN d={g_nn_d[i]:.1f}",
                           fontsize=8)
            ax_n.imshow(tr_i[g_nn_i[i], 0], cmap="inferno",
                        vmin=-1, vmax=1)
            ax_n.set_xticks([]); ax_n.set_yticks([])
            ax_n.set_title(f"NN train\nTSC={g_nn_tsc[i]:.1f}",
                           fontsize=8)
    fig.suptitle("Conditional NN check: gen (left of each pair) vs "
                 "nearest training image (right), by gen TSC row",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out_nn, dpi=110)
    plt.close(fig)
    print(f"saved {args.out_nn}")

    # ===== figure 2: condition-leakage histograms =====
    fig2, ax2 = plt.subplots(1, n_tsc, figsize=(3.4 * n_tsc, 4.0),
                             sharey=True, squeeze=False)
    tr_v = tr_l_gyr[valid_tr]
    bins = np.linspace(tr_v.min(), tr_v.max(), 18)
    for k, t in enumerate(unique):
        a = ax2[0, k]
        m = np.isclose(cond, t)
        a.hist(tr_v, bins=bins, color="C7", alpha=0.4,
               label="train (all)", density=True)
        a.hist(g_nn_tsc[m], bins=bins, color="C1", alpha=0.75,
               label="NN-train TSC", density=True)
        a.axvline(t, color="red", ls="--", lw=1.5,
                  label=f"gen cond = {t:.1f}")
        a.set_xlabel("TSC (Gyr)")
        if k == 0:
            a.set_ylabel("density")
        a.set_title(f"gen TSC = {t:.1f}", fontsize=10)
        if k == n_tsc - 1:
            a.legend(fontsize=8, loc="upper right")
    fig2.suptitle("Condition leakage: do gen samples find their "
                  "nearest train neighbours at the right TSC?", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(args.out_leak, dpi=110)
    plt.close(fig2)
    print(f"saved {args.out_leak}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=str,
                   default="diffusion_out_cond/samples_cond.npz")
    p.add_argument("--data", type=str, default="diffusion_radio_64_v2.h5")
    p.add_argument("--labels", type=str,
                   default="TSC_Cutimages/TSC_eachhalo_snap99.hdf5")
    p.add_argument("--label-key", type=str, default="tsc_gyr")
    p.add_argument("--cond-scale-norm", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-nn", type=str,
                   default="diffusion_out_cond/nn_check_cond.png")
    p.add_argument("--out-leak", type=str,
                   default="diffusion_out_cond/cond_leakage.png")
    main(p.parse_args())
