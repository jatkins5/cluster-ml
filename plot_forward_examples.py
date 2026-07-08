"""Gallery: real LoTSS cutouts (with point sources) vs point-source-free
mock-LoTSS maps, both on the common 1 Mpc / 128px grid in S/N units."""
import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from forward_model_lotss import load_obs, make_mock

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_512.h5")
    ap.add_argument("--lotss-glob",
                    default=os.path.expanduser(
                        "~/data/cluster-ml/lotss_images/*.fits"))
    ap.add_argument("--targets-csv",
                    default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--catalog", default="Radio_Data/TNG-Cluster_Catalog.hdf5")
    ap.add_argument("--box-kpc", type=float, default=1000.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="forward_examples.png")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    obs = load_obs(args.lotss_glob, args.targets_csv, args.box_kpc)

    with h5py.File(args.dataset, "r") as f:
        halo_ids = f["meta/halo_id"][:]
        r500_kpc = f["meta/r500c_kpc"][:]
        tsc = f["labels/pseudo_tsc"][:]
        # spread example clusters across the TSC range
        order = np.argsort(tsc)
        pick = np.sort(order[np.linspace(5, len(order) - 6, args.n,
                                         dtype=int)])
        images = f["images"][pick, 0]           # first projection
    with h5py.File(args.catalog, "r") as f:
        m500_map = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))

    n_cols = 4
    n_rows = 2 * int(np.ceil(args.n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.4 * n_rows))
    half = n_rows // 2

    for k in range(args.n):
        ax = axes[k // n_cols, k % n_cols]
        o = obs[k % len(obs)]
        ax.imshow(o["map"] / o["rms"], vmin=-2, vmax=15, cmap="viridis",
                  origin="lower")
        ax.set_title(f"LoTSS {o['name']}  z={o['z']:.3f}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])

    for k, i in enumerate(pick):
        ax = axes[half + k // n_cols, k % n_cols]
        j = rng.integers(0, len(obs))
        z, rms = obs[j]["z"], obs[j]["rms"]
        px_kpc = 8.0 * r500_kpc[i] / 512
        mock = make_mock(images[k], px_kpc, m500_map[halo_ids[i]],
                         z, rms, args.box_kpc, rng)
        ax.imshow(mock / rms, vmin=-2, vmax=15, cmap="viridis",
                  origin="lower")
        ax.set_title(f"mock halo {halo_ids[i]}  TSC={tsc[i]:.1f}  "
                     f"z={z:.3f}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])

    fig.suptitle("top: real LoTSS (point sources present)   "
                 "bottom: mocks (diffuse emission only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
