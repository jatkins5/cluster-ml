#!/usr/bin/env python3
"""
Build ML training dataset from TNG-Cluster radio simulation data.

Pairs each cluster's radio particle data (Radio_Data/radio_FOF*.npz) with
merger labels from feats_labels_dict_tngcluster.pkl, producing 2D projected
radio emission images ready for ML training.

Output: dataset.h5 with groups:
  images/       - (N, 3, IMG_SIZE, IMG_SIZE) float32, one row per cluster,
                  3 projections (xy, yz, xz), arcsinh-normalized
  labels/       - scalar and tau-sweep label arrays (N,)
  meta/         - halo_id, mass_ratio, r500c_kpc per cluster

Usage:
    python build_dataset.py [--img-size 128] [--extent-r500 4.0] [--output dataset.h5]
"""

import argparse
import os
import pickle

import h5py
import numpy as np
from astropy.cosmology import Planck15
from astropy import units as u

# ---------- defaults ----------
IMG_SIZE    = 128      # pixels per side
EXTENT_R500 = 4.0      # half-width of image in units of R500c
SNAP        = 99       # z=0 snapshot


def project_image(pos, w, center, half_width, img_size):
    """
    Project 3D weighted particles onto three 2D planes (xy, yz, xz).

    Parameters
    ----------
    pos        : (N, 3) float, particle positions in kpc
    w          : (N,)   float, particle weights
    center     : (3,)   float, cluster center in kpc
    half_width : float, half-width of image in kpc
    img_size   : int, pixels per side

    Returns
    -------
    images : (3, img_size, img_size) float32
        Arcsinh-normalised projected weight maps for xy, yz, xz planes.
    """
    rel = pos - center  # shift to cluster frame

    # axis pairs for each projection: (horiz_axis, vert_axis)
    projections = [(0, 1), (1, 2), (0, 2)]
    images = np.zeros((3, img_size, img_size), dtype=np.float32)

    edges = np.linspace(-half_width, half_width, img_size + 1)

    for k, (ax0, ax1) in enumerate(projections):
        img, _, _ = np.histogram2d(
            rel[:, ax0], rel[:, ax1],
            bins=edges,
            weights=w,
        )
        # arcsinh normalise: compresses the large dynamic range of w
        images[k] = np.arcsinh(img).astype(np.float32)

    return images


def load_catalog(catalog_path):
    """Return dicts halo_id -> r500c_kpc and halo_id -> origID."""
    with h5py.File(catalog_path, "r") as f:
        halo_ids = f["haloID"][:]
        r500c    = f["r500c"][:]   # Mpc
    return {int(hid): float(r) * 1000.0 for hid, r in zip(halo_ids, r500c)}


def main(img_size, extent_r500, output_path):
    radio_dir    = "Radio_Data"
    catalog_path = os.path.join(radio_dir, "TNG-Cluster_Catalog.hdf5")
    pkl_path     = "feats_labels_dict_tngcluster.pkl"

    print("Loading catalog and labels...")
    r500c_map = load_catalog(catalog_path)

    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)

    # collect tau values from pkl keys (label_score_all_tau*)
    sample_entry = pkl[list(pkl.keys())[0]][SNAP]
    tau_keys_all = sorted(
        [k for k in sample_entry if k.startswith("label_score_all_tau")],
        key=lambda k: float(k.split("tau")[1]),
    )
    tau_keys_pre = [k.replace("_all_", "_pre_") for k in tau_keys_all]
    tau_vals = np.array([float(k.split("tau")[1]) for k in tau_keys_all], dtype=np.float32)
    n_tau = len(tau_vals)

    # discover NPZ files and sort by halo_id
    npz_files = sorted(
        [f for f in os.listdir(radio_dir) if f.startswith("radio_FOF") and f.endswith(".npz")]
    )
    halo_ids_ordered = []
    npz_map = {}
    for fname in npz_files:
        stem     = fname.replace("radio_FOF", "").replace(".npz", "")
        halo_id  = int(stem.split("_sub")[0])
        halo_ids_ordered.append(halo_id)
        npz_map[halo_id] = os.path.join(radio_dir, fname)

    N = len(halo_ids_ordered)
    print(f"Found {N} clusters.")

    # pre-allocate output arrays
    all_images      = np.zeros((N, 3, img_size, img_size), dtype=np.float32)
    all_halo_ids    = np.zeros(N, dtype=np.int64)
    all_mass_ratio  = np.zeros(N, dtype=np.float32)
    all_r500c_kpc   = np.zeros(N, dtype=np.float32)
    all_labels_all  = np.zeros((N, n_tau), dtype=np.float32)  # all-merger score
    all_labels_pre  = np.zeros((N, n_tau), dtype=np.float32)  # pre-merger score
    all_pseudo_tsc  = np.zeros(N, dtype=np.float32)           # interpolated TSC

    for i, halo_id in enumerate(halo_ids_ordered):
        if i % 50 == 0:
            print(f"  Processing {i}/{N}...")

        # load radio particles
        data   = np.load(npz_map[halo_id])
        pos    = data["pos"]   # (N_p, 3) physical kpc
        w      = data["w"]     # (N_p,)

        # cluster center: weight-averaged position
        w_sum  = w.sum()
        center = (pos * w[:, None]).sum(axis=0) / w_sum if w_sum > 0 else pos.mean(axis=0)

        # image half-width in kpc
        r500c_kpc  = r500c_map[halo_id]
        half_width = extent_r500 * r500c_kpc

        images = project_image(pos, w, center, half_width, img_size)

        # load labels from pkl
        entry = pkl[halo_id][SNAP]

        all_images[i]     = images
        all_halo_ids[i]   = halo_id
        all_mass_ratio[i] = float(entry["mass_ratio"])
        all_r500c_kpc[i]  = r500c_kpc
        for j, (ka, kp) in enumerate(zip(tau_keys_all, tau_keys_pre)):
            all_labels_all[i, j] = float(entry[ka])
            all_labels_pre[i, j] = float(entry[kp])

        # pseudo-TSC: tau at which label_score_all first crosses 0.5
        # interpolated linearly between bracketing tau values.
        # Capped at tau_max for quiescent clusters that never reach 0.5.
        score_curve = all_labels_all[i]
        cross_idx = np.argmax(score_curve >= 0.5)
        if score_curve[cross_idx] < 0.5:
            # never crosses — genuinely quiescent, cap at tau_max
            all_pseudo_tsc[i] = float(tau_vals[-1])
        elif cross_idx == 0:
            # already above 0.5 at tau=0.1, interpolate toward 0
            all_pseudo_tsc[i] = float(tau_vals[0])
        else:
            t0, t1 = tau_vals[cross_idx - 1], tau_vals[cross_idx]
            s0, s1 = score_curve[cross_idx - 1], score_curve[cross_idx]
            all_pseudo_tsc[i] = float(t0 + (0.5 - s0) * (t1 - t0) / (s1 - s0))

    print(f"Saving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.attrs["img_size"]        = img_size
        f.attrs["extent_r500"]     = extent_r500
        f.attrs["snapshot"]        = SNAP
        f.attrs["n_clusters"]      = N
        f.attrs["n_projections"]   = 3
        f.attrs["projections"]     = ["xy", "yz", "xz"]
        f.attrs["description"]     = (
            "TNG-Cluster radio emission images (arcsinh-normalised 2D projections) "
            "paired with merger label scores from feats_labels_dict_tngcluster.pkl. "
            "images shape: (N_clusters, 3_projections, H, W). "
            "label_score_all_tau: merger activity score in all-merger (past+future) "
            "time window tau [Gyr]; at snap 99 (z=0) equals label_score_pre_tau."
        )

        # images
        f.create_dataset("images",  data=all_images,     compression="gzip", compression_opts=4)

        # per-cluster metadata
        meta = f.create_group("meta")
        meta.create_dataset("halo_id",     data=all_halo_ids)
        meta.create_dataset("mass_ratio",  data=all_mass_ratio)
        meta.create_dataset("r500c_kpc",   data=all_r500c_kpc)

        # label sweeps
        labels = f.create_group("labels")
        labels.create_dataset("tau_gyr",          data=tau_vals)
        labels.create_dataset("label_score_all",  data=all_labels_all,
                              compression="gzip", compression_opts=4)
        labels.create_dataset("label_score_pre",  data=all_labels_pre,
                              compression="gzip", compression_opts=4)
        labels.create_dataset("pseudo_tsc",       data=all_pseudo_tsc)
        labels.attrs["pseudo_tsc_description"] = (
            "Interpolated tau [Gyr] at which label_score_all first crosses 0.5. "
            "Proxy for time since last major merger. "
            f"Capped at {float(tau_vals[-1]):.1f} Gyr for {int((all_pseudo_tsc == tau_vals[-1]).sum())} "
            "quiescent clusters whose score never reaches 0.5."
        )

    print("Done.")
    print(f"  images:           {all_images.shape}  ({all_images.nbytes/1e6:.1f} MB)")
    print(f"  label tau range:  {tau_vals[0]:.1f} – {tau_vals[-1]:.1f} Gyr  ({n_tau} steps)")
    n_capped = int((all_pseudo_tsc == tau_vals[-1]).sum())
    print(f"  pseudo_tsc:       min={all_pseudo_tsc.min():.2f}  max={all_pseudo_tsc.max():.2f}  "
          f"mean={all_pseudo_tsc.mean():.2f}  ({n_capped} capped at {tau_vals[-1]:.1f} Gyr)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img-size",    type=int,   default=IMG_SIZE,
                        help=f"Image resolution in pixels (default: {IMG_SIZE})")
    parser.add_argument("--extent-r500", type=float, default=EXTENT_R500,
                        help=f"Image half-width in units of R500c (default: {EXTENT_R500})")
    parser.add_argument("--output",      type=str,   default="dataset.h5",
                        help="Output HDF5 file path (default: dataset.h5)")
    args = parser.parse_args()

    main(args.img_size, args.extent_r500, args.output)
