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
import glob
import re
import os
import pickle

import h5py
import numpy as np
from astropy.cosmology import Planck15
from astropy import units as u

from project_cells import project_cells

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


def load_group_centers(gc_dir):
    """halo_id -> (centre_kpc, box_kpc) from the snap99 FOF catalogue.

    Group arrays are concatenated across chunks in order, so the global group
    index is the FOF halo id. Verified exactly: groupcat Group_R_Crit500
    divided by the catalogue's r500c is 1.000 across the 5-95th percentile.
    Coordinates are ckpc/h, matching Radio_generation.py's pos = Coords * a/h0.
    """
    files = sorted(glob.glob(os.path.join(gc_dir, "fof_subhalo_tab_*.hdf5")),
                   key=lambda p: int(re.search(r"\.(\d+)\.hdf5$", p).group(1)))
    if not files:
        raise SystemExit(f"no FOF catalogue chunks under {gc_dir}")
    pos, box, hpar, a = [], None, None, None
    for fn in files:
        with h5py.File(fn, "r") as f:
            if box is None:
                box = float(f["Header"].attrs["BoxSize"])
                hpar = float(f["Header"].attrs["HubbleParam"])
                a = float(f["Header"].attrs["Time"])
            if f["Header"].attrs["Ngroups_ThisFile"] == 0:
                continue
            pos.append(f["Group/GroupPos"][:])
    conv = a / hpar
    return np.concatenate(pos) * conv, box * conv


def load_catalog(catalog_path, radius_field="r500c"):
    """Return dict halo_id -> radius_kpc for the requested radius field."""
    with h5py.File(catalog_path, "r") as f:
        halo_ids = f["haloID"][:]
        radii    = f[radius_field][:]   # Mpc
    return {int(hid): float(r) * 1000.0 for hid, r in zip(halo_ids, radii)}


def main(img_size, extent_r500, output_path, extent_r200=None,
         center_mode="grouppos", groupcat=None, cells_dir=None,
         spread_cells=False):
    radio_dir    = "Radio_Data"
    catalog_path = os.path.join(radio_dir, "TNG-Cluster_Catalog.hdf5")
    pkl_path     = "feats_labels_dict_tngcluster.pkl"

    print("Loading catalog and labels...")
    if extent_r200 is not None:
        r500c_map = load_catalog(catalog_path, "r200c")
        extent_r500 = extent_r200
        print(f"Using R200c-scaled FOV: +-{extent_r200} R200c")
    else:
        r500c_map = load_catalog(catalog_path)

    group_pos = box_kpc = None
    if center_mode == "grouppos":
        print(f"Loading halo centres from {groupcat} ...")
        group_pos, box_kpc = load_group_centers(groupcat)
        print(f"  {len(group_pos)} groups, box {box_kpc:.4g} kpc")

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

    cell_map = {}
    if cells_dir is not None:
        for fn in os.listdir(cells_dir):
            m = re.match(r"cells_FOF(\d+)_sub\d+\.npz", fn)
            if m:
                cell_map[int(m.group(1))] = os.path.join(cells_dir, fn)
        print(f"Using volume-spread cells from {cells_dir} "
              f"({len(cell_map)} files)")

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
        if cells_dir is not None:
            # Regenerated by build_radio_cells.py: same weights, plus each
            # cell's equivalent-sphere radius so it can be spread over its
            # own volume instead of dumped into one pixel.
            cf = cell_map.get(halo_id)
            if cf is None:
                raise SystemExit(f"no cell file for halo {halo_id} in {cells_dir}")
            data = np.load(cf)
            r_kpc = data["r_kpc"] if spread_cells else None
        else:
            data = np.load(npz_map[halo_id])
            r_kpc = None
        pos    = data["pos"]   # (N_p, 3) physical kpc
        w      = data["w"]     # (N_p,)

        # Cluster centre. The weight-averaged position is NOT a halo centre:
        # one gas cell carries a median 29.5% of the total linear weight, so
        # the centroid tracks the brightest shock cell and lands a median
        # 0.70 r500 (836 kpc) from the halo, with 33% of clusters beyond
        # 1 r500 and 1.1% outside the image entirely. It also correlates with
        # the label (Spearman +0.19 vs pseudo-TSC), so centring on it removes
        # a real merger clock from every image. Default to GroupPos.
        if center_mode == "grouppos":
            if halo_id >= len(group_pos):
                raise SystemExit(f"halo {halo_id} is beyond the {len(group_pos)} "
                                 f"groups in the catalogue; refusing to guess")
            center = group_pos[halo_id]
            # Unwrap the periodic box, otherwise a halo straddling an edge
            # gets shifted by a full box length.
            pos = pos - center
            pos -= box_kpc * np.round(pos / box_kpc)
            center = np.zeros(3)
        else:
            w_sum  = w.sum()
            center = ((pos * w[:, None]).sum(axis=0) / w_sum if w_sum > 0
                      else pos.mean(axis=0))

        # image half-width in kpc
        r500c_kpc  = r500c_map[halo_id]
        half_width = extent_r500 * r500c_kpc

        if r_kpc is None:
            images = project_image(pos, w, center, half_width, img_size)
        else:
            lin = project_cells(pos, w, r_kpc, center, half_width, img_size)
            images = np.arcsinh(lin).astype(np.float32)

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
        f.attrs["center_mode"]     = center_mode
        f.attrs["cells_dir"]       = cells_dir or ""
        f.attrs["spread_cells"]    = bool(spread_cells)
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
    parser.add_argument("--extent-r200", type=float, default=None,
                        help="Image half-width in units of R200c; overrides "
                             "--extent-r500 (use to match the CAMELS FOV)")
    parser.add_argument("--center", choices=["grouppos", "weight"],
                        default="grouppos",
                        help="image centre: the FOF halo position (default) "
                             "or the legacy weight-averaged position, which "
                             "sits a median 0.70 r500 off the halo")
    parser.add_argument("--groupcat", type=str,
                        default="/oscar/data/idellant/Chuiyang/"
                                "groupcat_classification/groupcat_099",
                        help="snap99 FOF catalogue directory, for --center grouppos")
    parser.add_argument("--cells-dir", type=str, default=None,
                        help="directory of build_radio_cells.py output to use "
                             "as the particle source instead of Radio_Data "
                             "(e.g. a gas-phase-cut regeneration)")
    parser.add_argument("--spread-cells", action="store_true",
                        help="deposit each cell over its own volume rather "
                             "than into one pixel. Measured to be a no-op on "
                             "the linear map and mildly harmful on the "
                             "compressed one, so off by default")
    parser.add_argument("--output",      type=str,   default="dataset.h5",
                        help="Output HDF5 file path (default: dataset.h5)")
    args = parser.parse_args()

    main(args.img_size, args.extent_r500, args.output, args.extent_r200,
         args.center, args.groupcat, args.cells_dir,
         args.spread_cells)
