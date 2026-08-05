"""Build a CAMELS training dataset matching the TNG-Cluster format.

Reads Chuiyang's per-zoom radio maps (shock-cell coordinates in physical
kpc, already centered on the halo, plus radio weights) and the FoF
mass-history-derived TSC labels, and projects along the three axes onto
a fixed grid.

FOV is set in units of R200 because the CAMELS cutouts only extend to
2 R200 -- to combine with TNG-Cluster the two datasets must share the
same scaled field of view (build the TNG side with
`build_dataset.py --extent-r200`).

Output mirrors dataset.h5: images (N,3,S,S) + meta + labels/merger_tsc.
"""
import argparse
import glob
import os

import h5py
import numpy as np

from build_dataset import project_image
from camels_labels import BASE, HIST, TARGET_SNAP, derive_tsc, zoom_cosmology


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--extent-r200", type=float, default=2.0,
                    help="half-width of image in units of R200")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="fractional M200 jump flagged as a major merger")
    ap.add_argument("--use-rate", action="store_true",
                    help="threshold on fractional growth per Gyr instead")
    ap.add_argument("--require-detected", action="store_true",
                    help="drop zooms with no detected merger (censored TSC)")
    ap.add_argument("--min-logm200", type=float, default=14.0,
                    help="drop zooms below this log10 M200 at snap 90; most "
                         "CAMELS zooms are group-scale or smaller")
    ap.add_argument("--output", default="dataset_camels_128.h5")
    args = ap.parse_args()

    radios = sorted(glob.glob(f"{BASE}/Radio/radio_GZ28_*_snap090.hdf5"))
    print(f"radio files: {len(radios)}")

    rows = []
    with h5py.File(HIST, "r") as fh:
        for path in radios:
            zoom = os.path.basename(path).split("radio_")[1].split("_snap")[0]
            if zoom not in fh:
                continue
            g = fh[zoom]
            snap = g["SnapNum"][:]
            i90 = np.where(snap == TARGET_SNAP)[0]
            if len(i90) == 0 or not g["passes_contamination_cut"][i90[0]]:
                continue
            if np.log10(g["M200_Msun"][i90[0]]) < args.min_logm200:
                continue
            om0, h = zoom_cosmology(zoom)
            tsc, njump, detected = derive_tsc(
                g["M200_Msun"][:], snap, g["Redshift"][:],
                g["passes_contamination_cut"][:], om0, h,
                threshold=args.threshold, use_rate=args.use_rate)
            if args.require_detected and not detected:
                continue
            if not np.isfinite(tsc):
                continue
            rows.append(dict(zoom=zoom, path=path, tsc=float(tsc),
                             detected=bool(detected), n_jumps=int(njump),
                             m200=float(g["M200_Msun"][i90[0]]),
                             om0=om0, h=h if h else np.nan))

    N = len(rows)
    S = args.img_size
    print(f"usable zooms: {N}")

    images = np.zeros((N, 3, S, S), dtype=np.float32)
    tsc = np.zeros(N, dtype=np.float32)
    detected = np.zeros(N, dtype=bool)
    n_jumps = np.zeros(N, dtype=np.int32)
    r200_kpc = np.zeros(N, dtype=np.float32)
    m200 = np.zeros(N, dtype=np.float64)
    om0 = np.zeros(N, dtype=np.float32)
    hval = np.zeros(N, dtype=np.float32)
    zoom_ids = np.zeros(N, dtype=np.int64)
    n_cells = np.zeros(N, dtype=np.int64)

    for i, r in enumerate(rows):
        if i % 100 == 0:
            print(f"  {i}/{N}...")
        with h5py.File(r["path"], "r") as f:
            pos = f["Radio/Coordinates"][:]
            w = f["Radio/Power"][:]
            r200 = float(f["Header"].attrs["R200_kpc"])
        half_width = args.extent_r200 * r200
        # coordinates are already relative to the halo center
        images[i] = project_image(pos, w, np.zeros(3), half_width, S)
        tsc[i] = r["tsc"]
        detected[i] = r["detected"]
        n_jumps[i] = r["n_jumps"]
        r200_kpc[i] = r200
        m200[i] = r["m200"]
        om0[i] = r["om0"]
        hval[i] = r["h"]
        zoom_ids[i] = int(r["zoom"].split("_")[1])
        n_cells[i] = len(w)

    empty = (images.sum(axis=(1, 2, 3)) <= 0)
    if empty.any():
        print(f"WARNING: {empty.sum()} zooms produced empty images")

    print(f"\nTSC: min={tsc.min():.2f} med={np.median(tsc):.2f} "
          f"max={tsc.max():.2f} Gyr  ({detected.sum()}/{N} with a "
          f"detected merger)")
    print(f"M200 log10: {np.log10(m200.min()):.2f} - "
          f"{np.log10(m200.max()):.2f}")
    print(f"shock cells per zoom: min={n_cells.min()} "
          f"med={int(np.median(n_cells))} max={n_cells.max()}")

    with h5py.File(args.output, "w") as f:
        f.create_dataset("images", data=images, compression="gzip")
        m = f.create_group("meta")
        m.create_dataset("zoom_id", data=zoom_ids)
        m.create_dataset("r200_kpc", data=r200_kpc)
        m.create_dataset("m200_msun", data=m200)
        m.create_dataset("omega0", data=om0)
        m.create_dataset("hubble", data=hval)
        m.create_dataset("n_shock_cells", data=n_cells)
        lab = f.create_group("labels")
        lab.create_dataset("merger_tsc", data=tsc)
        lab.create_dataset("detected", data=detected)
        lab.create_dataset("n_jumps", data=n_jumps)
        f.attrs["extent_r200"] = args.extent_r200
        f.attrs["threshold"] = args.threshold
        f.attrs["use_rate"] = args.use_rate
        f.attrs["snap"] = TARGET_SNAP
        f.attrs["source"] = "CAMELS GZ28 zooms (Chuiyang Kong)"
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
