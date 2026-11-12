#!/usr/bin/env python3
"""Success-only producer manifest for stale-regeneration job B (review round
49): binds dataset_syn_nh4_1400_128.h5 to its radio prescription, builders,
source identities and final hash. Run by run_cells_1400.sh only after both
builders exit successfully; refuses to overwrite.

  manifest_cells_1400.py --dataset D --cells-dir C --radio-dir R --cutout-dir U \
      --ratio-table T --nu-ghz 1.4 --w-scale 16.48 --max-nh 1e-4 --img-size 128
"""
import argparse
import glob
import json
import os
import re
import subprocess

import h5py

import prov_utils as PU

HELPERS = ("build_radio_cells.py", "build_dataset.py", "manifest_cells_1400.py", "prov_utils.py")
EXPECT_ATTRS = dict(snapshot=99, redshift=0.0, center_mode="grouppos", extent_r500=4.0, spread_cells=False,
                    n_clusters=352, n_projections=3, projections=["xy", "yz", "xz"])


def main():
    ap = argparse.ArgumentParser()
    for k in ("dataset", "cells-dir", "radio-dir", "cutout-dir", "ratio-table"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--nu-ghz", type=float, required=True)
    ap.add_argument("--w-scale", type=float, required=True)
    ap.add_argument("--max-nh", type=float, required=True)
    ap.add_argument("--img-size", type=int, required=True)
    a = ap.parse_args()
    out = a.dataset + ".manifest.json"
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    with h5py.File(a.dataset) as f:
        at = {k: (f.attrs[k].tolist() if hasattr(f.attrs[k], "tolist") else f.attrs[k]) for k in f.attrs}
        shape = f["images"].shape
        hid = [int(h) for h in f["meta/halo_id"][:]]
    at = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in at.items()}
    at["projections"] = [p.decode() if isinstance(p, bytes) else str(p) for p in at.get("projections", [])]
    bad = {k: (at.get(k), v) for k, v in EXPECT_ATTRS.items() if at.get(k) != v}
    if bad or at.get("img_size") != a.img_size or at.get("cells_dir") != a.cells_dir \
            or shape != (352, 3, a.img_size, a.img_size) or len(set(hid)) != len(hid):
        raise SystemExit(f"dataset does not match the declared protocol: {bad}, img {at.get('img_size')}, "
                         f"cells_dir {at.get('cells_dir')}, shape {shape}")
    cells = sorted(glob.glob(os.path.join(a.cells_dir, "cells_FOF*_sub*.npz")))
    fofs = sorted(int(re.search(r"cells_FOF(\d+)_", os.path.basename(c)).group(1)) for c in cells)
    if fofs != sorted(hid):
        raise SystemExit("cell files do not cover exactly the dataset's parents")
    here = os.path.dirname(os.path.abspath(__file__))
    man = dict(status="complete", kind="cells_1400",
               prescription=dict(model="synch", nu_ghz=a.nu_ghz, w_scale=a.w_scale, max_nh=a.max_nh,
                                 snapshot=99, z=0.0, spread_cells=False),
               dataset=dict(path=a.dataset, sha256=PU.sha(a.dataset), attrs=at, shape=list(shape)),
               cells=PU.dir_digest(a.cells_dir), ratio_table=dict(path=a.ratio_table, sha256=PU.sha(a.ratio_table)),
               radio_dir=a.radio_dir, cutout_dir=a.cutout_dir,
               cutouts_partial={os.path.basename(p): PU.partial_fp(p)
                                for p in sorted(glob.glob(os.path.join(a.cutout_dir, "cutout_*.hdf5")))
                                if int(re.search(r"_FOF(\d+)\.hdf5$", p).group(1)) in set(hid)},
               helpers={h: PU.sha(os.path.join(here, h)) for h in HELPERS},
               head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                   text=True).stdout.strip(),
               uncommitted=subprocess.run(["git", "status", "--porcelain", "--"] + list(HELPERS), cwd=here,
                                          capture_output=True, text=True).stdout.strip())
    with open(out, "x") as f:
        json.dump(man, f, indent=1)
    print(f"wrote {out}: dataset sha256 {man['dataset']['sha256']}")


if __name__ == "__main__":
    main()
