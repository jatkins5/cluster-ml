#!/usr/bin/env python3
"""Two-stage producer record for stale-regeneration job B (review rounds 49-50).

  --stage begin   BEFORE the builders: refuses existing outputs; records every
                  source the builders read (helpers, ratio table, label pickle,
                  TNG catalogue, GroupPos catalogue chunks, the Radio_Data
                  source-ID listing build_dataset uses, the raw radio listing,
                  partial raw-cutout fingerprints) -> <dataset>.begin.json
  --stage end     AFTER both builders: recomputes that record and requires it
                  unchanged, validates the dataset and cell files against the
                  declared protocol, then writes the success manifest
                  <dataset>.manifest.json (exclusive create). Any failure leaves
                  no manifest.
Partial cutout fingerprints are size, mtime and sha256 of the first 16 MiB:
not full-content verification.
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
C = "/oscar/data/idellant/Chuiyang"


def sources_record(a, here):
    gc = sorted(glob.glob(os.path.join(a.groupcat, "fof_subhalo_tab_*.hdf5")))
    return dict(helpers={h: PU.sha(os.path.join(here, h)) for h in HELPERS},
                ratio_table=PU.sha(a.ratio_table), label_pickle=PU.sha(a.pkl), catalog=PU.sha(a.catalog),
                groupcat_partial={os.path.basename(p): PU.partial_fp(p) for p in gc},
                source_id_listing={os.path.basename(p): PU.stat_fp(p)
                                   for p in sorted(glob.glob(os.path.join(a.source_ids, "radio_FOF*.npz")))},
                radio_listing={os.path.basename(p): PU.stat_fp(p)
                               for p in sorted(glob.glob(os.path.join(a.radio_dir, "radio_FOF*.npz")))},
                cutouts_partial={os.path.basename(p): PU.partial_fp(p)
                                 for p in sorted(glob.glob(os.path.join(a.cutout_dir, "cutout_*.hdf5")))})


def prescription(a):
    return dict(model="synch", nu_ghz=a.nu_ghz, w_scale=a.w_scale, max_nh=a.max_nh, snapshot=99, z=0.0,
                spread_cells=False, img_size=a.img_size)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("begin", "end"), required=True)
    for k in ("dataset", "cells-dir", "radio-dir", "cutout-dir", "ratio-table"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--pkl", default="feats_labels_dict_tngcluster.pkl")
    ap.add_argument("--catalog", default="Radio_Data/TNG-Cluster_Catalog.hdf5")
    ap.add_argument("--groupcat", default=f"{C}/groupcat_classification/groupcat_099")
    ap.add_argument("--source-ids", default="Radio_Data", help="directory build_dataset lists radio_FOF*.npz from")
    ap.add_argument("--nu-ghz", type=float, required=True)
    ap.add_argument("--w-scale", type=float, required=True)
    ap.add_argument("--max-nh", type=float, required=True)
    ap.add_argument("--img-size", type=int, required=True)
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    begin_p, out = a.dataset + ".begin.json", a.dataset + ".manifest.json"
    git = lambda *c: subprocess.run(["git", *c], cwd=here, capture_output=True, text=True).stdout.strip()
    if a.stage == "begin":
        for p in (a.dataset, a.cells_dir, begin_p, out):
            if os.path.exists(p):
                raise SystemExit(f"refusing to start: {p} exists")
        rec = dict(prescription=prescription(a), sources=sources_record(a, here), head=git("rev-parse", "HEAD"),
                   uncommitted=git("status", "--porcelain", "--", *HELPERS), args=vars(a))
        with open(begin_p, "x") as f:
            json.dump(rec, f, indent=1)
        print(f"wrote {begin_p}")
        return 0
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    begin = json.load(open(begin_p))
    if begin["prescription"] != prescription(a):
        raise SystemExit("end-stage prescription differs from the begin record")
    now = sources_record(a, here)
    changed = sorted(k for k in now if now[k] != begin["sources"].get(k))
    if changed:
        raise SystemExit(f"sources changed during job B: {changed}; no manifest written")
    with h5py.File(a.dataset) as f:
        at = {k: (f.attrs[k].tolist() if hasattr(f.attrs[k], "tolist") else f.attrs[k]) for k in f.attrs}
        shape = f["images"].shape
        hid = [int(h) for h in f["meta/halo_id"][:]]
    at = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in at.items()}
    at["projections"] = [p.decode() if isinstance(p, bytes) else str(p) for p in at.get("projections", [])]
    n = EXPECT_ATTRS["n_clusters"]
    bad = {k: (at.get(k), v) for k, v in EXPECT_ATTRS.items() if at.get(k) != v}
    if bad or at.get("img_size") != a.img_size or at.get("cells_dir") != a.cells_dir \
            or tuple(shape) != (n, 3, a.img_size, a.img_size) or len(set(hid)) != len(hid):
        raise SystemExit(f"dataset does not match the declared protocol: {bad}, img {at.get('img_size')}, "
                         f"cells_dir {at.get('cells_dir')}, shape {shape}")
    cells = sorted(glob.glob(os.path.join(a.cells_dir, "cells_FOF*_sub*.npz")))
    fofs = sorted(int(re.search(r"cells_FOF(\d+)_", os.path.basename(c)).group(1)) for c in cells)
    if fofs != sorted(hid):
        raise SystemExit("cell files do not cover exactly the dataset's parents")
    man = dict(status="complete", kind="cells_1400", prescription=begin["prescription"],
               dataset=dict(path=a.dataset, sha256=PU.sha(a.dataset), attrs=at, shape=list(shape)),
               cells=PU.dir_digest(a.cells_dir), ratio_table=dict(path=a.ratio_table, sha256=now["ratio_table"]),
               sources=begin["sources"], sources_verified_unchanged=True, begin_head=begin["head"],
               begin_uncommitted=begin["uncommitted"], end_head=git("rev-parse", "HEAD"),
               note="raw-cutout and group-catalogue fingerprints are partial (size, mtime, first 16 MiB)")
    with open(out, "x") as f:
        json.dump(man, f, indent=1)
    print(f"wrote {out}: dataset sha256 {man['dataset']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
