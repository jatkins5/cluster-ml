#!/usr/bin/env python3
"""Attach the real Chandra inputs to an X-ray training set, on its stretch.

process_chandra.py stretched the real images with the constant of the
training set it was pointed at. Each rebuild of the mocks gets its own
constant (it is the median positive pixel of that set), so the real images
are undone with their own constant and re-stretched with the target set's.
Without this the real and mock images would sit on different scales and
the model would see a brightness shift that is purely bookkeeping.
"""
import argparse

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--obs-h5", default="xray_obs_acisi.h5")
ap.add_argument("--train-h5", required=True)
args = ap.parse_args()

with h5py.File(args.obs_h5, "r") as f:
    a_obs = float(f.attrs["stretch_scale"])
    img = f["obs/images"][:].astype(np.float64)
    meta = {k: f[f"obs/{k}"][:] for k in f["obs"] if k != "images"}
with h5py.File(args.train_h5, "a") as g:
    a_tr = float(g.attrs["stretch_scale"])
    out = np.arcsinh(np.sinh(img) * a_obs / a_tr).astype(np.float32)
    if "obs" in g:
        del g["obs"]
    o = g.create_group("obs")
    o.create_dataset("images", data=out)
    for k, v in meta.items():
        o.create_dataset(k, data=v)
    g.attrs["obs_source"] = args.obs_h5
print(f"attached {len(out)} real clusters to {args.train_h5}: "
      f"restretched {a_obs:.4g} -> {a_tr:.4g}")
