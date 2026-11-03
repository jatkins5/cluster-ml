#!/usr/bin/env python3
"""Check the 3 Mpc follow-up cohort and persist the running protocol's splits.

Run from the frozen follow-up code directory so group_folds is the same
implementation as the control training. No images are loaded here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from train_cnn_mock import group_folds

RADIO = "injected_syn_phys_b3000_s9991.h5"
XRAY = "xray_fix_b3000_s9991.h5"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()
    out = args.run_dir / "preflight"
    out.mkdir(exist_ok=False)
    with h5py.File(RADIO) as r, h5py.File(XRAY) as x:
        keys = ["group_id", "halo_id", "snap", "pseudo_tsc", "log_m500", "z"]
        for key in keys:
            if not np.array_equal(r[f"mock/{key}"][:], x[f"mock/{key}"][:]):
                raise ValueError(f"Unmatched mock/{key}; do not train the control")
        shape = r["mock/images"].shape
        if shape != x["mock/images"].shape:
            raise ValueError("Image/cohort shapes differ")
        groups = np.repeat(r["mock/group_id"][:], shape[1])
    outer = np.full(len(groups), -1, dtype=int)
    splits = list(group_folds(groups, 5))
    for k, (_, heldout) in enumerate(splits):
        outer[heldout] = k
    for seed in range(42, 47):
        sel_groups, sel_folds = [], []
        for k, (train, heldout) in enumerate(splits):
            split = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed+k+1)
            fit_idx, sel_idx = next(split.split(train, groups=groups[train]))
            fit, sel = groups[train[fit_idx]], groups[train[sel_idx]]
            assert not (set(fit) & set(sel) or set(fit) & set(groups[heldout])
                        or set(sel) & set(groups[heldout]))
            sg = np.unique(sel)
            sel_groups.append(sg)
            sel_folds.append(np.full(len(sg), k))
        np.savez(out / f"splits_s{seed}.npz", sample_group=groups, fold_of=outer,
                 sel_groups_flat=np.concatenate(sel_groups),
                 sel_groups_fold=np.concatenate(sel_folds))
    record = dict(radio=RADIO, xray=XRAY, shape=shape,
                  matched_metadata=keys, parents=len(np.unique(groups)),
                  seeds=list(range(42, 47)),
                  protocol="unchanged default pooled training/selection, 5 folds, 60 epochs",
                  sha256={p: sha256(p) for p in [RADIO, XRAY]})
    (out / "verified.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
