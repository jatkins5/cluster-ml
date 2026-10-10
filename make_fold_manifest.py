#!/usr/bin/env python3
"""Persisted parent-halo fold manifest (review R4-B).

Folds are assigned per z=0 parent halo, from the halo id alone: a fixed
random permutation of the sorted ids, dealt round-robin into K folds. The
assignment therefore does not depend on how many snapshots, realizations or
modalities a dataset holds for a halo, on row order, or on the hardware --
so a snapshot-99-only arm and a 99+91 arm restricted to the same manifest
hold out exactly the same parent halos. For each outer fold the inner
checkpoint-selection halos are fixed too (a fraction of that fold's training
halos, again chosen from the ids alone), so selection sets also match.

  python make_fold_manifest.py --datasets a.h5 b.h5 ... --output folds.npz
The manifest covers the union of the datasets' parent halos; trainers
restrict it to their own rows and refuse halos it does not contain.
"""
import argparse

import h5py
import numpy as np


def parent_ids(path):
    with h5py.File(path, "r") as f:
        g = "mock" if "mock" in f else "meta"
        key = f"{g}/group_id" if f"{g}/group_id" in f else f"{g}/halo_id"
        return np.unique(f[key][:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    ids = np.unique(np.concatenate([parent_ids(p) for p in a.datasets]))
    rng = np.random.default_rng(a.seed)
    perm = ids[rng.permutation(len(ids))]
    fold = np.empty(len(ids), dtype=np.int16)
    pos = {h: i for i, h in enumerate(ids)}
    for j, h in enumerate(perm):
        fold[pos[h]] = j % a.folds
    # inner selection halos for each outer fold k: a fixed fraction of the
    # halos NOT in fold k
    inner = np.zeros((a.folds, len(ids)), dtype=bool)
    for k in range(a.folds):
        tr = ids[fold != k]
        n = int(round(a.inner_frac * len(tr)))
        sel = np.random.default_rng(a.seed + 1 + k).choice(np.sort(tr), n, replace=False)
        inner[k] = np.isin(ids, sel)
    np.savez(a.output, group_id=ids, fold=fold, inner_sel=inner,
             seed=a.seed, folds=a.folds, inner_frac=a.inner_frac,
             sources=np.array(a.datasets))
    print(f"wrote {a.output}: {len(ids)} parent halos, fold sizes "
          f"{np.bincount(fold).tolist()}, inner-selection sizes {inner.sum(1).tolist()}")


if __name__ == "__main__":
    main()
