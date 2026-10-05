#!/usr/bin/env python3
"""Concatenate single-snapshot training files into one multi-snapshot file.

Every row of every file is keyed by its z=0 halo id, so a cluster seen at two
snapshots would appear twice under one id. In the merged file:

  <grp>/halo_id   unique row id: the z=0 id for snapshot 99 (so snap-99 rows
                  keep their existing ids), snap * 1e9 + z=0 id otherwise
  <grp>/group_id  the z=0 id -- the trainers' fold key, so both epochs of a
                  cluster always fall in the same fold
  <grp>/snap      snapshot of each row

<grp> is "meta" for clean datasets (dataset_*.h5) and "mock" for injected /
X-ray training files. Per-row arrays (first dimension = number of rows) are
concatenated; everything else (tau grid, the real "obs" group) is taken from
the first file, and must agree across files where it matters (checked for
the X-ray stretch scale). Snapshot-99 rows come first, in their original
order, so the trainers' cluster-level folds for them match the
single-snapshot runs.
"""
import argparse

import h5py
import numpy as np


def uid(z0, snap):
    z0 = np.asarray(z0, dtype=np.int64)
    return z0 if snap == 99 else np.int64(snap) * 10 ** 9 + z0


def snap_of(f, grp):
    if f"{grp}/snap" in f:
        raise SystemExit("input is already a merged file")
    return int(f.attrs.get("snapshot", 99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    files = [h5py.File(p, "r") for p in args.inputs]
    grp = "meta" if "meta/halo_id" in files[0] else "mock"
    snaps = [snap_of(f, grp) for f in files]
    if snaps[0] != 99 and 99 in snaps:
        raise SystemExit("list the snapshot-99 file first")
    if "stretch_scale" in files[0].attrs:
        s = [float(f.attrs["stretch_scale"]) for f in files]
        if not np.allclose(s, s[0], rtol=1e-6):
            raise SystemExit(f"X-ray stretch scales differ: {s}")
    ns = [len(f[f"{grp}/halo_id"]) for f in files]
    print(f"{grp}: " + ", ".join(f"{p} (snap {s}, {n})"
                                 for p, s, n in zip(args.inputs, snaps, ns)))

    def rowwise(name, obj):
        return (isinstance(obj, h5py.Dataset) and obj.ndim > 0
                and obj.shape[0] == ns[0] and not name.startswith("obs/")
                and name != "labels/tau_gyr")

    with h5py.File(args.output, "w") as out:
        for k, v in files[0].attrs.items():
            out.attrs[k] = v
        out.attrs["snapshots"] = np.array(snaps)
        out.attrs["merged_from"] = np.array(args.inputs, dtype=h5py.string_dtype())

        names = []
        files[0].visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        for name in names:
            src = files[0][name]
            if rowwise(name, src):
                parts = []
                for f, n in zip(files, ns):
                    if name not in f or f[name].shape[0] != n:
                        raise SystemExit(f"{name} missing or wrong length in {f.filename}")
                    parts.append(f[name][:])
                data = np.concatenate(parts)
                if name == f"{grp}/halo_id":
                    out.create_dataset(f"{grp}/group_id", data=data)
                    data = np.concatenate([uid(p, s) for p, s in zip(parts, snaps)])
                kw = {"compression": "gzip", "compression_opts": 4} if data.ndim > 2 else {}
                out.create_dataset(name, data=data, **kw)
            else:
                out.create_dataset(name, data=src[:])
            for k, v in src.attrs.items():
                out[name].attrs[k] = v
        out.create_dataset(f"{grp}/snap", data=np.concatenate(
            [np.full(n, s, dtype=np.int16) for n, s in zip(ns, snaps)]))
        print(f"wrote {args.output}: {sum(ns)} rows, "
              f"{len(np.unique(out[f'{grp}/group_id'][:]))} distinct z=0 clusters")
    for f in files:
        f.close()


if __name__ == "__main__":
    main()
