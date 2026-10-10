"""Review F1: realism of each injected radio mock set against the real LoTSS
targets, with the metric fixed BEFORE the new sets were built (2026-10-09):

  primary: median over {area fraction, perimeter, components, Euler
           characteristic} x 10 thresholds (1-20 sigma, log-spaced) of
           |median(real) - median(mock)| / sd(mock)
  secondary: the same per functional; and max |separation|.

A background-only control (fields with nothing injected, same pipeline) is
scored identically: a mock set that is no closer to the real clusters than
empty sky is not adding realistic emission. Lower is closer.
Usage: check_radio_realism.py label=file.h5 [label=file.h5 ...]
"""
import sys

import h5py
import numpy as np

from minkowski_functionals import binary_mfs

KS = np.geomspace(1.0, 20.0, 10)
NAMES = ["area", "perimeter", "components", "Euler"]
COLS = [0, 1, 2, 4]


def mfs(imgs):
    thr = np.arcsinh(KS)                      # maps are arcsinh(S / rms)
    return np.array([[binary_mfs(im > t) for t in thr] for im in imgs])[:, :, COLS]


rows = []
for arg in sys.argv[1:]:
    lab, path = arg.split("=", 1)
    with h5py.File(path) as f:
        mock = f["mock/images"][:]
        # each set against the real cutouts made at ITS box size and from its
        # own field set (an earlier version reused the first file's)
        obs_mf = mfs(f["obs/images"][:])
        n_obs = len(f["obs/images"])
        attrs = dict(f.attrs)
        attrs["n_obs"] = n_obs
    m = mfs(mock.reshape(-1, *mock.shape[-2:]))
    sep = (np.median(obs_mf, 0) - np.median(m, 0)) / (m.std(0) + 1e-9)   # (k, 4)
    a = np.abs(sep)
    rows.append((lab, np.median(a), a.max(), np.median(a, axis=0), attrs))
print(f"thresholds {np.round(KS, 1)} sigma; real targets per set in brackets")
print(f"{'set':<28}{'median|sep|':>12}{'max|sep|':>10}  " + "".join(f"{n:>11}" for n in NAMES))
for lab, med, mx, per, at in rows:
    print(f"{lab:<28}{med:>12.3f}{mx:>10.2f}  " + "".join(f"{v:>11.3f}" for v in per)
          + f"   [recipe={at.get('recipe', 'legacy')}, box={at.get('box_kpc', 1000)}, "
            f"control={at.get('control', False)}, n_obs={at['n_obs']}]")
