"""Eyeball check: real Chandra model inputs next to mocks at similar redshift.

Everything that could silently go wrong in process_chandra.py -- a flipped
RA axis, a bad centre, a wrong exposure normalisation, the aperture in the
wrong place -- shows up here faster than in any statistic. Same stretch
constant and colour scale throughout, so brightness differences are real.
"""
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with h5py.File("xray_obs_acisi.h5") as f:
    obs = f["obs/images"][:]
    names = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/name"][:]]
    zs = f["obs/z"][:]
    texp = f["obs/exposure_ks"][:]
with h5py.File("xray_real_placed.h5") as f:
    mock = f["mock/images"][:]
    mz = f["mock/z"][:]
    tsc = f["mock/pseudo_tsc"][:]

rng = np.random.default_rng(0)
n = len(obs)
fig, ax = plt.subplots(n, 4, figsize=(9, 2.3 * n))
vmax = np.percentile(np.concatenate([obs.ravel(), mock[:50].ravel()]), 99.8)
for i in range(n):
    ax[i, 0].imshow(obs[i], origin="lower", cmap="magma", vmin=0, vmax=vmax)
    ax[i, 0].set_title(f"{names[i]} z={zs[i]:.3f}\n{texp[i]:.0f} ks (real)",
                       fontsize=7)
    idx = np.argwhere(np.abs(mz - zs[i]) < 0.005)
    for j in range(3):
        if len(idx):
            c, r = idx[rng.integers(len(idx))]
            ax[i, j + 1].imshow(mock[c, r, 0], origin="lower", cmap="magma",
                                vmin=0, vmax=vmax)
            ax[i, j + 1].set_title(f"mock, TSC {tsc[c]:.1f} Gyr", fontsize=7)
for a in ax.ravel():
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
fig.savefig("xray_real_vs_mock.png", dpi=110)

lin_o = np.sinh(obs.astype(np.float64))
lin_m = np.sinh(mock.astype(np.float64))
print("median stretched pixel inside the aperture: real "
      f"{np.median(obs[obs > 0]):.3f}, mock {np.median(mock[mock > 0]):.3f}")
print("wrote xray_real_vs_mock.png")
