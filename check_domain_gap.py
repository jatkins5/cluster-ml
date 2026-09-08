"""What does the CNN see that separates mocks from real cutouts?

The transfer failure is extreme extrapolation, which means real images sit far
outside the mock distribution in feature space. Before guessing at
augmentations, measure which simple image statistics actually differ -- in the
exact input space the CNN gets, arcsinh(map / rms).

Any statistic that separates the two populations cleanly is a channel the CNN
can key on, and a candidate for either augmentation or removal.
"""
import argparse

import h5py
import numpy as np
from scipy import ndimage, stats


def radial_power(img):
    """Azimuthally averaged power in three spatial bands (large/mid/small)."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    n = img.shape[0]
    y, x = np.mgrid[:n, :n] - n / 2
    r = np.hypot(y, x)
    tot = f.sum()
    if tot <= 0:
        return np.array([np.nan] * 3)
    return np.array([f[r < 8].sum(), f[(r >= 8) & (r < 24)].sum(),
                     f[r >= 24].sum()]) / tot


def stats_for(img):
    v = img.ravel()
    # A masked region is a disk of pure noise: locally flat, no structure.
    # Count pixels whose 5x5 neighbourhood has near-zero gradient but nonzero
    # noise -- a crude but direct probe of mask-disk artefacts.
    g = ndimage.sobel(img, 0) ** 2 + ndimage.sobel(img, 1) ** 2
    return np.array([
        v.mean(), v.std(), np.percentile(v, 99), v.max(),
        (v > 1.0).mean(),            # fraction above ~1 sigma (arcsinh units)
        (v > 3.0).mean(),
        g.mean(),                    # mean squared gradient: overall roughness
        *radial_power(img),
    ])


NAMES = ["mean", "std", "p99", "max", "frac>1", "frac>3", "roughness",
         "power large-scale", "power mid-scale", "power small-scale"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mock_dataset_nh4.h5")
    args = ap.parse_args()

    with h5py.File(args.dataset, "r") as f:
        mock = f["mock/images"][:]
        obs = f["obs/images"][:]
    m = mock.reshape(-1, mock.shape[-2], mock.shape[-1])
    print(f"mock {m.shape}, obs {obs.shape}\n")

    ms = np.array([stats_for(x) for x in m[::7]])   # subsample for speed
    os_ = np.array([stats_for(x) for x in obs])

    print(f"{'statistic':<22}{'mock':>12}{'obs':>12}{'ratio':>9}"
          f"{'sep':>8}{'KS p':>10}")
    for j, n in enumerate(NAMES):
        a, b = ms[:, j], os_[:, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        # Separation in units of the mock spread: how far outside the training
        # distribution the median real image sits.
        sep = (np.median(b) - np.median(a)) / (a.std() + 1e-12)
        p = stats.ks_2samp(a, b).pvalue
        print(f"{n:<22}{np.median(a):>12.4g}{np.median(b):>12.4g}"
              f"{np.median(b)/(np.median(a)+1e-12):>9.2f}{sep:>8.1f}{p:>10.2g}")

    print("\n'sep' is (obs median - mock median) / mock sd. Anything beyond")
    print("about +-3 is a channel the CNN can separate the domains on.")


if __name__ == "__main__":
    main()
