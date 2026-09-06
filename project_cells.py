"""Project shock cells onto a grid, spreading each over its own volume.

build_dataset.py's project_image histograms the weights, which deposits a
30 kpc-diameter gas cell into a single 18 kpc pixel. With one cell carrying a
median 30% of the total linear weight, that is what makes the linear map a
delta function and what forced the forward model to lean on the arcsinh
compression as an accidental stand-in for the missing smoothing.

Each Voronoi cell is treated as a sphere of equivalent volume, projected to a
2D Gaussian of the same half-mass scale. Rather than stamping every cell
individually (~50-250k per halo), cells are grouped into logarithmic radius
bins, each bin histogrammed and then convolved once with its representative
kernel. The cost is one FFT per bin instead of one stamp per cell, and the
error is bounded by the bin width.
"""
import numpy as np
from scipy import ndimage

# A uniform sphere of radius R projects to a profile whose second moment
# matches a Gaussian of sigma = R * sqrt(2/5) in each transverse axis.
SPHERE_TO_SIGMA = np.sqrt(0.4)


def project_cells(pos, w, r_kpc, center, half_width, img_size,
                  n_bins=12, max_sigma_px=64.0):
    """Return (3, img_size, img_size) linear weight maps for xy, yz, xz."""
    rel = pos - center
    edges = np.linspace(-half_width, half_width, img_size + 1)
    px_kpc = 2.0 * half_width / img_size
    sigma_px = np.clip(r_kpc * SPHERE_TO_SIGMA / px_kpc, 0.0, max_sigma_px)

    # Log-spaced bins over the resolved cells; everything below half a pixel
    # is unresolved and goes into a single unsmoothed bin.
    resolved = sigma_px > 0.5
    out = np.zeros((3, img_size, img_size), dtype=np.float64)
    groups = []
    if (~resolved).any():
        groups.append((~resolved, 0.0))
    if resolved.any():
        lo, hi = sigma_px[resolved].min(), sigma_px[resolved].max()
        bins = np.geomspace(lo, hi * (1 + 1e-9), n_bins + 1)
        idx = np.digitize(sigma_px, bins) - 1
        for b in range(n_bins):
            m = resolved & (idx == b)
            if m.any():
                # Weight-weighted sigma, so a bin's bright cells set its scale.
                sw = w[m]
                s = (np.average(sigma_px[m], weights=sw)
                     if sw.sum() > 0 else np.median(sigma_px[m]))
                groups.append((m, float(s)))

    for k, (ax0, ax1) in enumerate([(0, 1), (1, 2), (0, 2)]):
        for m, s in groups:
            img, _, _ = np.histogram2d(rel[m, ax0], rel[m, ax1],
                                       bins=edges, weights=w[m])
            if s > 0:
                # 'constant' so flux leaving the field of view is not folded
                # back in, matching the crop the histogram already applies.
                img = ndimage.gaussian_filter(img, s, mode="constant")
            out[k] += img
    return out
