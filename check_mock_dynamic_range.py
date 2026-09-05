"""Is the high-threshold sim/obs mismatch AGN contamination or dynamic range?

Compact-source masking removed 215 sources from the LoTSS cutouts and barely
moved the k>=8 sigma KS distance. That points away from the AGN explanation:
if the mocks have no pixels above 8 sigma at all, nothing removed from the
observed side can close the gap.

The Minkowski area at threshold k is non-zero exactly when a map's peak
exceeds k*rms, so the saved MF curves already encode a peak-SNR ECDF. This
prints it for mocks and observations side by side.
"""
import argparse

import numpy as np


def peak_frac(mfs, ksigma):
    """Fraction of maps with any pixel above each k*rms (area > 0)."""
    area = mfs[..., 0]
    return (area > 0).mean(axis=tuple(range(area.ndim - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="forward_lotss_masked_mfs.npz")
    ap.add_argument("--baseline-npz", default="forward_lotss_mfs.npz")
    args = ap.parse_args()

    for tag, path in (("masked", args.npz), ("unmasked", args.baseline_npz)):
        try:
            d = np.load(path, allow_pickle=True)
        except FileNotFoundError:
            print(f"{tag}: {path} not found, skipping\n")
            continue
        k = d["ksigma"]
        mock = d["mock_mfs"]
        obs = d["obs_mfs"]
        fm = peak_frac(mock, k)
        fo = peak_frac(obs, k)
        print(f"=== {tag} ({path}) ===")
        print(f"mock maps: {mock.shape[0]}x{mock.shape[1]} proj   "
              f"obs maps: {obs.shape[0]}")
        print(f"{'k':>5}{'mock frac >k':>15}{'obs frac >k':>14}")
        for i, kk in enumerate(k):
            print(f"{kk:>5.0f}{fm[i]:>15.3f}{fo[i]:>14.3f}")

        # peak SNR per map, interpolated as the largest k with area > 0
        def peak_k(a):
            a = a.reshape(-1, a.shape[-2], a.shape[-1])[..., 0]
            hits = a > 0
            idx = np.where(hits.any(1), hits.shape[1] - 1
                           - np.argmax(hits[:, ::-1], axis=1), -1)
            return np.where(idx >= 0, k[np.clip(idx, 0, None)], 0.0)

        pm, po = peak_k(mock), peak_k(obs)
        print(f"\nhighest k with signal (median): mock {np.median(pm):.1f}  "
              f"obs {np.median(po):.1f}")
        print(f"  mock 90th pct {np.percentile(pm, 90):.1f}   "
              f"obs 10th pct {np.percentile(po, 10):.1f}\n")


if __name__ == "__main__":
    main()
