"""What bright emission survives compact-source masking?

After the brightness-independent mask, LoTSS cutouts still hold a median of
3 components above 13 sigma while the mocks have none. Either the mask is
still failing, or the survivors are bright *resolved* radio galaxies (BCG
jets, head-tail sources) that no compactness cut can remove -- and that the
forward model does not simulate at all, since it only paints diffuse
shock-driven emission.

Elongation and offset from the cluster centre separate the two cases: a
mask failure leaves round residuals at the position of masked sources,
whereas radio galaxies are elongated and sit near the centre.
"""
import argparse
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward_model_lotss import (BEAM_FWHM_ARCSEC, GRID, kpc_per_arcsec,
                                 load_obs, mask_compact)
from minkowski_functionals import STRUCT8


def shape_stats(comp):
    ys, xs = np.nonzero(comp)
    if len(ys) < 3:
        return 1.0
    c = np.cov(np.vstack([ys - ys.mean(), xs - xs.mean()]))
    ev = np.sort(np.linalg.eigvalsh(c))
    return float(np.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotss-glob", default=os.path.expanduser(
        "~/data/cluster-ml/lotss_images/*.fits"))
    ap.add_argument("--targets-csv",
                    default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--box-kpc", type=float, default=1000.0)
    ap.add_argument("--ksig", type=float, default=13.0)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    obs = load_obs(args.lotss_glob, args.targets_csv, args.box_kpc,
                   mask_rng=rng)
    px_kpc = args.box_kpc / GRID
    ctr = GRID / 2.0

    print(f"\n{'target':<12}{'n>k':>5}{'halfmax/beam':>14}{'elong':>8}"
          f"{'offset_kpc':>12}{'peak/rms':>10}")
    rows = []
    for o in obs:
        img, rms = o["map"], o["rms"]
        fwhm_px = BEAM_FWHM_ARCSEC * kpc_per_arcsec(o["z"]) / px_kpc
        hp_area = 0.7854 * fwhm_px ** 2
        lab, n = ndimage.label(img > args.ksig * rms, structure=STRUCT8)
        if n == 0:
            print(f"{o['name']:<12}{0:>5}")
            continue
        for sl, i in zip(ndimage.find_objects(lab), range(1, n + 1)):
            comp = lab[sl] == i
            vals = img[sl][comp]
            nhalf = (vals > 0.5 * vals.max()).sum()
            cy, cx = ndimage.center_of_mass(comp)
            cy += sl[0].start
            cx += sl[1].start
            off = np.hypot(cy - ctr, cx - ctr) * px_kpc
            rows.append((nhalf / hp_area, shape_stats(comp), off,
                         vals.max() / rms))
            print(f"{o['name']:<12}{n:>5}{nhalf / hp_area:>14.2f}"
                  f"{shape_stats(comp):>8.2f}{off:>12.0f}"
                  f"{vals.max() / rms:>10.1f}")

    if rows:
        a = np.array(rows)
        print(f"\nsurvivors above {args.ksig:.0f} sigma: n={len(a)}")
        print(f"  half-max area / beam : median {np.median(a[:, 0]):.2f}  "
              f"(mask cut was 2.0)")
        print(f"  elongation           : median {np.median(a[:, 1]):.2f}")
        print(f"  offset from centre   : median {np.median(a[:, 2]):.0f} kpc")
        print(f"  peak / rms           : median {np.median(a[:, 3]):.0f}")
        print(f"  within 150 kpc of centre: {(a[:, 2] < 150).mean():.0%}")


if __name__ == "__main__":
    main()
