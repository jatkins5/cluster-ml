"""What are the >10 sigma pixels in the AGN-masked LoTSS cutouts?

The last remaining domain gap is brightness: mocks never exceed ~10 sigma
(arcsinh(map/rms) > 3) while real cutouts always have some such pixels. Whether
that is fixable, and how, depends entirely on what those pixels are.

  compact, off-centre, elongated  -> resolved radio galaxies the forward model
                                     does not attempt to paint. Fixable only by
                                     injecting a source population.
  extended, central, round        -> genuine diffuse emission the mocks are
                                     failing to reproduce. A forward-model or
                                     emission-model problem.
"""
import argparse

import h5py
import numpy as np
from scipy import ndimage

STRUCT8 = np.ones((3, 3), dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mockcn_nh4.h5")
    ap.add_argument("--thresh", type=float, default=3.0,
                    help="threshold in arcsinh(map/rms); 3.0 ~ 10 sigma")
    args = ap.parse_args()

    with h5py.File(args.dataset, "r") as f:
        obs = f["obs/images"][:]
        name = [s.decode() if isinstance(s, bytes) else str(s)
                for s in f["obs/name"][:]]
        mock = f["mock/images"][:]

    m = mock.reshape(-1, mock.shape[-2], mock.shape[-1])
    print(f"mock maps above threshold: "
          f"{(m.max(axis=(1, 2)) > args.thresh).mean():.1%} of {len(m)}")
    print(f"obs  maps above threshold: "
          f"{(obs.max(axis=(1, 2)) > args.thresh).mean():.1%} of {len(obs)}\n")

    n = obs.shape[-1]
    cy0 = cx0 = n / 2.0
    px_kpc = 1000.0 / n
    rows = []
    print(f"{'target':<14}{'blobs':>7}{'area px':>9}{'axis ratio':>12}"
          f"{'r [kpc]':>10}{'peak':>8}")
    for img, nm in zip(obs, name):
        lab, k = ndimage.label(img > args.thresh, structure=STRUCT8)
        if k == 0:
            print(f"{nm:<14}{0:>7}")
            continue
        for i in range(1, k + 1):
            ys, xs = np.where(lab == i)
            area = len(ys)
            cy, cx = ys.mean(), xs.mean()
            r = np.hypot(cy - cy0, cx - cx0) * px_kpc
            if area >= 3:
                cov = np.cov(np.vstack([ys - cy, xs - cx]))
                ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
                ar = np.sqrt(ev[0] / max(ev[1], 1e-9))
            else:
                ar = np.nan
            rows.append((area, ar, r, img[ys, xs].max()))
        a = np.array([x for x in rows[-k:]], dtype=float)
        print(f"{nm:<14}{k:>7}{np.median(a[:, 0]):>9.1f}"
              f"{np.nanmedian(a[:, 1]):>12.2f}{np.median(a[:, 2]):>10.0f}"
              f"{np.max(a[:, 3]):>8.2f}")

    a = np.array(rows, dtype=float)
    print(f"\n{len(a)} blobs above {args.thresh} across {len(obs)} cutouts")
    for j, lab in enumerate(["area [px]", "axis ratio", "r from centre [kpc]",
                             "peak (arcsinh units)"]):
        v = a[:, j]
        v = v[np.isfinite(v)]
        print(f"  {lab:<24} median {np.median(v):>8.2f}  "
              f"[{np.percentile(v, 10):.2f}, {np.percentile(v, 90):.2f}]")
    print(f"\n  blobs within 150 kpc of centre: {(a[:, 2] < 150).mean():.0%}")
    print(f"  blobs beyond 300 kpc:           {(a[:, 2] > 300).mean():.0%}")
    beam_px = (9.0 / (1000.0 / 128 / 0.6)) ** 2 * 0.785  # rough, for scale
    print(f"  (a beam is very roughly {beam_px:.0f} px at these redshifts)")


if __name__ == "__main__":
    main()
