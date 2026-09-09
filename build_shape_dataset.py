"""B2 option 4: rescale every map to the same total flux, keeping only shape.

The pooled CNN reaches OOF R2 0.487 (0.528 as a 5-seed ensemble) while a
single scalar -- log total linear weight -- reaches 0.432, and the scalars
together explain 72% of the CNN's own output. B1 showed there is real
within-mass signal on top of that, but "how much of the claim is morphology
rather than brightness" is better answered by removing brightness outright
than by regressing it away.

Each projection is divided by its own total linear weight and multiplied
back up by the dataset median, so every map in the output carries exactly
the same total flux and differs only in how that flux is arranged. The
arcsinh stretch is then re-applied with the same parameter-free np.arcsinh
`build_dataset.py` uses, so the output is drop-in for the existing
trainers; labels and meta are copied verbatim.

Normalisation is per projection, not per cluster, because that is what an
observer with a single map would do.

WHY THIS IS NOT APPLIED TO THE MOCKS. The obvious companion experiment --
do the same to `injected_nh4.h5` and see whether the transfer model stops
reading mass -- does not work, and it is worth writing down why rather than
rediscovering it. Those maps are stored as arcsinh(map / rms), i.e. in
units of their own noise. Dividing a map by its own brightness scale
divides the *source* down to a common level but leaves the noise floor
where it was, so a bright cluster ends up looking like a clean map and a
faint one like a noisy map: SNR still encodes brightness, and brightness
still encodes mass. There is no way to remove source amplitude from a noisy
map without the noise level itself becoming the giveaway. The transfer-side
confound has to be fixed at the source (B2 options 1 or 3), not normalised
away.
"""
import argparse

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset_nh4_128.h5")
    ap.add_argument("--output", default="dataset_nh4_128_shape.h5")
    args = ap.parse_args()

    with h5py.File(args.input, "r") as f:
        images = f["images"][:]
        N, P, H, W = images.shape
        print(f"{args.input}: {images.shape}")

        lin = np.sinh(images.astype(np.float64))
        tot = lin.sum(axis=(2, 3))                      # (N, P)
        bad = ~np.isfinite(tot) | (tot <= 0)
        if bad.any():
            print(f"  {bad.sum()} maps with non-positive total flux; "
                  f"left untouched")
        scale = float(np.median(tot[~bad]))
        print(f"  total linear weight: median {scale:.4g}, "
              f"spread {np.log10(tot[~bad].max() / tot[~bad].min()):.2f} dex "
              f"-> all maps set to the median")

        out = images.copy()
        ok = ~bad
        idx = np.argwhere(ok)
        for i, p in idx:
            out[i, p] = np.arcsinh(lin[i, p] * (scale / tot[i, p]))

        chk = np.sinh(out.astype(np.float64)).sum(axis=(2, 3))[ok]
        print(f"  after: total flux spread "
              f"{np.log10(chk.max() / chk.min()):.2e} dex (should be ~0)")

        with h5py.File(args.output, "w") as g:
            g.create_dataset("images", data=out, compression="gzip",
                             compression_opts=4)
            for grp in ("meta", "labels"):
                if grp in f:
                    f.copy(grp, g)
            for k, v in f.attrs.items():
                g.attrs[k] = v
            g.attrs["shape_only"] = True
            g.attrs["shape_source"] = args.input
            g.attrs["shape_note"] = (
                "each projection rescaled to the dataset median total linear "
                "weight; morphology retained, brightness removed")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
