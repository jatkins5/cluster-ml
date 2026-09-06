"""Sanity-check the recentred dataset against the legacy one.

Three things must hold before the baselines are worth running:
  1. --center weight reproduces the existing dataset.h5 exactly (the refactor
     did not change legacy behaviour).
  2. --center grouppos actually produces different images.
  3. Recentring does not silently throw emission out of the field of view.
"""
import h5py
import numpy as np

def load(p):
    with h5py.File(p, "r") as f:
        return (f["images"][:], f["meta/halo_id"][:],
                f["labels/pseudo_tsc"][:],
                f.attrs.get("center_mode", "(legacy, unlabelled)"))

old, hid_o, tsc_o, mode_o = load("dataset.h5")
wc,  hid_w, tsc_w, mode_w = load("dataset_wc_128.h5")
gp,  hid_g, tsc_g, mode_g = load("dataset_gp_128.h5")
print(f"center_mode attrs: dataset.h5={mode_o!r}  wc={mode_w!r}  gp={mode_g!r}")

assert np.array_equal(hid_o, hid_w) and np.array_equal(hid_o, hid_g)
assert np.array_equal(tsc_o, tsc_w) and np.array_equal(tsc_o, tsc_g)
print("halo_id and pseudo_tsc identical across all three")

d = np.abs(old - wc).max()
print(f"\n1. legacy reproduction: max |dataset.h5 - wc| = {d:.3e} "
      f"({'OK' if d == 0 else 'MISMATCH'})")

frac_diff = (np.abs(gp - wc) > 1e-6).mean()
print(f"2. recentring changed {frac_diff:.1%} of pixels")

# arcsinh(sum w) per pixel; sum of sinh recovers the linear in-frame weight.
def captured(imgs):
    return np.sinh(imgs.astype(np.float64)).sum(axis=(1, 2, 3))
cw, cg = captured(wc), captured(gp)
r = cg / np.where(cw > 0, cw, np.nan)
print(f"3. in-frame linear weight, grouppos / weight-centred:")
print(f"   median {np.nanmedian(r):.3f}  "
      f"10th {np.nanpercentile(r, 10):.3f}  90th {np.nanpercentile(r, 90):.3f}")
print(f"   clusters losing >10% of in-frame weight: {np.nansum(r < 0.9)}")
print(f"   clusters gaining >10%:                   {np.nansum(r > 1.1)}")
