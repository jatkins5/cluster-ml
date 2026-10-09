"""What fraction of each cluster's 144 MHz shock emission (n_H < 1e-4 cut)
falls inside a centred square box of a given physical size? The realistic
radio mocks are 1 Mpc boxes (+-500 kpc)."""
import h5py
import numpy as np

with h5py.File("dataset_nh4_144_512.h5") as f:
    img = np.sinh(f["images"][:].astype(np.float64))     # (N, 3, 512, 512)
    r500 = f["meta/r500c_kpc"][:]
N, P, H, _ = img.shape
tot = img.sum(axis=(2, 3))
print(f"r500 median {np.median(r500):.0f} kpc; image half-width 4 r500")
for half in (500, 1000, 1500, 2000, 3000):
    fr = []
    for i in range(N):
        px = 8 * r500[i] / H
        h = int(round(half / px))
        h = min(h, H // 2)
        c = img[i, :, H // 2 - h:H // 2 + h, H // 2 - h:H // 2 + h].sum(axis=(1, 2))
        fr.append(np.mean(c / tot[i]))
    fr = np.array(fr)
    print(f"  box +-{half:>4} kpc ({2 * half / 1000:.0f} Mpc): emission inside "
          f"median {np.median(fr):.3f}, 16-84% {np.percentile(fr, 16):.3f}-"
          f"{np.percentile(fr, 84):.3f}")
for k in (0.5, 1, 2):
    h = int(round(k / 8 * H))
    fr = (img[:, :, H // 2 - h:H // 2 + h, H // 2 - h:H // 2 + h].sum(axis=(2, 3)) / tot).mean(1)
    print(f"  box +-{k} r500: median {np.median(fr):.3f}")
