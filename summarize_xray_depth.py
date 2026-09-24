"""X-ray accuracy as a function of realistic Chandra depth.

Each depth is the same 352 clusters thinned from the 2 Ms mocks with sky
background added (build_xray_realistic.py), trained with the transfer-model
protocol (train_cnn_mock.py: grouped folds, inner-split selection, three
noise realizations scored as the cluster mean). The 2000 ks row carries the
same added background and is the in-protocol reference; the clean-image
number from train_cnn_pooled.py (0.511) uses a different protocol and is
quoted only for orientation.
"""
import glob
import re

import h5py
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

ORDER = ["1", "3", "10", "30", "100", "archive", "2000"]
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))

print(f"{'depth':<10}{'exposure':>10}{'counts/map':>12}{'n':>4}{'mean R2':>10}"
      f"{'sd':>7}{'per seed':>24}{'rho(p,M)':>10}{'terciles':>22}")
for d in ORDER:
    runs = sorted(glob.glob(f"cnn_preds/xreal_{d}_s*_preds.npz"))
    if not runs:
        continue
    with h5py.File(f"xray_real_{d}.h5") as f:
        t = f["mock/exposure_ks"][:]
        img = f["mock/images"][:]
        a = f.attrs["stretch_scale"]
    counts = (np.sinh(img.astype(np.float64)) * a
              * t[:, :, None, None, None]).sum(axis=(3, 4))
    r2s, ens = [], None
    for p in runs:
        z = np.load(p, allow_pickle=True)
        r2s.append(r2_score(z["tsc"], z["oof_cluster"]))
        ens = z["oof_cluster"] if ens is None else ens + z["oof_cluster"]
    ens /= len(runs)
    tsc, hid = z["tsc"], z["halo_id"]
    logm = np.array([mm[h] for h in hid])
    q = np.quantile(logm, [1 / 3, 2 / 3])
    terc = [r2_score(tsc[b], ens[b]) for b in
            (logm <= q[0], (logm > q[0]) & (logm <= q[1]), logm > q[1])]
    print(f"{d:<10}{np.median(t):>8.0f}ks{np.median(counts):>12.3g}"
          f"{len(runs):>4}{np.mean(r2s):>10.3f}"
          f"{np.std(r2s, ddof=1) if len(r2s) > 1 else np.nan:>7.3f}"
          f"{'  ' + ', '.join(f'{v:.3f}' for v in r2s):>24}"
          f"{stats.spearmanr(ens, logm)[0]:>10.3f}"
          f"{'  ' + ' / '.join(f'{v:.2f}' for v in terc):>22}")

print("\nreference: clean 2 Ms X-ray, no sky background, train_cnn_pooled.py "
      "protocol: 0.511 +- 0.026 (not directly comparable)")
