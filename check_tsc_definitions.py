"""Compare the definitions in Chuiyang's snapshot-99 and snapshot-91
merger-time files: zeros (collision at the snapshot itself), missing values,
and the smallest nonzero times."""
import h5py
import numpy as np
from astropy.cosmology import Planck15

C = "/oscar/data/idellant/Chuiyang/TSC_Cutimages"
for s in (99, 91):
    with h5py.File(f"{C}/TSC_eachhalo_snap{s}.hdf5") as f:
        t = f["tsc_gyr"][:]
        print(f"snap {s}: {dict(f.attrs)}")
    ok = t[np.isfinite(t)]
    print(f"  n={len(t)}  NaN {np.isnan(t).sum()}  exactly 0: {(t == 0).sum()}  "
          f"smallest nonzero {np.round(np.sort(ok[ok > 0])[:4], 3)}")
print(f"time from snap 91 to 99: {Planck15.lookback_time(0.0994018).value:.2f} Gyr")
