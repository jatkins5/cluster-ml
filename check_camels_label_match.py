"""Compare label distributions on the zooms that actually reach training.

The earlier KS test used all 768 zooms, but dataset_camels_128.h5 holds only
the 341 cluster-scale ones, so the number quoted for the proxy label was
computed over a sample that includes 427 zooms the model never sees.
"""
import h5py
import numpy as np
from scipy import stats

TNG = "TSC_Cutimages/TSC_eachhalo_snap99.hdf5"
CAM = "dataset_camels_128.h5"
PROXY = "camels_proxy_tsc.hdf5"


def summarize(name, v):
    v = v[np.isfinite(v)]
    print(f"{name:<28}{len(v):>6}{np.median(v):>9.2f}"
          f"{np.percentile(v, 25):>8.2f}{np.percentile(v, 75):>8.2f}"
          f"{v.max():>8.2f}{100 * (v <= 2).mean():>9.1f}%")
    return v


def main():
    with h5py.File(TNG, "r") as f:
        tng = f["tsc_gyr"][:]
    with h5py.File(CAM, "r") as f:
        zoom = f["meta/zoom_id"][:]
        massjump = f["labels/merger_tsc"][:]
        m200 = f["meta/m200_msun"][:]
    with h5py.File(PROXY, "r") as f:
        sub = dict(zip(f["zoom_id"][:], f["tsc_proxy"][:]))
    proxy_all = np.array([sub.get(int(z), np.nan) for z in sub])
    proxy_train = np.array([sub.get(int(z), np.nan) for z in zoom])

    print(f"dataset zooms: {len(zoom)}  "
          f"log10 M200 [{np.log10(m200).min():.2f}, "
          f"{np.log10(m200).max():.2f}]")
    print(f"\n{'label':<28}{'n':>6}{'median':>9}{'IQR25':>8}{'IQR75':>8}"
          f"{'max':>8}{'<=2 Gyr':>10}")
    t = summarize("TNG (target)", tng)
    summarize("proxy, all 768 zooms", proxy_all)
    p = summarize("proxy, 341 in dataset", proxy_train)
    mj = summarize("mass-jump, 341 in dataset", massjump)
    pc = summarize("proxy, clipped at 7.73", np.minimum(proxy_train, 7.73))

    print(f"\nKS distance against the TNG label (lower = better match):")
    for name, v in (("proxy, all 768", proxy_all),
                    ("proxy, 341 in dataset", p),
                    ("proxy clipped at 7.73", pc),
                    ("mass-jump, 341", mj)):
        v = v[np.isfinite(v)]
        print(f"  {name:<26} D={stats.ks_2samp(t, v).statistic:.3f}")


if __name__ == "__main__":
    main()
