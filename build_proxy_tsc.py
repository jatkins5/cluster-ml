"""Write the best tree-free proxy TSC as a training label.

The point is to measure, on TNG where both exist, how much real signal
survives when the network is trained on a label a tree-free simulation
(CAMELS) could actually produce. The definition is the winner of the
grid search in validate_merger_proxy.py: a companion above mass ratio
0.02 inside 1.0 R200, last such snapshot sets the clock.

Two variants: `tsc_proxy` fills censored halos at the catalog span (what
CAMELS would be forced to do), `tsc_proxy_unc` drops them.
"""
import h5py
import numpy as np

VALID = "merger_proxy_validation.npz"
OUT = "tsc_proxy_snap99.hdf5"
THR = 0.02
RAD = 1.0


def main():
    d = np.load(VALID)
    halo_id, tsc_truth = d["halo_id"], d["tsc_truth"]
    t_of_snap, t0, span = d["t_of_snap"], float(d["t0"]), float(d["span"])

    thresholds = list(d["thresholds"])
    t = thresholds.index(THR)
    hit = d["min_sep"][:, :, t] < RAD                        # (n, nsnap)
    n = len(halo_id)
    proxy = np.full(n, np.nan)
    for i in range(n):
        w = np.where(hit[i])[0]
        if len(w):
            proxy[i] = t0 - t_of_snap[w.max()]

    uncens = np.isfinite(proxy)
    filled = np.where(uncens, proxy, span)
    print(f"proxy: mass ratio > {THR} within {RAD} R200")
    print(f"uncensored {uncens.sum()}/{n}; censored filled at {span:.2f} Gyr")
    print(f"proxy med={np.median(filled):.2f}  truth med="
          f"{np.nanmedian(tsc_truth):.2f} Gyr")

    with h5py.File(OUT, "w") as f:
        f.create_dataset("halo_id", data=halo_id)
        f.create_dataset("tsc_proxy", data=filled)
        f.create_dataset("tsc_proxy_unc", data=np.where(uncens, proxy, np.nan))
        f.create_dataset("tsc_truth", data=tsc_truth)
        f.attrs["thr"] = THR
        f.attrs["rad_r200"] = RAD
        f.attrs["span_gyr"] = span
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
