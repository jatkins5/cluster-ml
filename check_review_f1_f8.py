"""Read-only diagnostics for the Codex review (CODEX_CLAUDE_REVIEW.md).

F1: the injected mocks' power estimator, sim_logp(crop_resample(arcsinh map)),
    versus the linear emission summed on the native pixels in the same box.
    - is the clip at arcsinh = 40 active, and how much power does it drop?
    - does zoom-upsampling add a pixel-count factor that scales with r500
      (box pixels ~ 1/r500, so area factor ~ r500^2 ~ M^(2/3))?
    - how far does the recipe move the power-mass slope?
F8: permutation p-values for the real radio-vs-X-ray partial correlations,
    and halo-level bootstrap/permutation for the relic Spearman values.
"""
import glob

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from forward_model_lotss import crop_resample

rng = np.random.default_rng(0)

print("======== F1")
with h5py.File("dataset_nh4_144_512.h5") as f:
    img = f["images"][:].astype(np.float64)          # arcsinh(weight)
    r500 = f["meta/r500c_kpc"][:]
    hid = f["meta/halo_id"][:]
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
lm = np.array([mm[h] for h in hid])
N, P, H, _ = img.shape
print(f"  map maxima (arcsinh units): median {np.median(img.max(axis=(1, 2, 3))):.1f}, "
      f"max {img.max():.1f}; projections with any pixel > 40: "
      f"{(img.max(axis=(2, 3)) > 40).mean():.1%}")
for box in (1000.0, 3000.0):
    rec, lin, clipfrac = [], [], []
    for i in range(N):
        px = 8.0 * r500[i] / H
        hp = box / 2.0 / px
        r_, l_, c_ = [], [], []
        for p in range(P):
            cut = crop_resample(img[i, p], hp)
            h = int(round(hp))
            nat = img[i, p, H // 2 - h:H // 2 + h, H // 2 - h:H // 2 + h]
            w = np.sinh(nat)
            r_.append(np.sinh(np.clip(cut, 0, 40)).sum())
            l_.append(w.sum())
            c_.append(np.sinh(np.clip(nat, 0, 40)).sum() / w.sum())
        rec.append(np.mean(r_)), lin.append(np.mean(l_)), clipfrac.append(np.mean(c_))
    rec, lin, clipfrac = map(np.array, (rec, lin, clipfrac))
    d = np.log10(rec) - np.log10(lin)
    fr = stats.linregress(np.log10(r500), d)
    fm_rec = stats.linregress(lm, np.log10(rec))
    fm_lin = stats.linregress(lm, np.log10(lin))
    print(f"  box {box / 1000:.0f} Mpc: power kept by the clip alone (native "
          f"pixels): median {np.median(clipfrac):.3f}, 5th pct "
          f"{np.percentile(clipfrac, 5):.2e}, clusters losing >10%: "
          f"{(clipfrac < 0.9).mean():.0%}")
    print(f"    log(recipe) - log(linear native sum): median {np.median(d):+.2f}, "
          f"sd {d.std():.2f} dex; slope vs log r500 {fr.slope:+.2f} +- {fr.stderr:.2f}"
          f" (pure area factor would be +2)")
    print(f"    P-M slope: recipe {fm_rec.slope:.2f} (scatter "
          f"{np.std(np.log10(rec) - fm_rec.intercept - fm_rec.slope * lm):.2f} dex)  vs "
          f"linear {fm_lin.slope:.2f} ({np.std(np.log10(lin) - fm_lin.intercept - fm_lin.slope * lm):.2f} dex)")

print("\n======== F8a: real radio vs X-ray, partial correlations")
wl = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")
key = lambda s: "".join(str(s).split()).replace("_", "").upper()


def ens(pat):
    ds = [np.load(p) for p in sorted(glob.glob(pat))]
    names = [key(n) for n in ds[0]["obs_name"]]
    return (dict(zip(names, np.mean([d["obs_pred"] for d in ds], 0))),
            dict(zip(names, ds[0]["obs_z"])))


def partial(a, b, c):
    A = np.column_stack([stats.rankdata(c), np.ones(len(c))])
    res = lambda v: stats.rankdata(v) - A @ np.linalg.lstsq(A, stats.rankdata(v), rcond=None)[0]
    return stats.pearsonr(res(a), res(b))[0]


def perm_p(a, b, c, n=20000):
    """Permute b within the fit to c (Freedman-Lane style on ranks):
    shuffle b's residuals from c, then add back the fitted part."""
    obs = partial(a, b, c)
    rb = stats.rankdata(b)
    A = np.column_stack([stats.rankdata(c), np.ones(len(c))])
    fit = A @ np.linalg.lstsq(A, rb, rcond=None)[0]
    res = rb - fit
    cnt = 0
    for _ in range(n):
        bb = fit + rng.permutation(res)
        cnt += abs(partial(a, bb, c)) >= abs(obs) - 1e-12
    return obs, (cnt + 1) / (n + 1)


r, rz = ens("cnn_preds/s9991_sfimg_s*_preds.npz")
x, _ = ens("cnn_preds/s9991_x_s*_preds.npz")
both = sorted(set(r) & set(x))
a = np.array([r[k] for k in both]); b = np.array([x[k] for k in both])
z = np.array([rz[k] for k in both])
rho, p = stats.spearmanr(a, b)
print(f"  plain Spearman {rho:+.2f}, p {p:.4f} (n={len(both)})")
pz, pp = perm_p(a, b, z)
n = len(a); t = pz * np.sqrt((n - 3) / (1 - pz ** 2))
print(f"  partial | z: {pz:+.2f}; reported (pearsonr, n-2) "
      f"{stats.pearsonr(a, b)[1] if False else '':}t-test n-3 p "
      f"{2 * stats.t.sf(abs(t), n - 3):.4f}; permutation p {pp:.4f}")
m = [k in wl.index for k in both]
am, bm = a[m], b[m]
cm = np.array([wl.loc[k, "log_m500c"] for k, q in zip(both, m) if q])
pm, ppm = perm_p(am, bm, cm)
n = len(am); t = pm * np.sqrt((n - 3) / (1 - pm ** 2))
print(f"  partial | WL mass: {pm:+.2f} (n={n}); t-test n-3 p "
      f"{2 * stats.t.sf(abs(t), n - 3):.3f}; permutation p {ppm:.3f}")

print("\n======== F8b: relic Spearman with halo-level resampling")
with h5py.File("relic_catalog_radio_nh4_144.h5") as f:
    d = f["d_max"][:] / 1000.0
    mt = f["merger_tsc"][:]
    mr = f["mass_ratio"][:]
for lab, sel in (("TSC <= 2", np.isfinite(mt) & (mt <= 2)),
                 ("TSC <= 2, mass ratio >= 0.1", np.isfinite(mt) & (mt <= 2) & (mr >= 0.1))):
    D, T = d[sel], mt[sel]                      # (halos, 3)
    keep = np.isfinite(D)
    flat = stats.spearmanr(D[keep], np.repeat(T[:, None], 3, 1)[keep])
    # halo-level: one value per halo (projection mean), and a cluster bootstrap
    dm = np.nanmean(D, axis=1)
    ok = np.isfinite(dm)
    halo = stats.spearmanr(dm[ok], T[ok])
    boots = []
    idx = np.arange(len(D))
    for _ in range(5000):
        s = rng.choice(idx, len(idx))
        Ds, Ts = D[s], np.repeat(T[s][:, None], 3, 1)
        k = np.isfinite(Ds)
        boots.append(stats.spearmanr(Ds[k], Ts[k])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {lab}: flattened rho {flat[0]:+.3f} (p {flat[1]:.2g}, n={keep.sum()} "
          f"projections); halo-mean rho {halo[0]:+.3f} (p {halo[1]:.2g}, n={ok.sum()} "
          f"halos); cluster-bootstrap 95% CI on flattened rho [{lo:+.3f}, {hi:+.3f}]")
