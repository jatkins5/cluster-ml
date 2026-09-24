"""Is there an error in the X-ray pipeline, or is X-ray genuinely weak here?

Collaborators expected X-ray to help and it did not (X-ray-only OOF R2 0.321
vs radio 0.511; dual-encoder no better than radio alone). Before accepting
that, check the places an error would hide:

  1. Stretch. dataset_xray_*.h5 stores arcsinh(counts/s/pixel). arcsinh only
     compresses values >~1; if count rates are ~1e-4 it is the identity, the
     map spans orders of magnitude un-stretched, and after global
     standardisation the CNN sees a bright core and near-zeros.
  2. Projection pairing. The builder maps radio (xy, yz, xz) to X-ray
     (snap99_z, snap99_x, snap99_y). Test empirically: on a common +-1 r500
     field, matched radio/X-ray pairs should resemble each other more than
     mismatched ones, and the best in-plane transform reveals any transpose.
  3. Centring. Where is the X-ray peak relative to the image centre?
  4. Is the signal there at all? Standard X-ray dynamical-state indicators
     (concentration c, centroid shift w) are the literature baseline. If they
     predict TSC and the CNN does not beat them, the CNN input is suspect;
     if they barely predict it either, X-ray is genuinely weak for this label.
"""
import h5py
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KF = KFold(5, shuffle=True, random_state=0)


def ridge_oof(X, y):
    X = X if X.ndim == 2 else X[:, None]
    return cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(
        alphas=np.logspace(-3, 3, 13))), X, y, cv=KF)


with h5py.File("dataset_xray_128.h5") as f:
    xa = f["images"][:].astype(np.float64)
    xh = f["meta/halo_id"][:]
lin = np.sinh(xa)                      # back to counts/s/pixel
N, P, H, _ = lin.shape

# ---------- 1. stretch ----------
print("======== 1. what the arcsinh stretch is doing")
pos = lin[lin > 0]
print(f"  count-rate per pixel: median {np.median(pos):.3g}, "
      f"p99 {np.percentile(pos, 99):.3g}, max {pos.max():.3g}")
print(f"  pixels with value > 1 (where arcsinh starts compressing): "
      f"{(lin > 1).mean():.2%}")
print(f"  max |arcsinh(x) - x| / x over all pixels: "
      f"{np.nanmax(np.abs(xa[lin > 0] - lin[lin > 0]) / lin[lin > 0]):.2e}")
dr = np.array([lin[i, p].max() / max(np.median(lin[i, p][lin[i, p] > 0]), 1e-30)
               for i in range(N) for p in range(P)])
print(f"  per-map dynamic range (max / median): median {np.median(dr):.0f}, "
      f"10-90% {np.percentile(dr, 10):.0f}-{np.percentile(dr, 90):.0f}")
std = (xa - xa.mean()) / xa.std()
yy, xx = np.mgrid[:H, :H]
rr = np.hypot(yy - H / 2 + .5, xx - H / 2 + .5)
core = rr < H * 0.05            # inner 0.1 r500
print(f"  after global standardisation, share of total variance in the "
      f"central 0.1 r500 ({core.mean():.2%} of pixels): "
      f"{(std[:, :, core] ** 2).sum() / (std ** 2).sum():.1%}")
print(f"  median standardised pixel outside 0.5 r500: "
      f"{np.median(std[:, :, rr > H * 0.25]):+.3f}  (image min "
      f"{std.min():+.3f})")

# ---------- 3. centring ----------
print("\n======== 3. centring")
off = []
for i in range(N):
    for p in range(P):
        k = np.unravel_index(np.argmax(lin[i, p]), (H, H))
        off.append(np.hypot(k[0] - H / 2 + .5, k[1] - H / 2 + .5))
off = np.array(off) / (H / 2)    # in r500 (half-width = 1 r500)
print(f"  X-ray brightest-pixel offset from image centre: median "
      f"{np.median(off):.3f} r500, 90th {np.percentile(off, 90):.3f}, "
      f">0.2 r500: {(off > 0.2).mean():.1%}")

# ---------- 2. projection pairing ----------
print("\n======== 2. projection pairing against radio")
with h5py.File("dataset_gp_512.h5") as f:
    rimg = f["images"][:]
    rh = f["meta/halo_id"][:]
    tsc_p = f["labels/pseudo_tsc"][:].astype(np.float64)
assert np.array_equal(rh, xh), "halo order differs between radio and X-ray"
# radio 512px spans +-4 r500 -> central +-1 r500 is 128px, same as X-ray.
c0 = 256
rcrop = rimg[:, :, c0 - 64:c0 + 64, c0 - 64:c0 + 64].astype(np.float64)
logx = np.log10(np.clip(lin, np.percentile(pos, 1), None))

TF = [lambda a: a, lambda a: a.T, lambda a: a[::-1], lambda a: a[:, ::-1],
      lambda a: a[::-1, ::-1], lambda a: a.T[::-1], lambda a: a.T[:, ::-1],
      lambda a: a.T[::-1, ::-1]]
TFN = ["id", "T", "flipud", "fliplr", "rot180", "T+flipud", "T+fliplr",
       "T+rot180"]


def sim(a, b):
    """Spearman over the annulus 0.1-1 r500: excludes the core, where both
    peak at the centre regardless of pairing."""
    m = (rr > H * 0.05) & (rr < H * 0.5)
    return stats.spearmanr(a[m], b[m])[0]


# which X-ray projection matches each radio projection best?
M = np.zeros((P, P))
for i in range(0, N, 4):               # every 4th cluster is plenty
    for pr in range(P):
        for px in range(P):
            M[pr, px] += sim(rcrop[i, pr], logx[i, px])
M /= len(range(0, N, 4))
print("  mean radio/X-ray similarity (rows: radio xy,yz,xz; "
      "cols: X-ray as stored 0,1,2)")
for pr, nm in enumerate(["xy", "yz", "xz"]):
    print(f"    {nm}  " + "  ".join(f"{M[pr, px]:+.3f}" for px in range(P))
          + ("   <- diagonal is best" if np.argmax(M[pr]) == pr else
             f"   <- best is col {np.argmax(M[pr])}"))

# is there an in-plane transpose/flip between the two?
best = np.zeros(len(TF))
for i in range(0, N, 4):
    for p in range(P):
        for t, fn in enumerate(TF):
            best[t] += sim(rcrop[i, p], fn(logx[i, p]))
best /= len(range(0, N, 4)) * P
print("  in-plane transform of X-ray that best matches radio: "
      + ", ".join(f"{n} {v:+.3f}" for n, v in zip(TFN, best)))

# ---------- 4. is the signal there? ----------
print("\n======== 4. standard X-ray morphology indicators vs pseudo-TSC")
with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5") as f:
    mt = dict(zip(f["halo_id"][:], f["tsc_gyr"][:]))
tsc_m = np.array([mt.get(h, np.nan) for h in xh])
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
logm = np.array([mm[h] for h in xh])


def centroid(img, r):
    m = rr < r
    w = img * m
    s = w.sum()
    return np.array([(w * yy).sum() / s, (w * xx).sum() / s]) if s > 0 else None


r500px = H / 2
feat = {"c (S<0.1r500 / S<r500)": [], "w centroid shift": [],
        "log total flux": [], "ellipticity": []}
for i in range(N):
    cs, ws, ts, es = [], [], [], []
    for p in range(P):
        im = lin[i, p]
        inner = im[rr < 0.1 * r500px].sum()
        outer = im[rr < r500px].sum()
        cs.append(inner / outer if outer > 0 else np.nan)
        # Bohringer+10: centroid in apertures 0.1..1 r500 around the peak
        cents = [centroid(im, f * r500px) for f in np.linspace(0.1, 1.0, 10)]
        cents = np.array([c for c in cents if c is not None])
        ws.append(cents.std(axis=0).sum() / r500px if len(cents) > 2 else np.nan)
        ts.append(np.log10(outer) if outer > 0 else np.nan)
        m = rr < 0.5 * r500px
        w = im * m
        s = w.sum()
        cy, cx = (w * yy).sum() / s, (w * xx).sum() / s
        cov = np.cov(np.vstack([(yy - cy)[m], (xx - cx)[m]]),
                     aweights=im[m] + 1e-30)
        ev = np.linalg.eigvalsh(cov)
        es.append(1 - np.sqrt(ev[0] / ev[1]))
    feat["c (S<0.1r500 / S<r500)"].append(np.nanmean(cs))
    feat["w centroid shift"].append(np.nanmean(ws))
    feat["log total flux"].append(np.nanmean(ts))
    feat["ellipticity"].append(np.nanmean(es))
feat = {k: np.array(v) for k, v in feat.items()}

print(f"  {'feature':<26}{'rho(pseudo)':>12}{'rho(merger)':>12}"
      f"{'rho(M500)':>11}{'OOF R2 pseudo':>15}")
ok_m = np.isfinite(tsc_m)
for k, v in feat.items():
    good = np.isfinite(v)
    print(f"  {k:<26}{stats.spearmanr(v[good], tsc_p[good])[0]:>12.3f}"
          f"{stats.spearmanr(v[good & ok_m], tsc_m[good & ok_m])[0]:>12.3f}"
          f"{stats.spearmanr(v[good], logm[good])[0]:>11.3f}"
          f"{r2_score(tsc_p[good], ridge_oof(v[good], tsc_p[good])):>15.3f}")
X = np.column_stack(list(feat.values()))
good = np.all(np.isfinite(X), axis=1)
print(f"  {'all four (ridge)':<26}{'':>35}"
      f"{r2_score(tsc_p[good], ridge_oof(X[good], tsc_p[good])):>15.3f}")
Xm = np.column_stack([X, logm])
print(f"  {'all four + log M500':<26}{'':>35}"
      f"{r2_score(tsc_p[good], ridge_oof(Xm[good], tsc_p[good])):>15.3f}")
print(f"  {'log M500 alone':<26}{'':>35}"
      f"{r2_score(tsc_p, ridge_oof(logm, tsc_p)):>15.3f}")
