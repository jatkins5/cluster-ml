"""Why do the faintest real clusters get X-ray predictions beyond 4 Gyr?

A763, A2443 and A2050 are predicted at 7.4, 5.1 and 4.6 Gyr -- beyond every
training label -- and are the three faintest real clusters. Either they are
fainter than anything the model saw, or something else about faint real
images (background level, chip gaps, the ratio of source to background)
differs from faint mocks. Measure rather than guess:

  1. per-image statistics in the model's input units, real vs mock, as a
     separation in mock standard deviations and a KS test (the same
     diagnostic that found the radio white-noise bug);
  2. for each real cluster, where it falls among mocks at the same redshift
     and exposure, statistic by statistic -- is it outside the mock range?
  3. brightness at matched weak-lensing mass and redshift: is TNG's L_X
     excess a constant 3.6x (what the rebuild assumed) or mass-dependent?
  4. what the model does with the faintest *mocks*: if equally faint mocks
     are predicted inside the label range, faintness alone is not the cause.
"""
import glob
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from build_xray_realistic import GRID, aperture_blocks

H5 = "xray_real_placed_lx.h5"
key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()

with h5py.File(H5) as f:
    a = float(f.attrs["stretch_scale"])
    lx = float(f.attrs.get("lx_scale", 1.0))
    mimg = f["mock/images"][:]                    # (N, R, P, H, W)
    mz = f["mock/z"][:]                           # (N, R)
    mt = f["mock/exposure_ks"][:]
    mh = f["mock/halo_id"][:]
    mtsc = f["mock/pseudo_tsc"][:]
    oimg = f["obs/images"][:]
    oz = f["obs/z"][:]
    ot = f["obs/exposure_ks"][:]
    on = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/name"][:]]
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
mlogm = np.array([mm[h] for h in mh])
N, R, P = mimg.shape[:3]

yy, xx = np.mgrid[:GRID, :GRID]
rr = np.hypot(yy - GRID / 2 + 0.5, xx - GRID / 2 + 0.5)


def feats(img, z, t_ks):
    """Statistics of one input image. `img` is arcsinh(rate / a)."""
    rate = np.sinh(img.astype(np.float64)) * a        # counts / ks / block
    R0 = aperture_blocks(z)
    ap = rr <= R0
    counts = rate * t_ks
    tot = rate[ap].sum()
    outer = ap & (rr > 0.7 * R0)
    core = rr <= 0.15 * R0
    mid = ap & (rr > 0.3 * R0) & (rr <= 0.5 * R0)
    bkg = np.median(rate[outer]) if outer.any() else np.nan
    gy, gx = np.gradient(img.astype(np.float64))
    return dict(
        log_total_rate=np.log10(max(tot, 1e-9)),
        log_counts=np.log10(max(counts[ap].sum(), 1.0)),
        bkg_rate=bkg,
        src_to_bkg=tot / max(bkg * ap.sum(), 1e-9),
        core_frac=rate[core].sum() / max(tot, 1e-9),
        mid_over_core=(rate[mid].mean() / max(rate[core].mean(), 1e-9)),
        zero_frac=np.mean(counts[ap] == 0),
        peak_input=img[ap].max(),
        mean_input=img[ap].mean(),
        roughness=np.mean(np.hypot(gy, gx)[ap]) / max(img[ap].mean(), 1e-9),
    )


mrows = []
for i in range(N):
    for r in range(R):
        for p in range(P):
            d = feats(mimg[i, r, p], mz[i, r], mt[i, r])
            d.update(i=i, z=mz[i, r], t=mt[i, r], logm=mlogm[i], tsc=mtsc[i])
            mrows.append(d)
M = pd.DataFrame(mrows)
O = pd.DataFrame([dict(feats(oimg[j], oz[j], ot[j]), name=on[j], z=oz[j],
                       t=ot[j]) for j in range(len(oimg))])
S = [c for c in M.columns if c not in ("i", "z", "t", "logm", "tsc")]

print(f"{len(M)} mock images ({N} clusters x {R} x {P}), {len(O)} real; "
      f"stretch a={a:.4g}, L_X scale {lx:.3f}")
print("\n======== 1. whole-sample separation, model input units")
print(f"{'statistic':<16}{'mock med':>10}{'real med':>10}{'sep (sd)':>10}{'KS p':>10}")
for c in S:
    mv, ov = M[c].dropna(), O[c].dropna()
    sep = (ov.median() - mv.median()) / mv.std()
    print(f"{c:<16}{mv.median():>10.3g}{ov.median():>10.3g}{sep:>10.2f}"
          f"{stats.ks_2samp(mv, ov).pvalue:>10.2g}")

print("\n======== 2. each real cluster vs mocks at similar z (|dz|<0.015) and "
      "exposure (x0.5-2): percentile")
pred = {}
for p in sorted(glob.glob("cnn_preds/xlx_s*_preds.npz")):
    d = np.load(p, allow_pickle=True)
    for n, v in zip(d["obs_name"], d["obs_pred"]):
        pred.setdefault(str(n), []).append(float(v))
show = ["log_counts", "log_total_rate", "bkg_rate", "src_to_bkg", "core_frac",
        "zero_frac", "roughness"]
print(f"{'cluster':<18}{'pred':>6}{'n mock':>7}" + "".join(f"{c[:11]:>12}" for c in show))
for _, o in O.sort_values("log_total_rate").iterrows():
    sel = M[(np.abs(M.z - o.z) < 0.015) & (M.t > 0.5 * o.t) & (M.t < 2 * o.t)]
    if len(sel) < 20:
        sel = M[np.abs(M.z - o.z) < 0.015]
    pc = [stats.percentileofscore(sel[c].dropna(), o[c]) for c in show]
    pv = np.mean(pred.get(o["name"], [np.nan]))
    print(f"{o['name']:<18}{pv:>6.2f}{len(sel):>7}"
          + "".join(f"{v:>12.0f}" for v in pc))
print("  (percentile of the real value among matched mocks; 0 or 100 = "
      "outside everything the model saw)")

print("\n======== 3. brightness at matched weak-lensing mass and redshift")
wl = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")["log_m500c"]
rows = []
for _, o in O.iterrows():
    k = key(o["name"])
    if k not in wl.index:
        continue
    lm = float(wl[k])
    sel = M[(np.abs(M.z - o.z) < 0.02) & (np.abs(M.logm - lm) < 0.15)]
    if len(sel) < 5:
        continue
    ratio = 10 ** (o.log_total_rate - sel.log_total_rate.median())
    rows.append((o["name"], lm, len(sel), ratio))
    print(f"  {o['name']:<18} log M_WL {lm:.2f}  {len(sel):>4} mocks  "
          f"real/mock rate {ratio:.2f}  (after the {lx:.2f} correction)")
if len(rows) >= 5:
    lm_, r_ = np.array([r[1] for r in rows]), np.log10([r[3] for r in rows])
    s = stats.linregress(lm_, r_)
    print(f"  log(real/mock) vs log M: slope {s.slope:+.2f} +- {s.stderr:.2f}, "
          f"Spearman {stats.spearmanr(lm_, r_)[0]:+.2f}; median ratio "
          f"{10 ** np.median(r_):.2f} (1.0 = correction exactly right)")

print("\n======== 4. what the X-ray model predicts for the faintest mocks")
oof = {}
for p in sorted(glob.glob("cnn_preds/xlx_s*_preds.npz")):
    d = np.load(p, allow_pickle=True)
    for h, v in zip(d["halo_id"], d["oof_cluster"]):
        oof.setdefault(int(h), []).append(float(v))
C = M.groupby("i").agg(log_total_rate=("log_total_rate", "mean"),
                        tsc=("tsc", "first"))
C["pred"] = [np.mean(oof.get(int(mh[i]), [np.nan])) for i in C.index]
faint = C.log_total_rate <= C.log_total_rate.quantile(0.1)
print(f"  faintest 10% of mocks ({faint.sum()} clusters): predicted "
      f"median {C.pred[faint].median():.2f}, max {C.pred[faint].max():.2f} Gyr; "
      f"true median {C.tsc[faint].median():.2f}")
print(f"  all mocks: prediction range {C.pred.min():.2f}-{C.pred.max():.2f} Gyr; "
      f"rho(pred, brightness) {stats.spearmanr(C.pred, C.log_total_rate)[0]:+.2f}")
print(f"  faintest mock log rate {C.log_total_rate.min():.2f}; real clusters "
      f"below it: {', '.join(O.name[O.log_total_rate < C.log_total_rate.min()]) or 'none'}")
