"""B2 option 3: does dropping the observed flux anchor change the story?

The anchored mocks take their total flux from the Cuciti P150-M500 relation,
so brightness is a steep function of halo mass by construction and a model
reading brightness gets most of the label free. In sim-flux mode the total
comes from the emission model's own predicted power, with one global
constant setting the absolute scale.

Three questions:

  1. Does the mass confound shrink? rho(pred, M500) should fall if the
     anchor was the cause.
  2. Does the signal survive? The simulation's power spans far more decades
     than the observed relation, so faint clusters may sink into the
     background and bright ones saturate.
  3. What P150-M500 relation does the simulation actually predict? With the
     anchor gone, the observed relation becomes a test the mocks either pass
     or fail -- the slope is a physics result either way.
"""
import glob
import re

import h5py
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

PAIRS = [("image only", "img", "sfimg"),
         ("image + blurred mass", "imgmassn", "sfimgmassn"),
         ("mass only", "massonly", "sfmassonly")]

runs = {}
for p in sorted(glob.glob("cnn_preds/b2_*_preds.npz")):
    m = re.search(r"b2_([a-z]+)_s(\d+)_preds\.npz$", p)
    if m:
        runs[(m.group(1), int(m.group(2)))] = np.load(p, allow_pickle=True)


def seeds_for(tag):
    return sorted(s for c, s in runs if c == tag)


def stats_for(tag):
    ss = seeds_for(tag)
    if not ss:
        return None
    d0 = runs[(tag, ss[0])]
    tsc, logm = d0["tsc"], d0["sim_logm"]
    r2s = [r2_score(tsc, runs[(tag, s)]["oof_cluster"]) for s in ss]
    ens = np.mean([runs[(tag, s)]["oof_cluster"] for s in ss], axis=0)
    op = np.mean([runs[(tag, s)]["obs_pred"] for s in ss], axis=0)
    rk = stats.rankdata
    def resid(a, b):
        A = np.column_stack([b, np.ones_like(b)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    partial = stats.pearsonr(resid(rk(ens), rk(logm)),
                             resid(rk(tsc), rk(logm)))[0]
    return dict(n=len(ss), r2=np.mean(r2s), sd=np.std(r2s, ddof=1) if len(ss) > 1
                else np.nan, per_seed=r2s, seeds=ss,
                rho_m=stats.spearmanr(ens, logm)[0], partial=partial,
                obs_med=np.median(op), obs_sd=op.std(),
                obs_min=op.min(), obs_max=op.max(), n_obs=len(op))


print("======== mock OOF R2 by flux mode")
print(f"{'condition':<24}{'anchored':>10}{'sim power':>11}{'delta':>9}"
      f"{'rho(p,M) anch':>15}{'sim':>7}{'partial anch':>14}{'sim':>7}")
res = {}
for label, a, b in PAIRS:
    sa, sb = stats_for(a), stats_for(b)
    res[label] = (sa, sb)
    if sa is None or sb is None:
        print(f"{label:<24}{'--':>10}{'--':>11}")
        continue
    print(f"{label:<24}{sa['r2']:>10.3f}{sb['r2']:>11.3f}"
          f"{sb['r2'] - sa['r2']:>+9.3f}"
          f"{sa['rho_m']:>15.3f}{sb['rho_m']:>7.3f}"
          f"{sa['partial']:>14.3f}{sb['partial']:>7.3f}")

print("\n======== the increment: image over mass-only, within each flux mode")
for mode, img, mo in [("anchored", "imgmassn", "massonly"),
                      ("sim power", "sfimgmassn", "sfmassonly")]:
    si, sm = stats_for(img), stats_for(mo)
    if si is None or sm is None:
        continue
    common = sorted(set(si["seeds"]) & set(sm["seeds"]))
    d = np.array([r2_score(runs[(img, s)]["tsc"], runs[(img, s)]["oof_cluster"])
                  - r2_score(runs[(mo, s)]["tsc"], runs[(mo, s)]["oof_cluster"])
                  for s in common])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    p = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 else np.nan
    print(f"  {mode:<12} {d.mean():+.3f} +- {se:.3f} (SE)  p={p:.4f}  "
          f"{int((d > 0).sum())}/{len(d)} positive  "
          f"(seeds {', '.join(map(str, common))})")

print("\n======== real predictions")
print(f"{'condition':<24}{'mode':<11}{'n':>4}{'median':>9}{'sd':>8}"
      f"{'min':>8}{'max':>8}")
for label, a, b in PAIRS:
    for mode, tag in [("anchored", a), ("sim power", b)]:
        s = stats_for(tag)
        if s:
            print(f"{label:<24}{mode:<11}{s['n_obs']:>4}{s['obs_med']:>9.2f}"
                  f"{s['obs_sd']:>8.2f}{s['obs_min']:>8.2f}{s['obs_max']:>8.2f}")

# ---- what P150-M500 relation does the simulation predict? ----
print("\n======== the simulation's own mass-power relation")
try:
    with h5py.File("injected_nh4_simflux.h5", "r") as f:
        logp = f["mock/logp150"][:]          # (N, R, P)
        logm = f["mock/log_m500"][:]
        off = f.attrs.get("sim_flux_offset_dex", np.nan)
except (OSError, KeyError) as e:
    raise SystemExit(f"sim-flux dataset not readable ({e})")

lp = np.nanmedian(logp.reshape(len(logm), -1), axis=1)
ok = np.isfinite(lp) & np.isfinite(logm)
x, y = logm[ok] - 14.9, lp[ok]
slope, icpt, r, p, se = stats.linregress(x, y)
resid = y - (icpt + slope * x)
print(f"  {ok.sum()} clusters, global offset {off:+.3f} dex applied")
print(f"  simulation : log P150 = {icpt:.2f} + {slope:.2f} "
      f"(+-{se:.2f}) log(M500/10^14.9)")
print(f"  Cuciti+2023: log P150 = 25.60 + 3.55 log(M500/10^14.9), "
      f"scatter 0.35 dex")
print(f"  Pearson r {r:.3f}, scatter about the fit {resid.std():.2f} dex "
      f"(observed 0.35)")
print(f"  simulated power spans {np.percentile(y, 5):.1f} to "
      f"{np.percentile(y, 95):.1f} (5-95%), i.e. "
      f"{np.percentile(y, 95) - np.percentile(y, 5):.1f} dex; the observed "
      f"relation spans about 3")
if abs(slope - 3.55) > 2 * se:
    print(f"  -> slope differs from the observed relation by "
          f"{abs(slope - 3.55) / se:.1f} sigma")

# ---- is the scatter about the relation a merger signal? ----
# Cuciti+2023 Fig. 3: clusters above the observed P150-M500 relation are the
# X-ray-disturbed ones. With the anchor gone, our mocks can either reproduce
# that or not, and it is the same quantity the earlier design was filling
# with N(0, 0.35 dex) of pure noise.
with h5py.File("injected_nh4_simflux.h5", "r") as f:
    tsc_all = f["mock/pseudo_tsc"][:]
tsc_ok = tsc_all[ok]
print("\n======== does the simulation put disturbed clusters above the relation?")
print(f"  corr(log P150, pseudo-TSC)      Spearman "
      f"{stats.spearmanr(y, tsc_ok)[0]:+.3f}  "
      f"(p={stats.spearmanr(y, tsc_ok)[1]:.1e})")
print(f"  corr(residual, pseudo-TSC)      Spearman "
      f"{stats.spearmanr(resid, tsc_ok)[0]:+.3f}  "
      f"(p={stats.spearmanr(resid, tsc_ok)[1]:.1e})")
print("  negative means brighter-than-the-relation clusters merged more "
      "recently,\n  which is the direction Cuciti+2023 Fig. 3 reports for "
      "X-ray disturbance.")
lo, hi = np.quantile(tsc_ok, [1 / 3, 2 / 3])
for lab, b in [("recent (low TSC)", tsc_ok <= lo),
               ("middle", (tsc_ok > lo) & (tsc_ok <= hi)),
               ("relaxed (high TSC)", tsc_ok > hi)]:
    print(f"    {lab:<20} n={b.sum():3d}  median residual "
          f"{np.median(resid[b]):+.2f} dex")
