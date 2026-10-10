"""Review F5 acceptance checks for the X-ray count model (--count-model real).

1. Monte Carlo of the processed rate for one block, against the analytic
   mean and variance Codex wrote down (CODEX_CLAUDE_REVIEW.md, round 1):
       R = (C - P_hat) / T_eq + B_ref,  C ~ Poisson(S T_eq + B T_live)
       E[R] = S + B_ref,  Var[R] = (S T_eq + B T_live)/T_eq^2 + Var(P_hat)/T_eq^2
   for faint and bright blocks, short and long exposures, using the same
   binomial-thinning steps as build_placed_real.
2. Real vs mock noise in rate units: per real target, the variance of the
   high-pass residual (rate - Gaussian(rate, 2 blocks)) inside the aperture,
   against the mocks assigned to that same target (new model: stored target
   identity; old model: matched by redshift), old vs new count model.
   Rates are recovered as sinh(image) * stretch scale.
"""
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

from build_xray_realistic import aperture_mask

rng = np.random.default_rng(0)
key = lambda s: "".join(str(s).split()).replace("_", "").upper()

print("======== 1. processed-rate mean and variance (Monte Carlo, 2e5 draws)")
T_MOCK = 2000.0
for S, B, Te, Tl, sig in [(0.002, 0.0008, 15.0, 9.0, 0.013),
                          (0.002, 0.0008, 170.0, 200.0, 0.003),
                          (0.2, 0.0008, 15.0, 9.0, 0.013),
                          (0.2, 0.0008, 230.0, 165.0, 0.003)]:
    n = 200000
    src2ms = rng.poisson(S * T_MOCK, n)              # focused photons at 2 Ms
    part2ms = rng.poisson(B * T_MOCK, n)             # particle at 2 Ms, separate
    p_src = Te / T_MOCK
    p_top = max(Tl / T_MOCK - p_src, 0.0)
    # baked-in particles are thinned with the source (f_mock = 1 here)
    baked = rng.poisson(B * T_MOCK, n)
    C = (rng.binomial(src2ms, p_src) + rng.binomial(baked, p_src)
         + rng.binomial(part2ms, p_top))
    present = B * T_MOCK * (p_src + p_top)
    eps = rng.normal(0, sig, n)
    Bref = 0.0008
    R = (C - present * (1 + eps)) / Te + Bref
    EB = B * T_MOCK * (p_src + p_top)                # particle counts present
    var_pred = (S * Te + EB) / Te ** 2 + (present * sig) ** 2 / Te ** 2
    excess = max(p_src - Tl / T_MOCK, 0) * B * T_MOCK
    print(f"  S={S:<6} T_eq={Te:>5} T_live={Tl:>5}: E[R] {R.mean():.6f} vs {S + Bref:.6f}; "
          f"Var {R.var():.3e} vs {var_pred:.3e} (ratio {R.var() / var_pred:.3f}); "
          f"particle counts present {EB:.3f} vs wanted {B * Tl:.3f} "
          f"(over-kept {excess:.3f})")

print("\n======== 2. real vs mock high-pass noise, per target (snapshot-99 mocks)")
tx = pd.read_csv("xray_target_exposures.csv").set_index("key")


def hp_var(rate, z):
    """Robust pixel noise of the processed rate in the OUTER aperture (60-100%
    of its radius, where cluster structure is weakest): 1.4826 x MAD of the
    high-pass residual, squared, so compact sources in the real images do not
    dominate as they do a plain variance. Plus the mean rate there."""
    from build_xray_realistic import aperture_blocks, GRID
    yy, xx = np.mgrid[:GRID, :GRID]
    rr = np.hypot(yy - GRID / 2 + 0.5, xx - GRID / 2 + 0.5) / aperture_blocks(z)
    m = (rr >= 0.6) & (rr <= 1.0)
    res = (rate - ndimage.gaussian_filter(rate, 2.0))[m]
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    return mad ** 2, rate[m].mean()


def load(path):
    with h5py.File(path) as f:
        a = float(f.attrs["stretch_scale"])
        snap = f["mock/snap"][:] if "mock/snap" in f else None
        sel = slice(None) if snap is None else (snap == 99)
        mi = f["mock/images"][:][sel]
        mz = f["mock/z"][:][sel]
        mt = f["mock/target"][:][sel] if "mock/target" in f else None
        oi, oz = f["obs/images"][:], f["obs/z"][:]
        ok = [key(s.decode() if isinstance(s, bytes) else s) for s in f["obs/key"][:]]
    return a, mi, mz, mt, oi, oz, ok


rows = []
for lab, path in (("old model", "xray_placed_144_s9991_lxcal.h5"),
                  ("new model", "xray_fix_s9991.h5")):
    a, mi, mz, mt, oi, oz, ok = load(path)
    for j, k in enumerate(ok):
        if k not in tx.index:
            continue
        z = float(oz[j])
        vr, mr = hp_var(np.sinh(oi[j].astype(float)) * a, z)
        zmatch = np.abs(mz - z) < 5e-4
        if mt is not None:
            # target identity AND its redshift: mocks at redshifts with no
            # Chandra target borrow a random target's exposure but keep their
            # own redshift, so identity alone mixes in other distances
            tk = np.vectorize(lambda s: key(s.decode() if isinstance(s, bytes) else s))(mt)
            sel = np.argwhere((tk == k) & zmatch)
        else:
            sel = np.argwhere(zmatch)
        if len(sel) == 0:
            continue
        vm, mm = [], []
        for i, r in sel[:60]:
            for p in range(mi.shape[2]):
                v, m_ = hp_var(np.sinh(mi[i, r, p].astype(float)) * a, z)
                vm.append(v), mm.append(m_)
        rows.append(dict(model=lab, target=k, t_eq=tx.loc[k, "t_eq_ks"],
                         noise_ratio=vr / np.median(vm), mean_ratio=mr / np.median(mm),
                         n_mock=len(vm)))
df = pd.DataFrame(rows)
# Poisson noise in rate units scales as mean / T_eq, so with real and mock
# brightness differing (~1.25), the exposure test is noise RELATIVE to mean:
# ~1 if the mocks have the real photon statistics, < 1 if they are too noisy.
df["noise_per_mean"] = df.noise_ratio / df.mean_ratio
print(df.pivot(index="target", columns="model",
               values=["noise_ratio", "mean_ratio", "noise_per_mean"]).round(2).to_string())
for lab, g in df.groupby("model"):
    print(f"  {lab}: real/mock robust outer-aperture noise variance, median {g.noise_ratio.median():.2f} "
          f"(16-84% {g.noise_ratio.quantile(.16):.2f}-{g.noise_ratio.quantile(.84):.2f}); "
          f"mean rate ratio {g.mean_ratio.median():.2f}; noise relative to mean "
          f"{g.noise_per_mean.median():.2f} (16-84% {g.noise_per_mean.quantile(.16):.2f}-"
          f"{g.noise_per_mean.quantile(.84):.2f}, n={len(g)})")
print("  (the old model thinned to raw time, i.e. too few photons, so its mocks should "
      "be NOISIER than the real data: ratio < 1; the new model should move it to ~1.)")
