#!/usr/bin/env python3
"""Regenerated radio model ("syn", Chuiyang's Radio_Snap*_new weights) vs the
runs it replaces (old HB07-Psi weights), every radio-dependent paper number.

Same estimators as paper_realistic_numbers.py: clean models by per-seed R2
(paired seeds); realistic models by pooled (per-seed cluster-mean R2), 1v-5
(single observations, 5-seed ensemble) and 1v-1, scored on snapshot 99; old
vs new realistic contrasts and the paper's within-model increments by paired
parent-halo bootstrap on 1v-5 (conditional on the fits). Real clusters:
25-model ensembles, Freedman-Lane partials.

Guards (review R26-D): every seed of every set is validated (identities,
truths, finite predictions, real-target key order); every contrast reports
outer-fold, inner-set and noisy-mass agreement per seed. Old (fix_*, r4_*)
and new runs have DIFFERENT outer folds (the old ones predate group_folds)
and the old/new mass arms have different simulated measurements, so old ->
new contrasts are descriptive comparisons of fitted pipelines, not
split-matched ablations; contrasts labelled matched reject any mismatch.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

SEEDS = range(42, 47)
key = lambda s: "".join(str(s).split()).replace("_", "").upper()
rng = np.random.default_rng(0)
def load(pat):
    ds = [np.load(pat.format(s) if "/" in pat else f"cnn_preds/{pat.format(s)}", allow_pickle=True)
          for s in SEEDS]
    validate_set(ds, pat)
    return ds


def validate_set(ds, name):
    """All seeds of one set share identities/truths/key order; predictions finite."""
    r = ds[0]
    for i, d in enumerate(ds):
        for k in ("group_id", "halo_id", "snap", "tsc", "y_sample", "label", "obs_name",
                  "sample_halo", "sample_realization"):
            if k in r and (k not in d or not np.array_equal(d[k], r[k])):
                raise SystemExit(f"{name} seed index {i}: {k} differs from the first seed")
        for k in ("oof_single", "oof_pred", "oof_sample"):
            if k in d and not np.isfinite(d[k]).all():
                raise SystemExit(f"{name} seed index {i}: non-finite {k}")
    if "obs_name" in r:
        keys = [key(n) for n in r["obs_name"]]
        if len(set(keys)) != len(keys):
            raise SystemExit(f"{name}: duplicate normalised real-target keys")


def split_status(da, db):
    """Per-seed agreement of outer folds, inner sets and noisy mass inputs."""
    out = {}
    for k, nm in (("fold_of", "outer folds"), ("sel_groups_flat", "inner sets"),
                  ("sim_logm_input", "mass inputs")):
        if all(k in a and k in b for a, b in zip(da, db)):
            out[nm] = sum(np.array_equal(a[k], b[k]) for a, b in zip(da, db))
        else:
            out[nm] = None                      # unavailable in at least one archive
    return out


CLEAN_REF = {}


def clean_r2(pat):
    ds = load(pat)
    ref = CLEAN_REF.setdefault("ids", (ds[0]["halo_id"], ds[0]["label"]))
    if not (np.array_equal(ds[0]["halo_id"], ref[0]) and np.array_equal(ds[0]["label"], ref[1])):
        CLEAN_REF.setdefault("other", []).append(pat)     # e.g. the 99+91 set: compared within its pair only
    return np.array([r2_score(d["label"], d["oof_pred"]) for d in ds])


def paired(lab, a, b):
    t = stats.ttest_rel(b, a)
    print(f"  {lab:<44} {a.mean():.3f} ± {a.std(ddof=1):.3f} -> {b.mean():.3f} ± {b.std(ddof=1):.3f}  "
          f"diff {np.mean(b - a):+.3f} ± {np.std(b - a, ddof=1):.3f}  p={t.pvalue:.3f}  "
          f"({int((b > a).sum())}/5 seeds higher)")


LAST_STATUS = [""]


def single(ds, snap99=True):
    R = len(ds[0]["oof_sample"]) // len(ds[0]["tsc"])
    ms = (np.repeat(ds[0]["snap"], R) == 99) if snap99 else np.ones(len(ds[0]["oof_sample"]), bool)
    P = ds[0]["oof_single"].shape[1]
    y = np.repeat(ds[0]["y_sample"][ms][:, None], P, 1).ravel()
    sgrp = np.repeat(ds[0]["group_id"], R)          # older files lack sample_group
    g = np.repeat(sgrp[ms][:, None], P, 1).ravel()
    return y, g, [d["oof_single"][ms].ravel() for d in ds]


def scores(ds):
    m = ds[0]["snap"] == 99
    pooled = [r2_score(d["tsc"][m], d["oof_cluster"][m]) for d in ds]
    y, _, sg = single(ds)
    return (np.mean(pooled), np.std(pooled, ddof=1), r2_score(y, np.mean(sg, 0)),
            np.mean([r2_score(y, q) for q in sg]),
            np.mean([stats.spearmanr(d["oof_cluster"][m], np.asarray(d["sim_logm"])[m])[0] for d in ds]))


def same_rows(a, b):
    """Pair only identical label rows: per-row ids, snapshots and labels, plus
    the per-sample fields both files carry."""
    for k in ("group_id", "halo_id", "snap", "tsc", "y_sample"):
        if not np.array_equal(a[0][k], b[0][k]):
            raise SystemExit(f"cannot pair: {k} differs")
    for k in ("sample_halo", "sample_realization"):
        if k in a[0] and k in b[0] and not np.array_equal(a[0][k], b[0][k]):
            raise SystemExit(f"cannot pair: {k} differs")
    # Folds may differ: the old fix_* runs predate group_folds (sklearn
    # GroupKFold, CPU-dependent tie order). Both are out-of-fold for every
    # row, so pairing by row is valid but the contrast includes split noise.
    if "fold_of" in a[0] and "fold_of" in b[0]:
        return bool(np.array_equal(a[0]["fold_of"], b[0]["fold_of"]))
    return None


def boot_diff(pa, pb, n=4000, matched=False, snap99=True):
    da, db = load(pa), load(pb)
    same_rows(da, db)
    st = split_status(da, db)
    if matched and any(v is not None and v != len(da) for v in st.values()):
        raise SystemExit(f"{pa} vs {pb}: labelled matched but split/measurement agreement {st}")
    if matched and st["outer folds"] is None:
        raise SystemExit(f"{pa} vs {pb}: labelled matched but folds unavailable")
    LAST_STATUS[0] = ", ".join(f"{k} {'n/a' if v is None else f'{v}/{len(da)}'}" for k, v in st.items())
    y, g, sa = single(da, snap99)
    _, _, sb = single(db, snap99)
    ea, eb = np.mean(sa, 0), np.mean(sb, 0)
    ug = np.unique(g)
    idx = {h: np.where(g == h)[0] for h in ug}
    r = np.random.default_rng(0)
    bs = []
    for _ in range(n):
        s_ = np.concatenate([idx[h] for h in r.choice(ug, len(ug))])
        bs.append(r2_score(y[s_], eb[s_]) - r2_score(y[s_], ea[s_]))
    return r2_score(y, eb) - r2_score(y, ea), np.percentile(bs, [2.5, 97.5])


print("======== clean simulation (per-seed R2, paired seeds; old HB07 -> new synch)")
for lab, o, n_ in (("pooled CNN, snapshot 99", "pooled_nh4_144_inner_s{}.npz", "pooled_syn_inner_s{}.npz"),
                   ("pooled CNN, 99+91", "pooled_s9991_inner_s{}.npz", "pooled_syn_s9991_inner_s{}.npz"),
                   ("shape-only (flux-normalised)", "pooled_shape144_inner_s{}.npz", "pooled_shape_syn_inner_s{}.npz"),
                   ("no density cut", "pooled_nocut144_inner_s{}.npz", "pooled_nocut_syn_inner_s{}.npz"),
                   ("clean joint radio + X-ray", "joint144_clean_inner_s{}.npz", "joint_syn_clean_inner_s{}.npz")):
    paired(lab, clean_r2(o), clean_r2(n_))
print("  within the NEW model:")
full, shape = clean_r2("pooled_syn_inner_s{}.npz"), clean_r2("pooled_shape_syn_inner_s{}.npz")
paired("full flux -> shape-only", full, shape)
print(f"    shape-only retains {shape.mean() / full.mean():.0%} of the full-flux R2")
paired("no cut -> density cut", clean_r2("pooled_nocut_syn_inner_s{}.npz"), full)
paired("radio -> clean joint radio + X-ray", full, clean_r2("joint_syn_clean_inner_s{}.npz"))

print("\n======== realistic (snapshot-99 scoring)")
ROWS = [("radio, physical 1 Mpc", "phys_b1000_sfimg"), ("radio, physical 2 Mpc", "phys_b2000_sfimg"),
        ("radio, physical 3 Mpc", "phys_b3000_sfimg"), ("radio, arcsinh arm 1 Mpc", "asinh_b1000_sfimg"),
        ("radio, arcsinh arm 3 Mpc", "asinh_b3000_sfimg"), ("radio 1 Mpc + X-ray", "phys_b1000_joint"),
        ("radio 2 Mpc + X-ray", "phys_b2000_joint"), ("radio 3 Mpc + X-ray", "phys_b3000_joint")]
OLDNEW = [(lab, f"fix_{t}_s{{}}_preds.npz", f"syn_{t}_s{{}}_preds.npz") for lab, t in ROWS] + [
    ("mass only (fixed measurement)", "r4_massonly_s{}_preds.npz", "syn_massonly_s{}_preds.npz"),
    ("radio 3 Mpc + mass (fixed measurement)", "r4_imgmass_b3000_s{}_preds.npz", "syn_imgmass_b3000_s{}_preds.npz")]
print(f"  {'model':<40}{'old pooled':>16}{'new pooled':>16}{'old 1v-5':>10}{'new 1v-5':>10}"
      f"{'new 1v-1':>10}{'new rho(M)':>12}   new-old 1v-5 [95%]")
for lab, o, n_ in OLDNEW:
    so, sn = scores(load(o)), scores(load(n_))
    dv, iv = boot_diff(o, n_)
    print(f"  {lab:<40}{so[0]:>9.3f}±{so[1]:.3f}{sn[0]:>9.3f}±{sn[1]:.3f}{so[2]:>10.3f}{sn[2]:>10.3f}"
          f"{sn[3]:>10.3f}{sn[4]:>+12.2f}   {dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]   ({LAST_STATUS[0]})")
sx = scores(load("fix_x_s{}_preds.npz"))
print(f"  {'X-ray (unchanged; reference)':<40}{sx[0]:>9.3f}±{sx[1]:.3f}{'':>16}{sx[2]:>10.3f}{'':>10}"
      f"{sx[3]:>10.3f}{sx[4]:>+12.2f}")
print("  (old -> new contrasts: descriptive comparisons of fitted pipelines; outer folds and, for the")
print("   mass arms, the simulated measurements differ -- see the per-row agreement counts)")
print("  paper increments, NEW model (1v-5, paired halo bootstrap):")
for lab, a_, b_, old, matched in (
        ("radio 3 Mpc + mass  -  mass only [matched]", "syn_massonly_s{}_preds.npz",
         "syn_imgmass_b3000_s{}_preds.npz", "+0.099 [0.039, 0.158]", True),
        ("radio 1 Mpc + X-ray  -  X-ray [descriptive]", "fix_x_s{}_preds.npz",
         "syn_phys_b1000_joint_s{}_preds.npz", "+0.031 [-0.014, 0.076]", False)):
    dv, iv = boot_diff(a_, b_, matched=matched)
    print(f"    {lab:<44} {dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]   (old: {old}; {LAST_STATUS[0]})")
print("  SEPARATE ANALYSIS (review R26-B): radio 3 Mpc + X-ray vs the matched 3 Mpc X-ray control")
print("  (runs/followup_20261103T180830Z), snapshot-99 single observations:")
CTRL = "runs/followup_20261103T180830Z/preds/xray_b3000_s{}_preds.npz"
sc = scores(load(CTRL))
sj = scores(load("syn_phys_b3000_joint_s{}_preds.npz"))
dv, iv = boot_diff(CTRL, "syn_phys_b3000_joint_s{}_preds.npz", matched=True)
print(f"    X-ray 3 Mpc control 1v-5 {sc[2]:.3f}; joint 3 Mpc 1v-5 {sj[2]:.3f}; joint - X-ray "
      f"{dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]   ({LAST_STATUS[0]})")
dv, iv = boot_diff(CTRL, "syn_phys_b3000_joint_s{}_preds.npz", matched=True, snap99=False)
print(f"    same, both snapshots (cf. the follow-up summary's mixed-snapshot scores): "
      f"{dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]")


def fl(a, b, c, n=100000):
    ra, rb, rc = (stats.rankdata(v) for v in (a, b, c))
    Z = np.column_stack([np.ones(len(rc)), rc])
    M = np.eye(len(rc)) - Z @ np.linalg.pinv(Z)
    ea, eb = M @ ra, M @ rb
    st = lambda x, y: np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
    obs = st(ea, eb)
    fit = rb - eb
    cnt = sum(abs(st(ea, M @ (fit + rng.permutation(eb)))) >= abs(obs) - 1e-12 for _ in range(n))
    return obs, (cnt + 1) / (n + 1)


def ens(pat):
    ds = load(pat)
    names = [key(n) for n in ds[0]["obs_name"]]
    return (dict(zip(names, np.mean([d["obs_pred"] for d in ds], 0))),
            dict(zip(names, ds[0]["obs_z"])), dict(zip(names, ds[0]["obs_name"])))


print("\n======== real clusters: radio 3 Mpc vs X-ray (25-model ensembles)")
wl = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")
x, _, _ = ens("fix_x_s{}_preds.npz")
for lab, pat in (("OLD radio", "fix_phys_b3000_sfimg_s{}_preds.npz"), ("NEW radio", "syn_phys_b3000_sfimg_s{}_preds.npz")):
    r, rz, rn = ens(pat)
    both = sorted(set(r) & set(x))
    a = np.array([r[k] for k in both]); b = np.array([x[k] for k in both])
    z = np.array([rz[k] for k in both])
    rho, p = stats.spearmanr(a, b)
    pz, ppz = fl(a, b, z)
    m = np.array([k in wl.index for k in both])
    pm, ppm = fl(a[m], b[m], np.array([wl.loc[k, "log_m500c"] for k in np.array(both)[m]]))
    d = a - b
    print(f"  {lab}: n={len(both)} Spearman {rho:+.2f} (p={p:.3f}); partial | z {pz:+.2f} (FL p={ppz:.3f}); "
          f"partial | WL mass {pm:+.2f} (FL p={ppm:.3f}, n={m.sum()}); radio - X-ray median {np.median(d):+.2f} Gyr, "
          f"{int((d > 0).sum())}/{len(d)} radio older")
ro, _, _ = ens("fix_phys_b3000_sfimg_s{}_preds.npz")
rnw, _, rnm = ens("syn_phys_b3000_sfimg_s{}_preds.npz")
ks = sorted(set(ro) & set(rnw))
print(f"  real radio predictions old vs new model (n={len(ks)}): Spearman "
      f"{stats.spearmanr([ro[k] for k in ks], [rnw[k] for k in ks])[0]:+.3f}; median change "
      f"{np.median([rnw[k] - ro[k] for k in ks]):+.2f} Gyr")
print("\n======== A399 / A401 ranks (1 = youngest), new radio model")
for lab, pat in (("mass only", "syn_massonly_s{}_preds.npz"), ("radio 3 Mpc", "syn_phys_b3000_sfimg_s{}_preds.npz"),
                 ("radio 3 Mpc + mass", "syn_imgmass_b3000_s{}_preds.npz"),
                 ("radio 1 Mpc + X-ray", "syn_phys_b1000_joint_s{}_preds.npz")):
    e, _, _ = ens(pat)
    kk = list(e); rk = stats.rankdata([e[k] for k in kk])
    print(f"  {lab:<22} " + "  ".join(f"{t} {e[t]:.2f} ({int(rk[kk.index(t)])}/{len(kk)})"
                                      for t in ("A399", "A401") if t in e))
