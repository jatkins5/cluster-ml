#!/usr/bin/env python3
"""Regenerated radio model ("syn", Chuiyang's Radio_Snap*_new weights) vs the
runs it replaces (old HB07-Psi weights), every radio-dependent paper number.

Same estimators as paper_realistic_numbers.py: clean models by per-seed R2
(paired seeds); realistic models by pooled (per-seed cluster-mean R2), 1v-5
(single observations, 5-seed ensemble) and 1v-1, scored on snapshot 99; old
vs new realistic contrasts and the paper's within-model increments by paired
parent-halo bootstrap on 1v-5 (conditional on the fits). Real clusters:
25-model ensembles, Freedman-Lane partials.

Guards (review R26-D/R27), importable without running the analysis:
  validate_set     every seed: identities, truths, row-to-sample expansion,
                   prediction/real-target shapes and finiteness, recorded seed
                   and unchanged-in-run flag where provenance exists
  split_status     per-seed agreement of outer folds, inner (fold, parent)
                   memberships and noisy mass inputs ("n/a" = no mass input,
                   None = unavailable in a legacy archive)
  check_matched    a contrast labelled matched must have all applicable
                   evidence present and equal (missing evidence rejects)
  check_contract   provenance: defining sources, common configuration with
                   only the declared arm differences, joint X-ray input ==
                   control dataset; legacy archives report "unavailable"
  clean_pair       clean comparisons reject a cohort mismatch between arms
Old (fix_*, r4_*) and new runs have DIFFERENT outer folds (the old ones
predate group_folds) and the old/new mass arms different simulated
measurements: old -> new contrasts are descriptive comparisons of fitted
pipelines, not split-matched ablations.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

SEEDS = list(range(42, 47))
key = lambda s: "".join(str(s).split()).replace("_", "").upper()
FREE = {"seed", "out_prefix"}
IDS = ("group_id", "halo_id", "snap", "tsc", "y_sample", "label", "obs_name", "obs_z",
       "sample_halo", "sample_realization", "sample_group", "sample_snap")


def fail(msg):
    raise SystemExit(msg)


def load(pat, seeds=SEEDS):
    ds = [dict(np.load(pat.format(s) if "/" in pat else f"cnn_preds/{pat.format(s)}", allow_pickle=True))
          for s in seeds]
    validate_set(ds, pat, seeds)
    return ds


def prov(d):
    return json.loads(str(d["provenance"])) if "provenance" in d else None


def validate_set(ds, name, seeds=SEEDS):
    """Per-artifact and across-seed validation of one model's seed set."""
    r = ds[0]
    for i, (d, s) in enumerate(zip(ds, seeds)):
        tag = f"{name} seed {s}"
        for k in IDS:
            if k in r and (k not in d or not np.array_equal(d[k], r[k])):
                fail(f"{tag}: {k} differs from the first seed")
        if "label" in d:                                 # clean (train_cnn_pooled) archive
            if d["oof_pred"].shape != d["label"].shape or not np.isfinite(d["oof_pred"]).all():
                fail(f"{tag}: oof_pred must match label shape and be finite")
        else:                                            # realistic (train_cnn_mock) archive
            n = len(d["tsc"])
            ns = len(d["oof_sample"])
            if n == 0 or ns % n:
                fail(f"{tag}: {ns} samples not a multiple of {n} rows")
            R = ns // n
            if d["oof_cluster"].shape != (n,) or d["y_sample"].shape != (ns,) \
                    or d["oof_single"].ndim != 2 or d["oof_single"].shape[0] != ns:
                fail(f"{tag}: prediction shapes inconsistent with {n} rows x R={R}")
            for k in ("oof_cluster", "oof_sample", "oof_single", "obs_pred"):
                if k in r and d[k].shape != r[k].shape:
                    fail(f"{tag}: {k} shape {d[k].shape} != first seed's {r[k].shape}")
            for k in ("oof_cluster", "oof_sample", "oof_single", "tsc", "y_sample"):
                if not np.isfinite(d[k]).all():
                    fail(f"{tag}: non-finite {k}")
            if not np.allclose(d["y_sample"], np.repeat(d["tsc"], R)):
                fail(f"{tag}: y_sample is not the row target expanded over realizations")
            for k, exp in (("sample_group", np.repeat(d["group_id"], R)),
                           ("sample_snap", np.repeat(d["snap"], R)),
                           ("sample_realization", np.tile(np.arange(R), n))):
                if k in d and not np.array_equal(d[k], exp):
                    fail(f"{tag}: {k} inconsistent with the row expansion")
            no = len(d["obs_name"]) if "obs_name" in d else 0
            if "obs_pred" in d:
                if d["obs_pred"].shape != (no,) or ("obs_z" in d and d["obs_z"].shape != (no,)):
                    fail(f"{tag}: real-target arrays must match obs_name ({no})")
                if no and (not np.isfinite(d["obs_pred"]).all()
                           or ("obs_z" in d and not np.isfinite(d["obs_z"]).all())):
                    fail(f"{tag}: non-finite real-target prediction or redshift")
        p = prov(d)
        if p is not None:
            if int(p["config"]["seed"]) != s:
                fail(f"{tag}: recorded seed {p['config']['seed']} != {s}")
            if bool(d.get("provenance_changed_during_run", False)):
                fail(f"{tag}: sources/inputs changed during the run")
    if "obs_name" in r:
        keys = [key(n) for n in r["obs_name"]]
        if len(set(keys)) != len(keys):
            fail(f"{name}: duplicate normalised real-target keys")


def _inner(d):
    if "sel_groups_flat" not in d or "sel_groups_fold" not in d:
        return None
    return set(zip(d["sel_groups_fold"].astype(int).tolist(), d["sel_groups_flat"].astype(int).tolist()))


def split_status(da, db):
    """Per-seed agreement counts; None = unavailable, 'n/a' = not applicable."""
    n = len(da)
    out = {}
    out["outer folds"] = (sum(np.array_equal(a["fold_of"], b["fold_of"]) for a, b in zip(da, db))
                          if all("fold_of" in x for x in da + db) else None)
    ia, ib = [_inner(x) for x in da], [_inner(x) for x in db]
    out["inner (fold, parent)"] = (None if any(v is None for v in ia + ib)
                                   else sum(x == y for x, y in zip(ia, ib)))
    if not all("sim_logm_input" in x for x in da + db):
        out["mass inputs"] = None
    elif all(x["sim_logm_input"].size == 0 for x in da + db):
        out["mass inputs"] = "n/a"
    else:
        out["mass inputs"] = sum(a["sim_logm_input"].size > 0 and np.array_equal(a["sim_logm_input"], b["sim_logm_input"])
                                 for a, b in zip(da, db))
    out["_n"] = n
    return out


def fmt(st):
    n = st["_n"]
    return ", ".join(f"{k} {'unavailable' if v is None else v if v == 'n/a' else f'{v}/{n}'}"
                     for k, v in st.items() if k != "_n")


def check_matched(da, db, st, kind, name):
    """A matched contrast needs every applicable piece of evidence, all equal."""
    n = st["_n"]
    for k in ("outer folds", "inner (fold, parent)"):
        if st[k] is None:
            fail(f"{name}: labelled matched but {k} unavailable")
        if st[k] != n:
            fail(f"{name}: labelled matched but {k} agree in {st[k]}/{n} seeds")
    if kind == "mass":
        if st["mass inputs"] in (None, "n/a") or st["mass inputs"] != n:
            fail(f"{name}: matched mass contrast needs identical noisy mass inputs ({st['mass inputs']})")
        for arm in (da, db):
            m0 = arm[0]["sim_logm_input"]
            if m0.ndim != 1 or m0.size == 0 or not np.isfinite(m0).all():
                fail(f"{name}: mass inputs must be finite, 1-D and nonempty")
            if not all(np.array_equal(x["sim_logm_input"], m0) for x in arm):
                fail(f"{name}: mass inputs differ across seeds of one fixed-measurement arm")
    elif st["mass inputs"] not in ("n/a", None):
        fail(f"{name}: unexpected mass inputs in a non-mass contrast")


def check_contract(da, db, allowed, name, matched, joint_x=None):
    """Provenance and configuration: identical defining sources and common
    settings; only `allowed` keys may differ between arms. joint_x=(control,
    joint) additionally requires the joint's X-ray input to be the control's
    dataset, by path and hash. Returns a status string."""
    pa, pb = [prov(x) for x in da], [prov(x) for x in db]
    if any(p is None for p in pa + pb):
        if matched:
            fail(f"{name}: labelled matched but provenance unavailable")
        return "provenance unavailable (legacy archive)"
    if not matched:                     # descriptive: report, never certify
        try:
            return _contract(pa, pb, allowed, name, joint_x)
        except SystemExit as e:
            return f"descriptive; provenance differs: {e}"
    return _contract(pa, pb, allowed, name, joint_x)


def _contract(pa, pb, allowed, name, joint_x):
    ref = pa[0]
    strip = lambda c, extra: {k: v for k, v in c.items() if k not in FREE | extra}
    for i, p in enumerate(pa + pb):
        if p["sha256"] != ref["sha256"]:
            fail(f"{name}: defining-source hashes differ (run {i})")
    for arm in (pa, pb):
        for p in arm:
            if strip(p["config"], set()) != strip(arm[0]["config"], set()):
                fail(f"{name}: configuration differs across seeds of one arm")
    ca, cb = strip(pa[0]["config"], set(allowed)), strip(pb[0]["config"], set(allowed))
    if ca != cb:
        diff = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        fail(f"{name}: undeclared configuration differences {diff}")
    if joint_x is not None:
        ctrl, joint = (pa, pb) if joint_x == "b_is_joint" else (pb, pa)
        if joint[0]["config"].get("xray_dataset") != ctrl[0]["config"].get("dataset"):
            fail(f"{name}: joint X-ray input is not the control's dataset")
        jh = (joint[0].get("inputs_sha256") or {}).get("xray_dataset")
        ch = (ctrl[0].get("inputs_sha256") or {}).get("dataset")
        if jh is None or ch is None or jh != ch:
            fail(f"{name}: joint X-ray input hash != control dataset hash ({jh} vs {ch})")
    return "provenance checked"


CLEAN_STATUS = {}


def clean_pair(pa, pb, seeds=SEEDS, loader=None):
    da, db = (loader or load)(pa, seeds), (loader or load)(pb, seeds)
    for k in ("halo_id", "label"):                    # required in both
        if k not in da[0] or k not in db[0] or not np.array_equal(da[0][k], db[0][k]):
            fail(f"{pa} vs {pb}: cohort mismatch in {k}")
    unavailable = []
    for k in ("group_id", "snap"):                    # older archives lack these
        if k in da[0] and k in db[0]:
            if not np.array_equal(da[0][k], db[0][k]):
                fail(f"{pa} vs {pb}: cohort mismatch in {k}")
        else:
            unavailable.append(k)
    CLEAN_STATUS[(pa, pb)] = ("halo_id, label identical" +
                              (f"; {', '.join(unavailable)} unavailable in one archive" if unavailable else
                               "; group_id, snap identical"))
    r2 = lambda ds: np.array([r2_score(d["label"], d["oof_pred"]) for d in ds])
    return r2(da), r2(db)


def same_rows(a, b, name):
    for k in ("group_id", "halo_id", "snap", "tsc", "y_sample"):
        if not np.array_equal(a[0][k], b[0][k]):
            fail(f"{name}: cannot pair, {k} differs")
    for k in ("sample_halo", "sample_realization", "sample_group", "sample_snap"):
        if k in a[0] and k in b[0] and not np.array_equal(a[0][k], b[0][k]):
            fail(f"{name}: cannot pair, {k} differs")


def single(ds, snap99=True):
    R = len(ds[0]["oof_sample"]) // len(ds[0]["tsc"])
    ms = (np.repeat(ds[0]["snap"], R) == 99) if snap99 else np.ones(len(ds[0]["oof_sample"]), bool)
    P = ds[0]["oof_single"].shape[1]
    y = np.repeat(ds[0]["y_sample"][ms][:, None], P, 1).ravel()
    g = np.repeat(np.repeat(ds[0]["group_id"], R)[ms][:, None], P, 1).ravel()
    return y, g, [d["oof_single"][ms].ravel() for d in ds]


def scores(ds):
    m = ds[0]["snap"] == 99
    pooled = [r2_score(d["tsc"][m], d["oof_cluster"][m]) for d in ds]
    y, _, sg = single(ds)
    return (np.mean(pooled), np.std(pooled, ddof=1), r2_score(y, np.mean(sg, 0)),
            np.mean([r2_score(y, q) for q in sg]),
            np.mean([stats.spearmanr(d["oof_cluster"][m], np.asarray(d["sim_logm"])[m])[0] for d in ds]))


def contrast(pa, pb, kind="plain", matched=False, allowed=(), joint_x=None, snap99=True, n=4000,
             loader=None):
    """Validated paired halo-bootstrap contrast b - a on 1v-5. Returns
    (delta, interval, status string)."""
    name = f"{pa} vs {pb}"
    da, db = (loader or load)(pa), (loader or load)(pb)
    same_rows(da, db, name)
    st = split_status(da, db)
    if matched:
        check_matched(da, db, st, kind, name)
    pstat = check_contract(da, db, allowed, name, matched, joint_x)
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
    return r2_score(y, eb) - r2_score(y, ea), np.percentile(bs, [2.5, 97.5]), f"{fmt(st)}; {pstat}"


def fl(a, b, c, n=100000, rng=None):
    rng = rng or np.random.default_rng(0)
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


def paired(lab, a, b):
    t = stats.ttest_rel(b, a)
    print(f"  {lab:<44} {a.mean():.3f} ± {a.std(ddof=1):.3f} -> {b.mean():.3f} ± {b.std(ddof=1):.3f}  "
          f"diff {np.mean(b - a):+.3f} ± {np.std(b - a, ddof=1):.3f}  p={t.pvalue:.3f}  "
          f"({int((b > a).sum())}/5 seeds higher)")


def main():
    print("======== clean simulation (per-seed R2, paired seeds; old HB07 -> new synch)")
    for lab, o, n_ in (("pooled CNN, snapshot 99", "pooled_nh4_144_inner_s{}.npz", "pooled_syn_inner_s{}.npz"),
                       ("pooled CNN, 99+91", "pooled_s9991_inner_s{}.npz", "pooled_syn_s9991_inner_s{}.npz"),
                       ("shape-only (flux-normalised)", "pooled_shape144_inner_s{}.npz", "pooled_shape_syn_inner_s{}.npz"),
                       ("no density cut", "pooled_nocut144_inner_s{}.npz", "pooled_nocut_syn_inner_s{}.npz"),
                       ("clean joint radio + X-ray", "joint144_clean_inner_s{}.npz", "joint_syn_clean_inner_s{}.npz")):
        paired(lab, *clean_pair(o, n_))
        print(f"    ({CLEAN_STATUS[(o, n_)]})")
    print("  within the NEW model:")
    full, shape = clean_pair("pooled_syn_inner_s{}.npz", "pooled_shape_syn_inner_s{}.npz")
    paired("full flux -> shape-only", full, shape)
    print(f"    shape-only = {shape.mean() / full.mean():.0%} of the full model's R2")
    paired("no cut -> density cut", *clean_pair("pooled_nocut_syn_inner_s{}.npz", "pooled_syn_inner_s{}.npz"))
    paired("radio -> clean joint radio + X-ray", *clean_pair("pooled_syn_inner_s{}.npz", "joint_syn_clean_inner_s{}.npz"))

    print("\n======== realistic (snapshot-99 scoring)")
    rows = [("radio, physical 1 Mpc", "phys_b1000_sfimg"), ("radio, physical 2 Mpc", "phys_b2000_sfimg"),
            ("radio, physical 3 Mpc", "phys_b3000_sfimg"), ("radio, arcsinh arm 1 Mpc", "asinh_b1000_sfimg"),
            ("radio, arcsinh arm 3 Mpc", "asinh_b3000_sfimg"), ("radio 1 Mpc + X-ray", "phys_b1000_joint"),
            ("radio 2 Mpc + X-ray", "phys_b2000_joint"), ("radio 3 Mpc + X-ray", "phys_b3000_joint")]
    oldnew = [(lab, f"fix_{t}_s{{}}_preds.npz", f"syn_{t}_s{{}}_preds.npz") for lab, t in rows] + [
        ("mass only (fixed measurement)", "r4_massonly_s{}_preds.npz", "syn_massonly_s{}_preds.npz"),
        ("radio 3 Mpc + mass (fixed measurement)", "r4_imgmass_b3000_s{}_preds.npz", "syn_imgmass_b3000_s{}_preds.npz")]
    print(f"  {'model':<40}{'old pooled':>16}{'new pooled':>16}{'old 1v-5':>10}{'new 1v-5':>10}"
          f"{'new 1v-1':>10}{'new rho(M)':>12}   new-old 1v-5 [95%] (descriptive)")
    for lab, o, n_ in oldnew:
        so, sn = scores(load(o)), scores(load(n_))
        dv, iv, st = contrast(o, n_, allowed=("dataset", "xray_dataset", "mass_seed", "fold_manifest"))
        print(f"  {lab:<40}{so[0]:>9.3f}±{so[1]:.3f}{sn[0]:>9.3f}±{sn[1]:.3f}{so[2]:>10.3f}{sn[2]:>10.3f}"
              f"{sn[3]:>10.3f}{sn[4]:>+12.2f}   {dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]   ({st})")
    sx = scores(load("fix_x_s{}_preds.npz"))
    print(f"  {'X-ray (unchanged; reference)':<40}{sx[0]:>9.3f}±{sx[1]:.3f}{'':>16}{sx[2]:>10.3f}{'':>10}"
          f"{sx[3]:>10.3f}{sx[4]:>+12.2f}")
    print("  paper increments, NEW model (1v-5, paired halo bootstrap):")
    dv, iv, st = contrast("syn_massonly_s{}_preds.npz", "syn_imgmass_b3000_s{}_preds.npz", kind="mass",
                          matched=True, allowed=("no_image",))
    print(f"    radio 3 Mpc + mass - mass only [matched]     {dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]   ({st})")
    dv, iv, st = contrast("fix_x_s{}_preds.npz", "syn_phys_b1000_joint_s{}_preds.npz",
                          allowed=("dataset", "xray_dataset"))
    print(f"    radio 1 Mpc + X-ray - X-ray [descriptive]    {dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]   ({st})")
    print("  SEPARATE ANALYSIS (review R26-B): radio 3 Mpc + X-ray vs the matched 3 Mpc X-ray control")
    ctrl = "runs/followup_20261103T180830Z/preds/xray_b3000_s{}_preds.npz"
    sc, sj = scores(load(ctrl)), scores(load("syn_phys_b3000_joint_s{}_preds.npz"))
    for snap99, lab in ((True, "snapshot 99"), (False, "both snapshots")):
        dv, iv, st = contrast(ctrl, "syn_phys_b3000_joint_s{}_preds.npz", matched=True,
                              allowed=("dataset", "xray_dataset"), joint_x="b_is_joint", snap99=snap99)
        print(f"    {lab}: X-ray control 1v-5 {sc[2]:.3f}; joint 1v-5 {sj[2]:.3f}; joint - X-ray "
              f"{dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]   ({st})" if snap99 else
              f"    {lab}: joint - X-ray {dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]")

    print("\n======== real clusters: radio 3 Mpc vs X-ray (25-model ensembles)")
    wl = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")
    x, _, _ = ens("fix_x_s{}_preds.npz")
    for lab, pat in (("OLD radio", "fix_phys_b3000_sfimg_s{}_preds.npz"),
                     ("NEW radio", "syn_phys_b3000_sfimg_s{}_preds.npz")):
        r, rz, _ = ens(pat)
        both = sorted(set(r) & set(x))
        a = np.array([r[k] for k in both]); b = np.array([x[k] for k in both])
        z = np.array([rz[k] for k in both])
        rho, p = stats.spearmanr(a, b)
        pz, ppz = fl(a, b, z)
        m = np.array([k in wl.index for k in both])
        pm, ppm = fl(a[m], b[m], np.array([wl.loc[k, "log_m500c"] for k in np.array(both)[m]]))
        d = a - b
        print(f"  {lab}: n={len(both)} Spearman {rho:+.2f} (p={p:.3f}); partial | z {pz:+.2f} (FL p={ppz:.3f}); "
              f"partial | WL mass {pm:+.2f} (FL p={ppm:.3f}, n={m.sum()}); radio - X-ray median "
              f"{np.median(d):+.2f} Gyr, {int((d > 0).sum())}/{len(d)} radio older")
    ro, _, _ = ens("fix_phys_b3000_sfimg_s{}_preds.npz")
    rnw, _, _ = ens("syn_phys_b3000_sfimg_s{}_preds.npz")
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


if __name__ == "__main__":
    main()
