#!/usr/bin/env python3
"""Paired, held-out 3 Mpc diagnostics for the regenerated-radio follow-up.

No inferred age is used as a true observed age. Observed comparisons are
reported separately by hypothetical true-age bin, with no fitted age prior.
Bootstrap intervals condition on fitted models and resample parent halos.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

SEEDS = range(42, 47)
AGE_EDGES = [0., .5, 1., 2., 4.01]
IDENTITY = ["sample_group", "sample_snap", "sample_realization", "sample_halo",
            "y_sample", "group_id", "snap", "halo_id", "tsc", "fold_of"]
PROTOCOL = ["train_view", "select_view", "select", "loss", "select_criterion"]


def load_predictions(path):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    if bool(d["provenance_changed_during_run"]):
        raise ValueError(f"Inputs or source changed during {path}")
    for k in ["oof_single", "oof_sample", "oof_cluster", "obs_pred_folds"]:
        if not np.isfinite(d[k]).all():
            raise ValueError(f"Nonfinite {k} in {path}")
    return d


def check_pair(a, b):
    for k in IDENTITY + PROTOCOL + ["seed", "sel_groups_flat", "sel_groups_fold"]:
        if not np.array_equal(a[k], b[k]):
            raise ValueError(f"Predictions do not match in {k}")
    pa, pb = [json.loads(str(d["provenance"])) for d in (a, b)]
    if pa["sha256"] != pb["sha256"]:
        raise ValueError("Training source hashes differ between arms")
    for k in ["epochs", "folds", "batch_size", "huber_delta", "inner_frac"]:
        if pa["config"][k] != pb["config"][k]:
            raise ValueError(f"Training protocol differs in {k}")


def mean_interval(values, groups, rng, nboot=2000):
    """Equal-parent mean, preserving every parent's correlated observations."""
    v = pd.DataFrame({"g": groups, "v": values}).groupby("g").v.mean().to_numpy()
    if not len(v):
        raise ValueError("Empty bootstrap sample")
    boots = v[rng.integers(0, len(v), (nboot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(boots, [.025, .975])
    return dict(mean=float(v.mean()), lo=float(lo), hi=float(hi), parents=len(v))


def conditional_rows(df, rng):
    work = df.copy()
    work["mass_bin"] = pd.qcut(work.log_m500, 3, labels=False, duplicates="drop")
    work["z_bin"] = pd.qcut(work.z, 3, labels=False, duplicates="drop")
    work["age_bin"] = pd.cut(work.true_pseudo_tsc, AGE_EDGES, labels=False,
                              right=False)
    rows = []
    for (m, z, t), sub in work.groupby(["mass_bin", "z_bin", "age_bin"]):
        row = dict(mass_bin=int(m), z_bin=int(z), age_bin=int(t),
                   mass_min=float(sub.log_m500.min()), mass_max=float(sub.log_m500.max()),
                   z_min=float(sub.z.min()), z_max=float(sub.z.max()),
                   age_min=AGE_EDGES[int(t)], age_max=AGE_EDGES[int(t)+1],
                   rows=len(sub), parents=sub.group_id.nunique())
        if row["parents"] >= 8:
            for col in ["radio_residual", "xray_residual", "delta_radio_minus_xray"]:
                stat = mean_interval(sub[col], sub.group_id, rng)
                row.update({f"{col}_{k}": stat[k] for k in ["mean", "lo", "hi"]})
            row["status"] = "estimated"
        else:
            row["status"] = "too_few_parents"
        rows.append(row)
    return rows


def observed_rows(df, radio, xray, rng):
    # Use dataset keys instead of introducing another name-normalisation rule.
    mappings = []
    for path in ["injected_syn_phys_b3000_s9991.h5", "xray_fix_b3000_s9991.h5"]:
        with h5py.File(path) as f:
            decode = lambda a: [s.decode() if isinstance(s, bytes) else str(s) for s in a]
            mappings.append(dict(zip(decode(f["obs/name"][:]), decode(f["obs/key"][:]))))
    names_r, names_x = radio[0]["obs_name"].astype(str), xray[0]["obs_name"].astype(str)
    keys_r = {mappings[0][n]: i for i, n in enumerate(names_r)}
    keys_x = {mappings[1][n]: i for i, n in enumerate(names_x)}
    for models, names in [(radio, names_r), (xray, names_x)]:
        for d in models:
            if not np.array_equal(d["obs_name"].astype(str), names):
                raise ValueError("Observation order changed across seeds")
    rp = np.mean([d["obs_pred_folds"] for d in radio], axis=0)
    xp = np.mean([d["obs_pred_folds"] for d in xray], axis=0)
    masses = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")
    with h5py.File("xray_fix_b3000_s9991.h5") as f:
        exposure = dict(zip(decode(f["obs/key"][:]), f["obs/exposure_ks"][:]))
    observed, matches = [], []
    for key in sorted(keys_r.keys() & keys_x.keys()):
        ir, ix = keys_r[key], keys_x[key]
        z = float(radio[0]["obs_z"][ir])
        if not np.isclose(z, xray[0]["obs_z"][ix], atol=1e-6):
            raise ValueError(f"Observed redshifts do not match for {key}")
        fold_delta = rp[:, ir] - xp[:, ix]
        row = dict(target=key, z=z, radio_prediction=float(rp[:, ir].mean()),
                   xray_prediction=float(xp[:, ix].mean()),
                   delta_radio_minus_xray=float(fold_delta.mean()),
                   exposure_ks=float(exposure[key]))
        if key not in masses.index or not np.isfinite(masses.loc[key, "log_m500c"]):
            row["matching_status"] = "missing_mass"
            observed.append(row)
            continue
        mass = float(masses.loc[key, "log_m500c"])
        row.update(log_m500=mass, matching_status="see_age_sensitivity_table")
        observed.append(row)
        # Fixed, predeclared calipers. Do not widen them in response to outcomes.
        near = ((np.abs(df.log_m500 - mass) <= .2) & (np.abs(df.z - z) <= .015)
                & (np.abs(np.log(df.exposure_ks / exposure[key])) <= np.log(2.)))
        for lo, hi in zip(AGE_EDGES[:-1], AGE_EDGES[1:]):
            sub = df[near & (df.true_pseudo_tsc >= lo) & (df.true_pseudo_tsc < hi)]
            record = dict(target=key, true_age_lo=lo, true_age_hi=hi,
                          parents=sub.group_id.nunique(), rows=len(sub))
            if record["parents"] < 8:
                record["status"] = "too_few_parents"
            else:
                sim = mean_interval(sub.delta_radio_minus_xray, sub.group_id, rng)
                # Each simulated halo uses one held-out fold per seed. Compare
                # the observed target under precisely those same five models.
                discrepancy = fold_delta[sub.fold.to_numpy(int)] - sub.delta_radio_minus_xray
                gap = mean_interval(discrepancy, sub.group_id, rng)
                record.update(status="estimated", simulated_delta=sim["mean"],
                              simulated_delta_lo=sim["lo"], simulated_delta_hi=sim["hi"],
                              observed_minus_simulated=gap["mean"],
                              discrepancy_lo=gap["lo"], discrepancy_hi=gap["hi"])
            matches.append(record)
    return observed, matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--include-joint", action="store_true")
    args = ap.parse_args()
    out = args.run_dir / ("joint_comparison" if args.include_joint else "offset")
    out.mkdir(exist_ok=False)
    verified = json.loads((args.run_dir / "preflight/verified.json").read_text())
    rng = np.random.default_rng(20261103)
    radio, xray, joint, paths, seed_metrics = [], [], [], [], []
    for seed in SEEDS:
        rp = Path(f"cnn_preds/syn_phys_b3000_sfimg_s{seed}_preds.npz")
        xp = args.run_dir / f"preds/xray_b3000_s{seed}_preds.npz"
        r, x = load_predictions(rp), load_predictions(xp)
        check_pair(r, x)
        expected = np.load(args.run_dir / f"preflight/splits_s{seed}.npz")
        for k in expected.files:
            if not np.array_equal(r[k], expected[k]):
                raise ValueError(f"Actual split differs from preflight: {seed}, {k}")
        for d in [r, x]:
            prov = json.loads(str(d["provenance"]))
            path = prov["config"]["dataset"]
            if prov["inputs_sha256"]["dataset"] != verified["sha256"][path]:
                raise ValueError(f"Dataset changed since preflight: {path}")
        if radio:
            for k in IDENTITY:
                if not np.array_equal(r[k], radio[0][k]):
                    raise ValueError(f"Cross-seed mismatch: {k}")
        radio.append(r); xray.append(x); paths.extend([str(rp), str(xp)])
        arms = [("radio", r), ("xray", x)]
        if args.include_joint:
            jp = Path(f"cnn_preds/syn_phys_b3000_joint_s{seed}_preds.npz")
            j = load_predictions(jp)
            check_pair(r, j)
            pj = json.loads(str(j["provenance"]))
            if (pj["inputs_sha256"]["dataset"] != verified["sha256"][verified["radio"]]
                    or pj["inputs_sha256"]["xray_dataset"] != verified["sha256"][verified["xray"]]):
                raise ValueError("Joint model input hashes differ from preflight")
            joint.append(j); paths.append(str(jp)); arms.append(("joint", j))
        for name, d in arms:
            for mode, truth, pred in [("pooled_cluster", d["tsc"], d["oof_cluster"]),
                                     ("single_view", np.repeat(d["y_sample"], 3),
                                      d["oof_single"].reshape(-1))]:
                seed_metrics.append(dict(seed=seed, arm=name, mode=mode,
                                         r2=float(r2_score(truth, pred)),
                                         mae=float(np.mean(np.abs(pred-truth)))))
    first = radio[0]
    with h5py.File(verified["radio"]) as r, h5py.File(verified["xray"]) as x:
        nreal = r["mock/images"].shape[1]
        if not np.array_equal(np.repeat(r["mock/group_id"][:], nreal), first["sample_group"]):
            raise ValueError("Metadata no longer in prediction order")
        mass = np.repeat(r["mock/log_m500"][:], nreal * 3)
        z = np.repeat(r["mock/z"][:].reshape(-1), 3)
        exposure = np.repeat(x["mock/exposure_ks"][:].reshape(-1), 3)
    rpred = np.mean([d["oof_single"] for d in radio], axis=0).reshape(-1)
    xpred = np.mean([d["oof_single"] for d in xray], axis=0).reshape(-1)
    truth = np.repeat(first["y_sample"], 3)
    df = pd.DataFrame(dict(group_id=np.repeat(first["sample_group"], 3),
                           snap=np.repeat(first["sample_snap"], 3),
                           realization=np.repeat(first["sample_realization"], 3),
                           projection=np.tile(np.arange(3), len(first["y_sample"])),
                           fold=np.repeat(first["fold_of"], 3), log_m500=mass, z=z,
                           exposure_ks=exposure, true_pseudo_tsc=truth,
                           radio_prediction=rpred, xray_prediction=xpred,
                           radio_residual=rpred-truth, xray_residual=xpred-truth,
                           delta_radio_minus_xray=rpred-xpred))
    df.to_csv(out / "paired_predictions.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(out / "seed_metrics.csv", index=False)
    pd.DataFrame(conditional_rows(df, rng)).to_csv(out / "conditional_bias.csv", index=False)
    observed, matches = observed_rows(df, radio, xray, rng)
    pd.DataFrame(observed).to_csv(out / "observed_offsets.csv", index=False)
    pd.DataFrame(matches).to_csv(out / "observed_age_sensitivity.csv", index=False)
    summary = dict(inputs=paths, sign="radio minus X-ray, Gyr", seeds=list(SEEDS),
                   simulated_single_view_offset=mean_interval(rpred-xpred, df.group_id, rng),
                   observed_median_offset=float(np.median([r["delta_radio_minus_xray"] for r in observed])),
                   matching_calipers=dict(log10_mass=.2, redshift=.015, exposure_factor=2),
                   limitations=["Real true ages are unknown: inspect all age bins; no age prior fitted.",
                                "Intervals resample parent halos conditional on fitted models.",
                                "Observed headline predictions average 25 models; matched diagnostics use the same five models as each held-out mock.",
                                "Finite covariate bins/calipers are descriptive, not an exact physical calibration."])
    if joint:
        comparisons = []
        for mode, key, yy, groups in [("single_view", "oof_single", truth, df.group_id),
                                      ("pooled_cluster", "oof_cluster", first["tsc"], first["group_id"])]:
            jpred = np.mean([d[key] for d in joint], axis=0).reshape(-1)
            baseline = np.mean([d[key] for d in xray], axis=0).reshape(-1)
            stat = mean_interval(np.abs(baseline-yy)-np.abs(jpred-yy), groups, rng)
            comparisons.append(dict(mode=mode, xray_r2=float(r2_score(yy, baseline)),
                                    joint_r2=float(r2_score(yy, jpred)),
                                    mae_gain_positive_favors_joint=stat))
        summary["joint_comparison"] = comparisons
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, col, label in zip(axes, ["true_pseudo_tsc", "log_m500", "z"],
                              ["True pseudo-TSC [Gyr]", "log10 M500 [solar masses]", "Observing redshift"]):
        ax.scatter(df[col], df.delta_radio_minus_xray, s=2, alpha=.08, rasterized=True)
        ax.axhline(0, color="black", lw=.8)
        ax.set_xlabel(label)
    axes[0].set_ylabel("Radio minus X-ray prediction [Gyr]")
    fig.suptitle("Held-out single-view predictions; mean of five seeds")
    fig.savefig(out / "offset_diagnostic.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
