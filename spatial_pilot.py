#!/usr/bin/env python3
"""Matched simulation pilot: do radio positions relative to X-rays help?

See SPATIAL_PILOT.md for the fixed experiment and coordinate conventions.
Full extraction, training and bootstrap scoring belong on a compute node.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
from scipy import ndimage
from sklearn.metrics import mean_absolute_error, r2_score

ARMS = ("masked", "origin", "registered")
COLS = ("radio_detected", "xray_peak_valid", "centroid_distance_kpc",
        "mean_radio_radius_kpc")
RADIO_FILE = "injected_fix_phys_b3000_s9991.h5"
XRAY_FILE = "xray_fix_b3000_s9991.h5"
MANIFEST_FILE = "folds_e0.npz"
SOURCE_FILES = ("spatial_pilot.py", "test_spatial_pilot.py", "train_cnn_mock.py",
                "train_cnn_pooled.py", "forward_model_lotss.py",
                "minkowski_functionals.py", "build_xray_realistic.py",
                "SPATIAL_PILOT.md", "submit_spatial_pilot.sh")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_order(r, x):
    """Join by parent AND snapshot, then verify label and realization identity."""
    keys = lambda f: list(zip(f["mock/group_id"][:].tolist(),
                              f["mock/snap"][:].tolist()))
    rk, xk = keys(r), keys(x)
    if len(set(rk)) != len(rk) or len(set(xk)) != len(xk):
        raise ValueError("duplicate (parent, snapshot) identities")
    if set(rk) != set(xk):
        raise ValueError("radio and X-ray cohorts differ")
    lookup = {k: i for i, k in enumerate(xk)}
    order = np.array([lookup[k] for k in rk])
    for key in ("halo_id", "pseudo_tsc", "z"):
        a, b = r["mock/" + key][:], x["mock/" + key][:][order]
        if not np.array_equal(a, b):
            raise ValueError(f"paired {key} differ (including observing realizations)")
    if r["mock/images"].shape[:3] != x["mock/images"].shape[:3]:
        raise ValueError("different realization/projection dimensions")
    if r["mock/images"].shape[2:] != (3, 128, 128):
        raise ValueError("expected xy,yz,xz projections on 128-pixel grids")
    if x["mock/images"].shape[2:] != (3, 128, 128):
        raise ValueError("unexpected X-ray grid")
    if float(r.attrs["box_kpc"]) != 3000 or r.attrs["recipe"] != "physical":
        raise ValueError("pilot requires the physical 3 Mpc radio inputs")
    if x.attrs["count_model"] != "real":
        raise ValueError("pilot requires the adopted realistic X-ray count model")
    return order


def partitions(groups, manifest):
    ids, folds = manifest["group_id"], manifest["fold"]
    inner = manifest["inner_sel"]
    nfold = int(manifest["folds"])
    if len(np.unique(ids)) != len(ids) or inner.shape != (nfold, len(ids)):
        raise ValueError("invalid manifest dimensions or duplicate parents")
    if not np.isin(folds, np.arange(nfold)).all():
        raise ValueError("invalid outer-fold identifiers")
    where = {int(g): i for i, g in enumerate(ids)}
    if not set(groups.tolist()) <= set(where):
        raise ValueError("parent missing from fixed manifest")
    ix = np.array([where[int(g)] for g in groups])
    outer = folds[ix]
    out = []
    for k in range(nfold):
        if np.any(inner[k] & (folds == k)):
            raise ValueError("manifest selects an outer-test parent")
        fit = np.flatnonzero((outer != k) & ~inner[k, ix])
        sel = np.flatnonzero((outer != k) & inner[k, ix])
        test = np.flatnonzero(outer == k)
        if min(len(fit), len(sel), len(test)) == 0:
            raise ValueError("empty fitting/selection/test partition")
        sets = [set(groups[a].tolist()) for a in (fit, sel, test)]
        if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("parent-halo leakage")
        out.append((fit, sel, test))
    return out, outer


def xray_to_radio_position(row_col_kpc, projection):
    """Convert SOXS row/column coordinates to the radio histogram convention.

    Radio histogram2d axes: rows/cols = (x,y), (y,z), (x,z).
    pyXSIM LOS z,x,y -> screen horizontal/vertical = (x,y),(y,z),(z,x).
    SOXS FITS arrays are vertical/horizontal; hence transpose views 0 and 1.
    RA decreases with horizontal screen position; SOXS CDELT1 is negative,
    so there is no additional reflection. The transform is label-independent.
    """
    if projection not in (0, 1, 2):
        raise ValueError("unknown projection")
    return np.asarray(row_col_kpc)[::-1] if projection < 2 else np.asarray(row_col_kpc)


def moments(weights, pixel_kpc, centre_row_col):
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(2, np.nan)
    yy, xx = np.indices(weights.shape, dtype=float)
    yy = (yy + 0.5 - weights.shape[0] / 2) * pixel_kpc
    xx = (xx + 0.5 - weights.shape[1] / 2) * pixel_kpc
    cy, cx = np.asarray(centre_row_col)
    ry, rx = np.sum(weights * yy) / total, np.sum(weights * xx) / total
    return np.array([np.hypot(ry - cy, rx - cx),
                     np.sum(weights * np.hypot(yy - cy, xx - cx)) / total])


def radio_weights(image):
    """Positive excess above 3*rms in 8-connected islands of >=4 pixels."""
    if not np.isfinite(image).all() or np.max(np.abs(image)) > 80:
        raise ValueError("invalid stored radio stretch")
    snr = np.sinh(image.astype(np.float64))
    islands, count = ndimage.label(snr > 3.0, structure=np.ones((3, 3)))
    sizes = np.bincount(islands.ravel(), minlength=count + 1)
    keep = sizes >= 4
    keep[0] = False
    return np.where(keep[islands], np.maximum(snr - 3.0, 0.0), 0.0)


def peak_position(image, pixel_kpc, search_kpc, aperture_kpc):
    """Smoothed rate peak; basic contrast flag, not a calibrated detection test."""
    if not np.isfinite(image).all() or np.max(np.abs(image)) > 80:
        raise ValueError("invalid stored X-ray stretch")
    yy, xx = np.indices(image.shape, dtype=float)
    yy = (yy + 0.5 - image.shape[0] / 2) * pixel_kpc
    xx = (xx + 0.5 - image.shape[1] / 2) * pixel_kpc
    rr = np.hypot(yy, xx)
    # Multiplying by the common stretch scale would not change the peak,
    # the contrast flag, or any coordinate, so work in rate/scale units.
    smooth = ndimage.gaussian_filter(np.sinh(image.astype(np.float64)), 2.0,
                                     mode="constant")
    search = rr <= min(search_kpc, 0.8 * aperture_kpc)
    annulus = (rr >= 0.6 * aperture_kpc) & (rr <= 0.9 * aperture_kpc)
    if not search.any() or annulus.sum() < 10:
        return np.full(2, np.nan), False
    bg = np.median(smooth[annulus])
    scatter = 1.4826 * np.median(np.abs(smooth[annulus] - bg))
    pos = np.unravel_index(np.argmax(np.where(search, smooth, -np.inf)), image.shape)
    valid = smooth[pos] > max(0.0, bg + 3.0 * scatter)
    return np.array([yy[pos], xx[pos]]), bool(valid)


def feature_pair(radio, xray, projection, radio_pixel, xray_pixel, search, aperture):
    weights = radio_weights(radio)
    peak, good_x = peak_position(xray, xray_pixel, search, aperture)
    peak = xray_to_radio_position(peak, projection)
    good_r = bool(weights.sum() > 0)
    origin = np.r_[float(good_r), float(good_x), moments(weights, radio_pixel, (0, 0))]
    registered = np.r_[float(good_r), float(good_x),
                       moments(weights, radio_pixel, peak) if good_x
                       else np.full(2, np.nan)]
    # Keep exactly the same usable cohort/availability in both geometry arms.
    if not (good_r and good_x):
        origin[2:] = np.nan
        registered[2:] = np.nan
    return origin, registered, peak


def preflight(root):
    with h5py.File(root / RADIO_FILE, "r") as r, h5py.File(root / XRAY_FILE, "r") as x:
        order = paired_order(r, x)
        groups = np.repeat(r["mock/group_id"][:], r["mock/images"].shape[1])
        with np.load(root / MANIFEST_FILE) as mf:
            splits, _ = partitions(groups, mf)
        print(f"Paired {len(order)} halo-snapshots; {len(np.unique(groups))} parents; "
              f"{len(splits)} fixed outer/inner partitions.", flush=True)


def extract(radio, xray, z, radio_pixel, out, identities):
    from astropy.cosmology import Planck15
    from build_xray_realistic import aperture_blocks
    xray_pixel = float(38 * 0.492 / 206265.0 *
                       Planck15.angular_diameter_distance(0.05).to_value("kpc"))
    origin = np.empty((*radio.shape[:3], 4), np.float32)
    registered = np.empty_like(origin)
    peaks = np.empty((*radio.shape[:3], 2), np.float32)
    conditions = {}
    for redshift in np.unique(z):
        scale = float(Planck15.angular_diameter_distance(float(redshift)).to_value("kpc")
                      / 206265.0)
        conditions[float(redshift)] = (120.0 * scale,
                                      aperture_blocks(float(redshift)) * xray_pixel)
    for i in range(len(radio)):
        for j in range(radio.shape[1]):
            search, aperture = conditions[float(z[i, j])]
            for p in range(3):
                origin[i, j, p], registered[i, j, p], peaks[i, j, p] = feature_pair(
                    radio[i, j, p], xray[i, j, p], p, radio_pixel, xray_pixel,
                    search, aperture)
        if (i + 1) % 100 == 0:
            print(f"geometry: {i + 1}/{len(radio)} halo-snapshots", flush=True)
    np.savez_compressed(out / "features.npz", origin=origin, registered=registered,
                        xray_peak_radio_axes_kpc=peaks, columns=np.array(COLS),
                        radio_pixel_kpc=radio_pixel, xray_pixel_kpc=xray_pixel,
                        observing_redshift=z, **identities)
    qa = {
        "observations": int(np.prod(origin.shape[:3])),
        "radio_detection_fraction": float(origin[..., 0].mean()),
        "xray_valid_fraction": float(origin[..., 1].mean()),
        "usable_geometry_fraction": float(np.isfinite(registered[..., 2]).mean()),
        "peak_distance_percentiles_kpc": np.percentile(
            np.linalg.norm(peaks, axis=-1), [16, 50, 84]).tolist(),
    }
    (out / "feature_quality.json").write_text(json.dumps(qa, indent=2) + "\n")
    print("Feature quality: " + json.dumps(qa), flush=True)
    if qa["usable_geometry_fraction"] < 0.05:
        raise RuntimeError("fewer than 5% of observations have usable geometry; stop pilot")
    return origin, registered


def score(out, seeds, draws=4000):
    runs = {a: [dict(np.load(out / f"{a}_s{s}.npz", allow_pickle=False))
                for s in seeds] for a in ARMS}
    ref = runs["masked"][0]
    for arm in ARMS:
        for seed, d in zip(seeds, runs[arm]):
            if str(d["arm"]) != arm or int(d["seed"]) != seed:
                raise ValueError("arm/seed identity mismatch")
            for key in ("sample_group", "sample_snap", "sample_halo", "sample_realization",
                        "y", "fold_of", "inner_groups", "inner_folds", "provenance_sha256",
                        "image_preprocessing"):
                if not np.array_equal(ref[key], d[key]):
                    raise ValueError(f"{arm}/{seed}: mismatched {key}")
            if not np.isfinite(d["prediction"]).all():
                raise ValueError("incomplete or non-finite predictions")
    keep = ref["sample_snap"] == 99
    y = np.repeat(ref["y"][keep, None], 3, axis=1).ravel()
    groups = np.repeat(ref["sample_group"][keep, None], 3, axis=1).ravel()
    per_seed = {a: [d["prediction"][keep].ravel() for d in runs[a]] for a in ARMS}
    ensemble = {a: np.mean(p, axis=0) for a, p in per_seed.items()}
    metrics = lambda pred: {"r2": float(r2_score(y, pred)),
                             "mae_gyr": float(mean_absolute_error(y, pred))}
    report = {"seeds": list(seeds), "evaluation": "snapshot 99, one view/realization",
              "scores": {a: metrics(p) for a, p in ensemble.items()}, "contrasts": {}}
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    ys = np.bincount(inverse, weights=y)
    y2s = np.bincount(inverse, weights=y * y)
    sse = {a: np.bincount(inverse, weights=(p - y)**2) for a, p in ensemble.items()}
    sae = {a: np.bincount(inverse, weights=np.abs(p - y)) for a, p in ensemble.items()}
    rng = np.random.default_rng(0)
    weights = np.array([np.bincount(rng.integers(len(unique), size=len(unique)),
                                   minlength=len(unique)) for _ in range(draws)])
    n = weights @ counts
    tss = weights @ y2s - (weights @ ys)**2 / n
    valid_draw = tss > 0
    if not valid_draw.any():
        raise ValueError("bootstrap labels have no variation")
    for new, base in (("registered", "masked"), ("origin", "masked"),
                      ("registered", "origin")):
        dr = (weights[valid_draw] @ (sse[base] - sse[new])) / tss[valid_draw]
        dm = (weights[valid_draw] @ (sae[new] - sae[base])) / n[valid_draw]
        point_r = report["scores"][new]["r2"] - report["scores"][base]["r2"]
        point_m = report["scores"][new]["mae_gyr"] - report["scores"][base]["mae_gyr"]
        seed_delta = [r2_score(y, b) - r2_score(y, a)
                      for a, b in zip(per_seed[base], per_seed[new])]
        report["contrasts"][new + "_minus_" + base] = {
            "delta_r2": point_r, "delta_mae_gyr": point_m,
            "r2_conditional_95ci": np.percentile(dr, [2.5, 97.5]).tolist(),
            "mae_conditional_95ci": np.percentile(dm, [2.5, 97.5]).tolist(),
            "paired_seed_delta_r2": seed_delta,
            "practical_followup": bool(point_r >= 0.02 and point_m <= 0.02),
        }
    report["bootstrap"] = {"draws": draws, "unit": "parent halo",
                           "valid_draws": int(valid_draw.sum()),
                           "conditional_on_fitted_runs": True}
    text = ["Spatial pilot: held-out simulated observations", "",
            f"{len(seeds)}-seed ensembles; {len(unique)} parent halos; {len(y)} observations.",
            "The masked arm retains quality flags; all arms share one architecture.", ""]
    for arm, m in report["scores"].items():
        text.append(f"{arm}: R²={m['r2']:.4f}; MAE={m['mae_gyr']:.4f} Gyr")
    text.append("")
    for name, d in report["contrasts"].items():
        text.append(f"{name}: ΔR²={d['delta_r2']:+.4f}, "
                    f"95% conditional interval {d['r2_conditional_95ci']}; "
                    f"ΔMAE={d['delta_mae_gyr']:+.4f} Gyr.")
    text.extend(["", "Intervals resample parent halos and condition on these fitted runs.",
                 "This is a simulation pilot, not an observational timing calibration.",
                 "Registered minus origin tests the chosen X-ray reference, including "
                 "its measurement errors; it does not exhaust all spatial information."])
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "results.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text), flush=True)
    return report


def run(root, out, work, seeds, epochs, batch_size):
    import torch
    from train_cnn_mock import prep_view_scalars, predict, run_fold
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("full pilot must run in a Slurm GPU allocation")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    out.mkdir(parents=True, exist_ok=True)
    if (out / "provenance.json").exists():
        raise RuntimeError("run directory already contains a pilot; use a fresh directory")
    preflight(root)
    work.mkdir(parents=True, exist_ok=True)
    local = {}
    input_sha = {}
    for name in (RADIO_FILE, XRAY_FILE, MANIFEST_FILE):
        before = (root / name).stat()
        local[name] = work / name
        shutil.copyfile(root / name, local[name])
        after = (root / name).stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"input changed during local staging: {name}")
        input_sha[name] = sha256(local[name])
    shutil.copyfile(local[MANIFEST_FILE], out / "fold_manifest.npz")
    code = Path(__file__).resolve().parent
    provenance = dict(inputs_sha256=input_sha,
                      code_sha256={name: sha256(code / name) for name in SOURCE_FILES},
                      seeds=seeds, epochs=epochs, batch_size=batch_size,
                      job_id=os.environ["SLURM_JOB_ID"],
                      train_view="pooled", select_view="single",
                      feature_columns=COLS, arms=ARMS,
                      image_preprocessing="stored arcsinh; X-ray xy,yz transposed; "
                                          "no extra standardization",
                      radio_threshold_sigma=3.0, min_radio_island_pixels=4,
                      xray_smoothing_sigma_pixels=2.0,
                      coordinate_transform="X-ray transpose for xy,yz; identity for xz",
                      torch_version=torch.__version__, numpy_version=np.__version__)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    prov_hash = sha256(out / "provenance.json")
    with h5py.File(local[RADIO_FILE], "r") as r, h5py.File(local[XRAY_FILE], "r") as x:
        order = paired_order(r, x)
        radio = r["mock/images"][:]
        xray = x["mock/images"][:][order]
        row_group = r["mock/group_id"][:]
        row_snap = r["mock/snap"][:]
        row_halo = r["mock/halo_id"][:]
        label = r["mock/pseudo_tsc"][:]
        z = r["mock/z"][:]
        radio_pixel = float(r.attrs["box_kpc"]) / radio.shape[-1]
    if not np.isfinite(radio).all() or not np.isfinite(xray).all():
        raise ValueError("non-finite images")
    origin, registered = extract(radio, xray, z, radio_pixel, out,
                                 dict(group_id=row_group, snap=row_snap, halo_id=row_halo))
    # Paired augmentation must rotate both images in the same physical axes.
    # Feature extraction above converted the raw FITS peak coordinates.
    xray[:, :, :2] = xray[:, :, :2].swapaxes(-1, -2).copy()
    N, R, P, H, W = radio.shape
    radio = radio.reshape(N * R, P, H, W)
    xray = xray.reshape(N * R, P, H, W)
    groups, snap = np.repeat(row_group, R), np.repeat(row_snap, R)
    y = np.repeat(label, R)
    raw_features = {"masked": origin.reshape(N * R, P, 4),
                    "origin": origin.reshape(N * R, P, 4),
                    "registered": registered.reshape(N * R, P, 4)}
    with np.load(local[MANIFEST_FILE]) as mf:
        splits, fold_of = partitions(groups, mf)
    inner_groups = np.concatenate([np.unique(groups[s]) for _, s, _ in splits])
    inner_folds = np.concatenate([np.full(len(np.unique(groups[s])), k)
                                  for k, (_, s, _) in enumerate(splits)])
    prep = {}
    for arm in ARMS:
        mask = np.array([False, False, arm == "masked", arm == "masked"])
        prep[arm] = [prep_view_scalars(raw_features[arm], fit, mask, COLS)
                     for fit, _, _ in splits]
    for seed in seeds:
        for arm in ARMS:
            prediction = np.full((N * R, P), np.nan, np.float32)
            epochs_best, select_scores = [], []
            for k, (fit, sel, test) in enumerate(splits):
                torch.manual_seed(seed + k)
                np.random.seed(seed + k)
                vs, _, _, _ = prep[arm][k]
                print(f"\n{arm} seed {seed}, fold {k + 1}/{len(splits)}: "
                      f"fit {len(fit)}, selection {len(sel)}, test {len(test)}", flush=True)
                model, best, ep = run_fold(
                    radio[fit], y[fit], radio[sel], y[sel], epochs, batch_size, "cuda", 0.5,
                    tr_xr=xray[fit], sel_xr=xray[sel], train_view="pooled",
                    select_view="single", tr_vs=vs[fit], sel_vs=vs[sel], n_view=4)
                for p in range(P):
                    prediction[test, p] = predict(
                        model, radio[test, p:p + 1], "cuda", batch_size=batch_size,
                        xray=xray[test, p:p + 1], view_scalars=vs[test, p:p + 1])
                epochs_best.append(ep)
                select_scores.append(best)
                del model
                torch.cuda.empty_cache()
                # Checkpoint predictions only; incomplete runs cannot be scored.
                np.savez_compressed(
                    out / f"{arm}_s{seed}.npz", prediction=prediction, y=y,
                    sample_group=groups, sample_snap=snap, sample_halo=np.repeat(row_halo, R),
                    sample_realization=np.tile(np.arange(R), N), fold_of=fold_of,
                    inner_groups=inner_groups, inner_folds=inner_folds,
                    arm=arm, seed=seed, best_epoch=np.array(epochs_best),
                    inner_selection_r2=np.array(select_scores),
                    view_median=np.array([p_[1] for p_ in prep[arm]]),
                    view_mu=np.array([p_[2] for p_ in prep[arm]]),
                    view_sd=np.array([p_[3] for p_ in prep[arm]]),
                    provenance_sha256=prov_hash,
                    image_preprocessing=provenance["image_preprocessing"])
                print(f"Saved {arm}/seed {seed}: completed {k + 1} folds", flush=True)
    for name in SOURCE_FILES:
        if sha256(code / name) != provenance["code_sha256"][name]:
            raise RuntimeError("frozen source changed during the pilot")
    score(out, seeds)
    (out / "COMPLETE").write_text("Training and matched scoring completed.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("preflight", "run", "score"))
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--out", type=Path)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be unique and nonempty")
    if a.stage == "preflight":
        preflight(a.root)
    elif a.stage == "score":
        score(a.out, seeds)
    else:
        if a.out is None or a.work is None:
            ap.error("run requires --out and --work")
        run(a.root, a.out, a.work, seeds, a.epochs, a.batch_size)


if __name__ == "__main__":
    main()
