#!/usr/bin/env python3
"""
XGBoost baseline: predict merger label score from tabular morphology features.

Features come from feats_labels_dict_tngcluster.pkl (pre-computed per projection):
  bic_1, bic_2, mean_r_0, mean_r_1, std_r, mean_v_0, mean_v_1,
  std_r_0, std_r_1, std_v_0, std_v_1, std_v, n0, n1, elongation_ratio

Each (cluster × projection) is one sample → 352 × 3 = 1056 rows.
CV is stratified at the cluster level to prevent leakage between projections.

Usage:
    python train_xgboost.py [--tau 1.0] [--folds 5] [--seed 42]
"""

import argparse
import pickle

import h5py
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import xgboost as xgb

SNAP = 99
PROJ_NAMES = ["xy", "yz", "xz"]
FEATURE_KEYS = [
    "bic_1", "bic_2",
    "mean_r_0", "mean_r_1", "std_r",
    "mean_v_0", "mean_v_1",
    "std_r_0", "std_r_1",
    "std_v_0", "std_v_1", "std_v",
    "n0", "n1",
    "elongation_ratio",
]


def build_tabular(pkl_path, dataset_path, tau):
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)

    # get tau index from dataset
    with h5py.File(dataset_path, "r") as f:
        tau_vals = f["labels/tau_gyr"][:]
        halo_ids_ordered = f["meta/halo_id"][:]
        r500c_kpc = f["meta/r500c_kpc"][:]
        mass_ratio = f["meta/mass_ratio"][:]

    tau_idx = int(np.argmin(np.abs(tau_vals - tau)))
    actual_tau = float(tau_vals[tau_idx])
    print(f"Using tau = {actual_tau:.1f} Gyr (index {tau_idx})")

    rows, labels, groups = [], [], []

    for i, halo_id in enumerate(halo_ids_ordered):
        entry = pkl[halo_id][SNAP]
        label = float(entry[f"label_score_all_tau{actual_tau:.1f}"])

        for proj in PROJ_NAMES:
            feats = entry["features"][proj]
            row = [float(feats[k]) for k in FEATURE_KEYS]
            # append cluster-level features
            row += [float(r500c_kpc[i]), float(mass_ratio[i])]
            rows.append(row)
            labels.append(label)
            groups.append(i)   # cluster index — used to keep projections together in CV

    feat_names = FEATURE_KEYS + ["r500c_kpc", "mass_ratio"]
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    groups = np.array(groups)
    return X, y, groups, feat_names


def run_cv(X, y, groups, n_folds, seed):
    gkf = GroupKFold(n_splits=n_folds)

    fold_r2, fold_mae, fold_rmse = [], [], []
    oof_preds = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            verbosity=0,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        r2   = r2_score(y_val, preds)
        mae  = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)
        fold_r2.append(r2)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        print(f"  Fold {fold+1}: R²={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")

    print()
    print(f"CV mean ± std:")
    print(f"  R²  : {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")
    print(f"  MAE : {np.mean(fold_mae):.3f} ± {np.std(fold_mae):.3f}")
    print(f"  RMSE: {np.mean(fold_rmse):.3f} ± {np.std(fold_rmse):.3f}")

    oof_r2 = r2_score(y, oof_preds)
    print(f"  OOF R²: {oof_r2:.3f}")
    return oof_preds


def main(tau, n_folds, seed):
    pkl_path     = "feats_labels_dict_tngcluster.pkl"
    dataset_path = "dataset.h5"

    print("Building feature matrix...")
    X, y, groups, feat_names = build_tabular(pkl_path, dataset_path, tau)
    print(f"X: {X.shape}  (samples × features)")
    print(f"y: min={y.min():.3f}  max={y.max():.3f}  mean={y.mean():.3f}")
    print()

    print(f"Running {n_folds}-fold GroupKFold CV (grouped by cluster)...")
    run_cv(X, y, groups, n_folds, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tau",   type=float, default=1.0,
                        help="Time window tau in Gyr for label_score (default: 1.0)")
    parser.add_argument("--folds", type=int,   default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()
    main(args.tau, args.folds, args.seed)
