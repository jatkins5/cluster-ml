#!/usr/bin/env python3
"""
Shallow CNN baseline: predict merger label score from 2D radio emission images.

Input: dataset.h5 images (352 clusters × 3 projections, 128×128, arcsinh-normalised).
Each projection is treated as an independent sample (1056 total).
Augmentation: 8 transforms (4 rotations × 2 flips) applied on-the-fly during training.
CV is grouped at the cluster level to prevent leakage.

Architecture: 4 conv blocks → global average pool → 2-layer MLP head.

Usage:
    python train_cnn.py [--tau 1.0] [--folds 5] [--epochs 60] [--batch-size 32]
"""

import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


# ---------- augmentation ----------

def augment(img):
    """Return one of 8 deterministic transforms: 4 rotations × 2 flips."""
    k = np.random.randint(8)
    if k >= 4:
        img = np.fliplr(img)
    img = np.rot90(img, k % 4)
    return np.ascontiguousarray(img)


# ---------- dataset ----------

class RadioDataset(Dataset):
    def __init__(self, images, labels, train=False, weights=None):
        # images: (N, H, W) float32
        self.images = images
        self.labels = labels
        self.train  = train
        self.weights = (np.ones(len(images), dtype=np.float32)
                        if weights is None else weights.astype(np.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()   # (H, W)
        if self.train:
            img = augment(img)
        img = torch.tensor(img[None], dtype=torch.float32)  # (1, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        return img, label, weight


# ---------- model ----------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ShallowCNN(nn.Module):
    def __init__(self, img_size=128):
        super().__init__()
        blocks = [
            ConvBlock(1,  32),   # /2
            ConvBlock(32, 64),   # /4
            ConvBlock(64, 128),  # /8
            ConvBlock(128, 256), # /16
        ]
        if img_size >= 512:
            blocks.append(ConvBlock(256, 256))  # /32
            blocks.append(ConvBlock(256, 256))  # /64
        elif img_size >= 256:
            blocks.append(ConvBlock(256, 256))  # /32
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.pool(self.encoder(x))).squeeze(1)


# ---------- training ----------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels, weights in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        weights = weights.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = (criterion(preds, labels) * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, _ in loader:
        preds = model(imgs.to(device)).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return preds, labels


def run_cv(images, labels, groups, n_folds, n_epochs, batch_size, seed, huber_delta=0.5, img_size=128,
           weights=None, weight_loss=True, select="inner", inner_frac=0.15):
    """`select` decides where the reported checkpoint comes from.

    "inner" holds out a further `inner_frac` of each fold's training
    *clusters* (grouped, so the 3 projections of a cluster stay together)
    and picks the epoch on that; the outer fold is predicted once, by a
    checkpoint chosen without seeing it. "final" takes the last epoch.

    Selecting the epoch on the outer fold and reporting that same number --
    what this did until 2026-09-09 -- inflated fold R2 by a measured +0.043
    across the 60 baseline folds. It is not available any more.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   checkpoint selection: {select}")

    weighted = weights is not None
    if not weighted:
        weights = np.ones(len(labels), dtype=np.float32)
    # Reporting the weighted score for an unweighted model is the ablation
    # that decides whether the weights earn their keep: without it, a weighted
    # model scoring well on the weighted metric proves nothing.
    train_w = weights if weight_loss else np.ones_like(weights)

    gkf = GroupKFold(n_splits=n_folds)
    fold_r2, fold_mae, fold_rmse = [], [], []
    oof_preds = np.zeros_like(labels)

    for fold, (outer_train_idx, val_idx) in enumerate(gkf.split(images, labels, groups)):
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)

        if select == "inner":
            gss = GroupShuffleSplit(n_splits=1, test_size=inner_frac,
                                    random_state=seed + fold)
            rel_fit, rel_sel = next(gss.split(outer_train_idx,
                                              groups=groups[outer_train_idx]))
            train_idx = outer_train_idx[rel_fit]
            sel_idx = outer_train_idx[rel_sel]
        else:
            train_idx, sel_idx = outer_train_idx, None

        # per-fold normalisation using training set stats only
        tr_mean = images[train_idx].mean()
        tr_std  = images[train_idx].std() + 1e-8
        imgs_tr = (images[train_idx] - tr_mean) / tr_std
        imgs_val = (images[val_idx]  - tr_mean) / tr_std

        train_ds = RadioDataset(imgs_tr, labels[train_idx], train=True,
                                weights=train_w[train_idx])
        val_ds   = RadioDataset(imgs_val, labels[val_idx],  train=False)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        sel_dl = None
        if sel_idx is not None:
            sel_ds = RadioDataset((images[sel_idx] - tr_mean) / tr_std,
                                  labels[sel_idx], train=False)
            sel_dl = DataLoader(sel_ds, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)

        model     = ShallowCNN(img_size=img_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.HuberLoss(delta=huber_delta, reduction="none")

        # Selection follows the same distribution the loss targets: with
        # importance weights on, an unweighted early-stopping criterion would
        # pull the model back toward the sim distribution it is meant to leave.
        w_sel = train_w[sel_idx] if sel_idx is not None else None

        best_sel, best_state, best_epoch = np.inf, None, n_epochs
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
            scheduler.step()
            if sel_dl is not None:
                sp, st = evaluate(model, sel_dl, device)
                sel_rmse = root_mean_squared_error(st, sp, sample_weight=w_sel)
                if sel_rmse < best_sel:
                    best_sel, best_epoch = sel_rmse, epoch + 1
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
            if (epoch + 1) % 30 == 0:
                train_preds, train_true = evaluate(model, train_dl, device)
                train_r2   = r2_score(train_true, train_preds)
                train_rmse = root_mean_squared_error(train_true, train_preds)
                preds, true = evaluate(model, val_dl, device)
                msg = (f"  Fold {fold+1}  Epoch {epoch+1:3d}/{n_epochs}  "
                       f"loss={train_loss:.4f}  "
                       f"train R²={train_r2:.3f}  RMSE={train_rmse:.3f}  |  "
                       f"val R²={r2_score(true, preds):.3f}  "
                       f"RMSE={root_mean_squared_error(true, preds):.3f}")
                if sel_dl is not None:
                    msg += f"  |  sel RMSE={sel_rmse:.3f}"
                print(msg)

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"  Fold {fold+1} checkpoint from epoch {best_epoch}/{n_epochs} "
                  f"(inner-val RMSE {best_sel:.3f})")
        best_preds, _ = evaluate(model, val_dl, device)

        oof_preds[val_idx] = best_preds
        r2   = r2_score(labels[val_idx], best_preds)
        mae  = mean_absolute_error(labels[val_idx], best_preds)
        rmse = root_mean_squared_error(labels[val_idx], best_preds)
        fold_r2.append(r2)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        print(f"  Fold {fold+1} best → R²={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}\n")

    print("CV mean ± std:")
    print(f"  R²  : {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")
    print(f"  MAE : {np.mean(fold_mae):.3f} ± {np.std(fold_mae):.3f}")
    print(f"  RMSE: {np.mean(fold_rmse):.3f} ± {np.std(fold_rmse):.3f}")
    print(f"  OOF R²: {r2_score(labels, oof_preds):.3f}")
    if weighted:
        print(f"  OOF R² (weighted, obs-like sample): "
              f"{r2_score(labels, oof_preds, sample_weight=weights):.3f}")
    return oof_preds


def main(tau, n_folds, n_epochs, batch_size, seed, pseudo_tsc=False,
         merger_tsc=False, huber_delta=0.5, tsc_max=None, dataset_path="dataset.h5",
         sample_weights=None, weights_eval_only=False, select="inner",
         inner_frac=0.15):


    print("Loading dataset...")
    with h5py.File(dataset_path, "r") as f:
        raw_images = f["images"][:]              # (352, 3, H, W)
        halo_ids = f["meta/halo_id"][:]

        if merger_tsc:
            tsc_path = "TSC_Cutimages/TSC_eachhalo_snap99.hdf5"
            with h5py.File(tsc_path, "r") as ft:
                tsc_hids = ft["halo_id"][:]
                tsc_vals = ft["tsc_gyr"][:]
            tsc_map = dict(zip(tsc_hids, tsc_vals))
            all_labels = np.array([tsc_map[h] for h in halo_ids], dtype=np.float32)
            valid = ~np.isnan(all_labels)
            raw_images = raw_images[valid]
            all_labels = all_labels[valid]
            halo_ids = halo_ids[valid]
            print(f"Using merger-catalog TSC label ({valid.sum()}/{len(valid)} clusters, "
                  f"{(~valid).sum()} dropped — no recorded collision)")
            if tsc_max is not None:
                keep = all_labels <= tsc_max
                raw_images = raw_images[keep]
                all_labels = all_labels[keep]
                halo_ids = halo_ids[keep]
                print(f"Filtered to TSC <= {tsc_max} Gyr: {keep.sum()} clusters")
        elif pseudo_tsc:
            all_labels = f["labels/pseudo_tsc"][:]
            print("Using pseudo-TSC label")
        else:
            tau_vals  = f["labels/tau_gyr"][:]
            tau_idx   = int(np.argmin(np.abs(tau_vals - tau)))
            actual_tau = float(tau_vals[tau_idx])
            all_labels = f["labels/label_score_all"][:, tau_idx]
            print(f"Using tau = {actual_tau:.1f} Gyr (index {tau_idx})")

    N, P, H, W = raw_images.shape

    # flatten projections into separate samples; track cluster group
    images = raw_images.reshape(N * P, H, W)         # (N*3, H, W)
    labels = np.repeat(all_labels, P).astype(np.float32)  # (N*3,)
    groups = np.repeat(np.arange(N), P)              # cluster index per sample

    print(f"Samples: {len(images)}  (clusters={N}, projections={P})")
    print(f"Labels:  min={labels.min():.3f}  max={labels.max():.3f}  mean={labels.mean():.3f}")

    sw = None
    if sample_weights is not None:
        wz = np.load(sample_weights)
        wmap = dict(zip(wz["halo_id"], wz["weight"]))
        missing = [h for h in halo_ids if h not in wmap]
        if missing:
            raise SystemExit(f"{len(missing)} halos have no weight in "
                             f"{sample_weights}; refusing to guess")
        w_cluster = np.array([wmap[h] for h in halo_ids], dtype=np.float32)
        w_cluster *= len(w_cluster) / w_cluster.sum()
        sw = np.repeat(w_cluster, P)
        eff_n = w_cluster.sum() ** 2 / (w_cluster ** 2).sum()
        print(f"Weights: {sample_weights}  min={w_cluster.min():.3f}  "
              f"max={w_cluster.max():.3f}  effective N={eff_n:.1f}/{len(w_cluster)}")
    print()

    print(f"Huber delta: {huber_delta}  Image size: {W}×{W}")
    run_cv(images, labels, groups, n_folds, n_epochs, batch_size, seed, huber_delta,
           img_size=W, weights=sw, weight_loss=not weights_eval_only,
           select=select, inner_frac=inner_frac)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tau",        type=float, default=1.0)
    parser.add_argument("--folds",      type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--pseudo-tsc", action="store_true",
                        help="Use pseudo-TSC label instead of fixed-tau score")
    parser.add_argument("--merger-tsc", action="store_true",
                        help="Use ground-truth TSC from merger catalog (TSC_Cutimages/)")
    parser.add_argument("--huber-delta", type=float, default=0.5,
                        help="Delta for Huber loss (default: 0.5)")
    parser.add_argument("--tsc-max", type=float, default=None,
                        help="Keep only clusters with TSC <= this value (Gyr)")
    parser.add_argument("--dataset", type=str, default="dataset.h5",
                        help="Path to dataset HDF5 file (default: dataset.h5)")
    parser.add_argument("--weights-eval-only", action="store_true",
                        help="load weights but train unweighted, so the "
                             "weighted score can be compared against a "
                             "weighted-training run")
    parser.add_argument("--sample-weights", type=str, default=None,
                        help="npz with halo_id/weight arrays (from "
                             "compare_sim_obs_distributions.py) to tilt the "
                             "training loss toward the observed L_X distribution")
    parser.add_argument("--select", choices=["inner", "final"], default="inner",
                        help="pick the reported checkpoint on an inner split "
                             "of the training folds (default) or just take "
                             "the last epoch")
    parser.add_argument("--inner-frac", type=float, default=0.15,
                        help="fraction of each training fold held out for "
                             "checkpoint selection (--select inner)")
    args = parser.parse_args()
    main(args.tau, args.folds, args.epochs, args.batch_size, args.seed,
         args.pseudo_tsc, args.merger_tsc, args.huber_delta, args.tsc_max,
         args.dataset, args.sample_weights, args.weights_eval_only,
         args.select, args.inner_frac)
