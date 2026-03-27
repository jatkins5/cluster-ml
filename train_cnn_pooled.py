#!/usr/bin/env python3
"""
Pooled CNN: predict merger label from all 3 projections simultaneously.

Instead of treating each projection as an independent sample, this model
processes all 3 projections through a shared CNN backbone, pools the
embeddings, then predicts from the combined representation.

This gives one prediction per cluster (N=352) rather than per projection,
which is more principled since the 3 views are correlated.

Usage:
    python train_cnn_pooled.py [--pseudo-tsc] [--folds 5] [--epochs 60]
"""

import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


# ---------- augmentation ----------

def augment(img):
    """Random rotation (0/90/180/270) + optional flip."""
    k = np.random.randint(8)
    if k >= 4:
        img = np.fliplr(img)
    img = np.rot90(img, k % 4)
    return np.ascontiguousarray(img)


# ---------- dataset ----------

class RadioClusterDataset(Dataset):
    def __init__(self, images, labels, train=False):
        # images: (N, 3, H, W) — 3 projections per cluster
        self.images = images
        self.labels = labels
        self.train  = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgs = self.images[idx].copy()   # (3, H, W)
        if self.train:
            # augment each projection independently
            imgs = np.stack([augment(imgs[p]) for p in range(3)])
        imgs  = torch.tensor(imgs[:, None], dtype=torch.float32)  # (3, 1, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return imgs, label


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


class PooledCNN(nn.Module):
    """Shared encoder + mean-pool across projections + regression head."""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1,  32),   # 128 → 64
            ConvBlock(32, 64),   # 64  → 32
            ConvBlock(64, 128),  # 32  → 16
            ConvBlock(128, embed_dim),  # 16 → 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, 3, 1, H, W) — batch of 3 projections
        B, P, C, H, W = x.shape
        # encode all projections through shared backbone
        x = x.reshape(B * P, C, H, W)         # (B*3, 1, H, W)
        x = self.pool(self.encoder(x))         # (B*3, 256, 1, 1)
        x = x.reshape(B, P, -1)               # (B, 3, 256)
        # mean-pool across projections
        x = x.mean(dim=1)                     # (B, 256)
        return self.head(x).squeeze(1)         # (B,)


# ---------- training ----------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        preds = model(imgs.to(device)).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def run_cv(images, labels, n_folds, n_epochs, batch_size, seed, huber_delta=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_r2, fold_mae, fold_rmse = [], [], []
    oof_preds = np.zeros(len(labels))

    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)

        # normalise using training stats only
        tr_mean = images[train_idx].mean()
        tr_std  = images[train_idx].std() + 1e-8
        imgs_tr  = (images[train_idx] - tr_mean) / tr_std
        imgs_val = (images[val_idx]   - tr_mean) / tr_std

        train_ds = RadioClusterDataset(imgs_tr, labels[train_idx], train=True)
        val_ds   = RadioClusterDataset(imgs_val, labels[val_idx],  train=False)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model     = PooledCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.HuberLoss(delta=huber_delta)

        best_rmse, best_preds = np.inf, None
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
            scheduler.step()
            preds, true = evaluate(model, val_dl, device)
            rmse = root_mean_squared_error(true, preds)
            if rmse < best_rmse:
                best_rmse  = rmse
                best_preds = preds.copy()
            if (epoch + 1) % 30 == 0:
                train_preds, train_true = evaluate(model, train_dl, device)
                train_r2   = r2_score(train_true, train_preds)
                train_rmse = root_mean_squared_error(train_true, train_preds)
                val_r2     = r2_score(true, preds)
                print(f"  Fold {fold+1}  Epoch {epoch+1:3d}/{n_epochs}  "
                      f"loss={train_loss:.4f}  "
                      f"train R²={train_r2:.3f}  RMSE={train_rmse:.3f}  |  "
                      f"val R²={val_r2:.3f}  RMSE={rmse:.3f}")

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


def main(tau, n_folds, n_epochs, batch_size, seed, pseudo_tsc=False,
         merger_tsc=False, huber_delta=0.5, log_transform=False):
    dataset_path = "dataset.h5"

    print("Loading dataset...")
    with h5py.File(dataset_path, "r") as f:
        images = f["images"][:]              # (352, 3, H, W)
        halo_ids = f["meta/halo_id"][:]

        if merger_tsc:
            # Load ground-truth TSC from merger catalog
            tsc_path = "TSC_Cutimages/TSC_eachhalo_snap99.hdf5"
            with h5py.File(tsc_path, "r") as ft:
                tsc_hids = ft["halo_id"][:]
                tsc_vals = ft["tsc_gyr"][:]
            tsc_map = dict(zip(tsc_hids, tsc_vals))
            labels = np.array([tsc_map[h] for h in halo_ids], dtype=np.float32)
            # Drop clusters with no recorded collision (NaN TSC)
            valid = ~np.isnan(labels)
            images = images[valid]
            labels = labels[valid]
            print(f"Using merger-catalog TSC label ({valid.sum()}/{len(valid)} clusters, "
                  f"{(~valid).sum()} dropped — no recorded collision)")
        elif pseudo_tsc:
            labels = f["labels/pseudo_tsc"][:]
            print("Using pseudo-TSC label")
        else:
            tau_vals  = f["labels/tau_gyr"][:]
            tau_idx   = int(np.argmin(np.abs(tau_vals - tau)))
            actual_tau = float(tau_vals[tau_idx])
            labels = f["labels/label_score_all"][:, tau_idx]
            print(f"Using tau = {actual_tau:.1f} Gyr (index {tau_idx})")

    labels = labels.astype(np.float32)
    if log_transform:
        labels = np.log1p(labels)
        print("Applied log1p transform to labels")
    N = len(labels)
    print(f"Clusters: {N}  (3 projections pooled per cluster)")
    print(f"Labels:  min={labels.min():.3f}  max={labels.max():.3f}  mean={labels.mean():.3f}")
    print()

    print(f"Huber delta: {huber_delta}")
    run_cv(images, labels, n_folds, n_epochs, batch_size, seed, huber_delta)


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
    parser.add_argument("--log-transform", action="store_true",
                        help="Apply log1p transform to labels")
    args = parser.parse_args()
    main(args.tau, args.folds, args.epochs, args.batch_size, args.seed,
         args.pseudo_tsc, args.merger_tsc, args.huber_delta, args.log_transform)
