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
from sklearn.model_selection import GroupKFold
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
    def __init__(self, images, labels, train=False):
        # images: (N, H, W) float32
        self.images = images
        self.labels = labels
        self.train  = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()   # (H, W)
        if self.train:
            img = augment(img)
        img = torch.tensor(img[None], dtype=torch.float32)  # (1, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


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
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1,  32),   # 128 → 64
            ConvBlock(32, 64),   # 64  → 32
            ConvBlock(64, 128),  # 32  → 16
            ConvBlock(128, 256), # 16  →  8
        )
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
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return preds, labels


def run_cv(images, labels, groups, n_folds, n_epochs, batch_size, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    gkf = GroupKFold(n_splits=n_folds)
    fold_r2, fold_mae, fold_rmse = [], [], []
    oof_preds = np.zeros_like(labels)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(images, labels, groups)):
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)

        # per-fold normalisation using training set stats only
        tr_mean = images[train_idx].mean()
        tr_std  = images[train_idx].std() + 1e-8
        imgs_tr = (images[train_idx] - tr_mean) / tr_std
        imgs_val = (images[val_idx]  - tr_mean) / tr_std

        train_ds = RadioDataset(imgs_tr, labels[train_idx], train=True)
        val_ds   = RadioDataset(imgs_val, labels[val_idx],  train=False)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model     = ShallowCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.HuberLoss(delta=0.5)

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


def main(tau, n_folds, n_epochs, batch_size, seed, pseudo_tsc=False):
    dataset_path = "dataset.h5"

    print("Loading dataset...")
    with h5py.File(dataset_path, "r") as f:
        raw_images = f["images"][:]              # (352, 3, H, W)
        if pseudo_tsc:
            all_labels = f["labels/pseudo_tsc"][:]
            print("Using pseudo-TSC label")
        else:
            tau_vals  = f["labels/tau_gyr"][:]
            tau_idx   = int(np.argmin(np.abs(tau_vals - tau)))
            actual_tau = float(tau_vals[tau_idx])
            all_labels = f["labels/label_score_all"][:, tau_idx]
            print(f"Using tau = {actual_tau:.1f} Gyr (index {tau_idx})")

    N, P, H, W = raw_images.shape   # 352, 3, 128, 128

    # flatten projections into separate samples; track cluster group
    images = raw_images.reshape(N * P, H, W)         # (1056, H, W)
    labels = np.repeat(all_labels, P).astype(np.float32)  # (1056,)
    groups = np.repeat(np.arange(N), P)              # cluster index per sample

    print(f"Samples: {len(images)}  (clusters={N}, projections={P})")
    print(f"Labels:  min={labels.min():.3f}  max={labels.max():.3f}  mean={labels.mean():.3f}")
    print()

    run_cv(images, labels, groups, n_folds, n_epochs, batch_size, seed)


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
    args = parser.parse_args()
    main(args.tau, args.folds, args.epochs, args.batch_size, args.seed, args.pseudo_tsc)
