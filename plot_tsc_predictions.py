#!/usr/bin/env python3
"""Re-run shallow CNN CV with merger-TSC and save a true-vs-predicted scatter plot."""

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- model + dataset classes (mirrors train_cnn.py) ----

def augment(img):
    k = np.random.randint(8)
    if k >= 4:
        img = np.fliplr(img)
    img = np.rot90(img, k % 4)
    return np.ascontiguousarray(img)


class RadioDataset(Dataset):
    def __init__(self, images, labels, train=False):
        self.images = images
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        if self.train:
            img = augment(img)
        img = torch.tensor(img[None], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


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
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        ]
        if img_size >= 512:
            blocks.append(ConvBlock(256, 256))
            blocks.append(ConvBlock(256, 256))
        elif img_size >= 256:
            blocks.append(ConvBlock(256, 256))
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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        preds = model(imgs.to(device)).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset.h5")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Override image size for model (auto-detected from dataset if omitted)")
    parser.add_argument("--output", type=str, default="tsc_true_vs_pred.png")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--huber-delta", type=float, default=2.0)
    args = parser.parse_args()

    seed = 42
    n_folds = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    with h5py.File(args.dataset, "r") as f:
        raw_images = f["images"][:]
        halo_ids = f["meta/halo_id"][:]

    with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5", "r") as f:
        tsc_hids = f["halo_id"][:]
        tsc_vals = f["tsc_gyr"][:]

    tsc_map = dict(zip(tsc_hids, tsc_vals))
    all_labels = np.array([tsc_map[h] for h in halo_ids], dtype=np.float32)
    valid = ~np.isnan(all_labels)
    raw_images = raw_images[valid]
    all_labels = all_labels[valid]

    N, P, H, W = raw_images.shape
    img_size = args.img_size if args.img_size is not None else W
    images = raw_images.reshape(N * P, H, W)
    labels = np.repeat(all_labels, P).astype(np.float32)
    groups = np.repeat(np.arange(N), P)

    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(images)}  (clusters={N}, projections={P}, {W}×{W}px)")
    print(f"Model img_size: {img_size}  Huber delta: {args.huber_delta}  Epochs: {args.epochs}")

    # Run CV
    gkf = GroupKFold(n_splits=n_folds)
    oof_preds = np.zeros_like(labels)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(images, labels, groups)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)

        tr_mean = images[train_idx].mean()
        tr_std = images[train_idx].std() + 1e-8
        imgs_tr = (images[train_idx] - tr_mean) / tr_std
        imgs_val = (images[val_idx] - tr_mean) / tr_std

        train_ds = RadioDataset(imgs_tr, labels[train_idx], train=True)
        val_ds = RadioDataset(imgs_val, labels[val_idx], train=False)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = ShallowCNN(img_size=img_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.HuberLoss(delta=args.huber_delta)

        best_rmse, best_preds = np.inf, None
        for epoch in range(args.epochs):
            train_epoch(model, train_dl, optimizer, criterion, device)
            scheduler.step()
            preds, true = evaluate(model, val_dl, device)
            rmse = np.sqrt(np.mean((true - preds) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_preds = preds.copy()
            if (epoch + 1) % 30 == 0:
                r2 = r2_score(true, preds)
                print(f"  Epoch {epoch+1:3d}: RMSE={rmse:.3f}  R²={r2:.3f}")

        oof_preds[val_idx] = best_preds
        fold_r2 = r2_score(labels[val_idx], best_preds)
        fold_mae = mean_absolute_error(labels[val_idx], best_preds)
        fold_rmse = root_mean_squared_error(labels[val_idx], best_preds)
        print(f"  Fold {fold+1} best -> R²={fold_r2:.3f}  MAE={fold_mae:.3f}  RMSE={fold_rmse:.3f}")

    # Aggregate to cluster level (mean of 3 projections)
    cluster_true = labels.reshape(N, P)[:, 0]
    cluster_pred = oof_preds.reshape(N, P).mean(axis=1)

    oof_r2 = r2_score(labels, oof_preds)
    cluster_r2 = r2_score(cluster_true, cluster_pred)
    print(f"\nOOF R² (per-projection): {oof_r2:.3f}")
    print(f"OOF R² (cluster-averaged): {cluster_r2:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Per-projection scatter
    ax = axes[0]
    ax.scatter(labels, oof_preds, alpha=0.3, s=12, c="steelblue", edgecolors="none")
    lims = [min(labels.min(), oof_preds.min()) - 0.2, max(labels.max(), oof_preds.max()) + 0.2]
    ax.plot(lims, lims, "k--", lw=1, label="Perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True TSC (Gyr)")
    ax.set_ylabel("Predicted TSC (Gyr)")
    ax.set_title(f"Per-projection OOF predictions (R²={oof_r2:.3f})")
    ax.legend()
    ax.set_aspect("equal")

    # Cluster-averaged scatter
    ax = axes[1]
    ax.scatter(cluster_true, cluster_pred, alpha=0.5, s=25, c="steelblue", edgecolors="none")
    ax.plot(lims, lims, "k--", lw=1, label="Perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True TSC (Gyr)")
    ax.set_ylabel("Predicted TSC (Gyr, mean of 3 projections)")
    ax.set_title(f"Cluster-averaged OOF predictions (R²={cluster_r2:.3f})")
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
