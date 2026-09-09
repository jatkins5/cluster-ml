#!/usr/bin/env python3
"""
Dual-encoder CNN: predict TSC from one matched (radio, X-ray) projection pair.

Each training sample is (radio_img, xray_img, label) for a single viewing axis
of a single cluster — the realistic single-observation regime.

Two independent shallow-CNN encoders process each modality, their global
average-pooled features are concatenated, and a 2-layer MLP head predicts TSC.

Projection pairing follows the existing convention:
    radio (xy, yz, xz)  ↔  X-ray view along (z, x, y)
Both datasets are stored in the same projection order, so we just stack
along a new modality axis.

Usage:
    python train_cnn_dual.py \
        --radio-dataset dataset.h5 \
        --xray-dataset dataset_xray_128.h5 \
        --merger-tsc [--huber-delta 2.0] [--epochs 120]
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

def augment_pair(radio, xray, rng):
    """Apply the same 4-rot * 2-flip transform to both modalities."""
    k = rng.integers(8)
    if k >= 4:
        radio = np.fliplr(radio)
        xray  = np.fliplr(xray)
    radio = np.rot90(radio, k % 4)
    xray  = np.rot90(xray,  k % 4)
    return np.ascontiguousarray(radio), np.ascontiguousarray(xray)


# ---------- dataset ----------

class DualModalDataset(Dataset):
    """Returns matched (radio, xray) projection pairs with shared augmentation."""

    def __init__(self, radio_imgs, xray_imgs, labels, train=False, seed=0):
        self.radio = radio_imgs
        self.xray  = xray_imgs
        self.labels = labels
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r = self.radio[idx].copy()
        x = self.xray[idx].copy()
        if self.train:
            r, x = augment_pair(r, x, self.rng)
        r = torch.tensor(r[None], dtype=torch.float32)   # (1, H, W)
        x = torch.tensor(x[None], dtype=torch.float32)   # (1, H, W)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return r, x, y


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


def make_encoder(img_size):
    """Encoder mirrors ShallowCNN feature extractor; depth scales with img_size."""
    blocks = [
        ConvBlock(1,   32),   # /2
        ConvBlock(32,  64),   # /4
        ConvBlock(64,  128),  # /8
        ConvBlock(128, 256),  # /16
    ]
    if img_size >= 512:
        blocks.append(ConvBlock(256, 256))   # /32
        blocks.append(ConvBlock(256, 256))   # /64
    elif img_size >= 256:
        blocks.append(ConvBlock(256, 256))   # /32
    return nn.Sequential(*blocks)


class DualCNN(nn.Module):
    """Two parallel encoders -> GAP -> concat -> MLP."""

    def __init__(self, radio_size=128, xray_size=128):
        super().__init__()
        self.radio_enc = make_encoder(radio_size)
        self.xray_enc  = make_encoder(xray_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 256 features per encoder, concatenated -> 512
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, radio, xray):
        fr = self.pool(self.radio_enc(radio)).flatten(1)   # (B, 256)
        fx = self.pool(self.xray_enc(xray)).flatten(1)     # (B, 256)
        return self.head(torch.cat([fr, fx], dim=1)).squeeze(1)


# ---------- training ----------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for radio, xray, y in loader:
        radio, xray, y = radio.to(device), xray.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(radio, xray)
        loss  = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for radio, xray, y in loader:
        preds = model(radio.to(device), xray.to(device)).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def run_cv(radio, xray, labels, groups, n_folds, n_epochs, batch_size, seed,
           huber_delta, radio_size, xray_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    gkf = GroupKFold(n_splits=n_folds)
    fold_r2, fold_mae, fold_rmse = [], [], []
    oof_preds = np.zeros_like(labels)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(radio, labels, groups)):
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)

        # per-fold normalisation, computed independently for each modality
        # using only the training set
        r_mean, r_std = radio[tr_idx].mean(), radio[tr_idx].std() + 1e-8
        x_mean, x_std = xray[tr_idx].mean(),  xray[tr_idx].std()  + 1e-8

        r_tr  = (radio[tr_idx] - r_mean) / r_std
        r_val = (radio[val_idx] - r_mean) / r_std
        x_tr  = (xray[tr_idx]  - x_mean) / x_std
        x_val = (xray[val_idx] - x_mean) / x_std

        train_ds = DualModalDataset(r_tr, x_tr, labels[tr_idx],
                                    train=True, seed=seed + fold)
        val_ds   = DualModalDataset(r_val, x_val, labels[val_idx], train=False)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
        val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

        model     = DualCNN(radio_size=radio_size, xray_size=xray_size).to(device)
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
                best_rmse, best_preds = rmse, preds.copy()
            if (epoch + 1) % 30 == 0:
                tr_preds, tr_true = evaluate(model, train_dl, device)
                tr_r2   = r2_score(tr_true, tr_preds)
                tr_rmse = root_mean_squared_error(tr_true, tr_preds)
                val_r2  = r2_score(true, preds)
                print(f"  Fold {fold+1}  Epoch {epoch+1:3d}/{n_epochs}  "
                      f"loss={train_loss:.4f}  "
                      f"train R²={tr_r2:.3f}  RMSE={tr_rmse:.3f}  |  "
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


def main(radio_path, xray_path, n_folds, n_epochs, batch_size, seed,
         merger_tsc, pseudo_tsc, huber_delta):

    print(f"Radio dataset: {radio_path}")
    print(f"X-ray dataset: {xray_path}")
    with h5py.File(radio_path, "r") as f:
        radio_imgs  = f["images"][:]              # (N, 3, Hr, Wr)
        radio_hids  = f["meta/halo_id"][:]
        if pseudo_tsc:
            all_labels = f["labels/pseudo_tsc"][:]
            label_name = "pseudo-TSC"
        elif merger_tsc:
            all_labels = None  # filled below
            label_name = "merger-TSC"
        else:
            raise SystemExit("must use --merger-tsc or --pseudo-tsc")

    with h5py.File(xray_path, "r") as f:
        xray_imgs = f["images"][:]                # (N, 3, Hx, Wx)
        xray_hids = f["meta/halo_id"][:]

    assert np.array_equal(radio_hids, xray_hids), \
        "halo_id ordering differs between radio and xray datasets"

    if merger_tsc:
        with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5", "r") as ft:
            tsc_hids = ft["halo_id"][:]
            tsc_vals = ft["tsc_gyr"][:]
        tsc_map = dict(zip(tsc_hids, tsc_vals))
        all_labels = np.array([tsc_map[h] for h in radio_hids], dtype=np.float32)
        valid = ~np.isnan(all_labels)
        radio_imgs = radio_imgs[valid]
        xray_imgs  = xray_imgs[valid]
        all_labels = all_labels[valid]
        print(f"merger-TSC: kept {valid.sum()}/{len(valid)} clusters")

    N, P, Hr, Wr = radio_imgs.shape
    _, _, Hx, Wx = xray_imgs.shape

    # flatten projections into independent samples (3 per cluster), but
    # keep the per-cluster group label so GroupKFold splits at cluster level
    radio = radio_imgs.reshape(N * P, Hr, Wr)
    xray  = xray_imgs.reshape(N * P, Hx, Wx)
    labels = np.repeat(all_labels, P).astype(np.float32)
    groups = np.repeat(np.arange(N), P)

    print(f"Samples: {len(labels)}  (clusters={N}, projections={P})")
    print(f"Radio:   {Hr}x{Wr}    X-ray: {Hx}x{Wx}")
    print(f"Label:   {label_name}  range=[{labels.min():.2f}, {labels.max():.2f}]")
    print(f"Huber delta: {huber_delta}")
    print()

    run_cv(radio, xray, labels, groups, n_folds, n_epochs, batch_size, seed,
           huber_delta, radio_size=Wr, xray_size=Wx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radio-dataset", type=str, default="dataset.h5")
    parser.add_argument("--xray-dataset",  type=str, default="dataset_xray_128.h5")
    parser.add_argument("--folds",       type=int, default=5)
    parser.add_argument("--epochs",      type=int, default=120)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--merger-tsc",  action="store_true")
    parser.add_argument("--pseudo-tsc",  action="store_true")
    parser.add_argument("--huber-delta", type=float, default=2.0)
    args = parser.parse_args()
    main(args.radio_dataset, args.xray_dataset, args.folds, args.epochs,
         args.batch_size, args.seed, args.merger_tsc, args.pseudo_tsc,
         args.huber_delta)
