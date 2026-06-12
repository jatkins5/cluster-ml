#!/usr/bin/env python3
"""
CNN training for the diffusion-augmentation experiment, at 64px.

Trains a small CNN to predict merger-TSC from 64x64 single-channel radio
maps. Optional --aug-samples NPZ mixes synthetic (image, TSC) pairs from
the TSC-conditional diffusion model into the train set. Cluster-level
train/val split fixed at seed=0 (matches the diffusion model's split), so
val clusters are identical across baseline and augmented runs.

The headline number is R² on the recent-merger subset (TSC <= 2 Gyr) —
the regime where the existing pooled CNN at 128px collapsed to R² 0.149.
Predictions are saved as NPZ for downstream side-by-side comparison.
"""

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

OUT = "cnn_aug_out"  # overridden by --out-dir at runtime


class RadioMapsCNN(Dataset):
    """Projection-level grayscale maps with 8x rot/flip augmentation."""

    def __init__(self, imgs: np.ndarray, labels: np.ndarray,
                 train: bool = False, seed: int = 0):
        self.imgs = imgs                            # (M, 1, S, S) in [-1, 1]
        self.labels = labels                        # (M,) in Gyr
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.imgs[i]
        if self.train:
            k = self.rng.integers(4)
            x = np.rot90(x, k, axes=(1, 2))
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=2)
            x = np.ascontiguousarray(x)
        return (torch.from_numpy(x),
                torch.tensor(self.labels[i], dtype=torch.float32))


class CNN(nn.Module):
    """4-stage CNN, single-channel → scalar regression. Size-agnostic via
    AdaptiveAvgPool in the head: 3 downsamples then global pool, so works at
    64 (8x8 → pool), 128 (16x16 → pool), etc."""

    def __init__(self, ch: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.AvgPool2d(2),                                            # 32
            nn.Conv2d(ch, 2*ch, 3, padding=1), nn.GroupNorm(8, 2*ch), nn.SiLU(),
            nn.Conv2d(2*ch, 2*ch, 3, padding=1), nn.GroupNorm(8, 2*ch), nn.SiLU(),
            nn.AvgPool2d(2),                                            # 16
            nn.Conv2d(2*ch, 4*ch, 3, padding=1), nn.GroupNorm(8, 4*ch), nn.SiLU(),
            nn.Conv2d(4*ch, 4*ch, 3, padding=1), nn.GroupNorm(8, 4*ch), nn.SiLU(),
            nn.AvgPool2d(2),                                            # 8
            nn.Conv2d(4*ch, 8*ch, 3, padding=1), nn.GroupNorm(8, 8*ch), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(8*ch, 4*ch), nn.SiLU(),
            nn.Linear(4*ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


def load_real(
    data_path: str, labels_path: str, label_key: str,
    split_seed: int = 0, val_frac: float = 0.15,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Cluster-level split matching the diffusion model's seed-0 split.
    Returns ((train_imgs, train_labels), (val_imgs, val_labels)); the 3
    projections per cluster stay in the same fold and NaN-label clusters
    are dropped."""
    with h5py.File(data_path, "r") as f:
        imgs = f["images"][:]                       # (N, 3, S, S)
        halo = f["meta/halo_id"][:]
    with h5py.File(labels_path, "r") as f:
        label_halo = f["halo_id"][:]
        label_vals = f[label_key][:]
    label_by_halo = {int(h): float(v) for h, v in zip(label_halo, label_vals)}
    labels = np.array([label_by_halo.get(int(h), np.nan) for h in halo],
                      dtype=np.float32)

    rng = np.random.default_rng(split_seed)
    unique_halos = np.unique(halo)
    rng.shuffle(unique_halos)
    n_val = int(len(unique_halos) * val_frac)
    val_halos = set(unique_halos[:n_val].tolist())
    train_mask = np.array([h not in val_halos for h in halo])

    def expand(cluster_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Drop NaN-label clusters, then flatten 3 projections into samples."""
        cluster_labels = labels[cluster_mask]
        valid = ~np.isnan(cluster_labels)
        selected = imgs[cluster_mask][valid]
        return (selected.reshape(-1, 1, *selected.shape[2:]).astype(np.float32),
                np.repeat(cluster_labels[valid], 3).astype(np.float32))

    return expand(train_mask), expand(~train_mask)


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main(args: argparse.Namespace) -> None:
    global OUT
    OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    (train_imgs, train_labels), (val_imgs, val_labels) = load_real(
        args.data, args.labels, args.label_key, split_seed=args.split_seed)
    n_real = len(train_imgs)
    print(f"real train: {n_real}  val: {len(val_imgs)}  size: {train_imgs.shape[-1]}")
    print(f"real train TSC range: [{train_labels.min():.2f}, {train_labels.max():.2f}]")

    # optionally mix in synthetic (image, TSC) pairs from the diffusion model
    if args.aug_samples:
        aug_npz = np.load(args.aug_samples)
        aug_imgs = aug_npz["samples"].astype(np.float32)
        aug_labels = aug_npz["tsc"].astype(np.float32)
        # keep only the TSC band where conditioning is clean (default 0.3-2.0)
        in_band = (aug_labels >= args.aug_tsc_lo) & (aug_labels <= args.aug_tsc_hi)
        aug_imgs, aug_labels = aug_imgs[in_band], aug_labels[in_band]
        print(f"aug: {len(aug_imgs)} samples in TSC [{args.aug_tsc_lo}, "
              f"{args.aug_tsc_hi}]  (from {len(aug_npz['samples'])} total)")
        train_imgs = np.concatenate([train_imgs, aug_imgs], axis=0)
        train_labels = np.concatenate([train_labels, aug_labels], axis=0)
        print(f"mixed train: {len(train_imgs)}  (real {n_real} + aug {len(aug_imgs)})")

    train_loader = DataLoader(
        RadioMapsCNN(train_imgs, train_labels, train=True, seed=args.seed),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, drop_last=True)

    model = CNN(ch=args.ch).to(dev)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CNN params: {nparam:.2f}M  (input size {train_imgs.shape[-1]})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    val_tensor = torch.from_numpy(val_imgs).to(dev)
    recent_mask = val_labels <= 2.0

    history = []
    best_r2 = -np.inf
    best_state = None
    for ep in range(args.epochs):
        model.train()
        loss_tot = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            pred = model(x)
            loss = F.huber_loss(pred, y, delta=args.huber_delta)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_tot += loss.item() * x.size(0)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor).cpu().numpy()
        r2_all = r2(val_pred, val_labels)
        r2_recent = (r2(val_pred[recent_mask], val_labels[recent_mask])
                     if recent_mask.sum() > 5 else float("nan"))
        history.append((ep, loss_tot/len(train_imgs), r2_all, r2_recent))
        # select best by recent-merger R^2 (the primary readout)
        if r2_recent > best_r2:
            best_r2 = r2_recent
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}  loss {loss_tot/len(train_imgs):.4f}  "
                  f"r2_all={r2_all:+.3f}  r2_recent={r2_recent:+.3f}")

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        val_pred = model(val_tensor).cpu().numpy()

    print(f"\n=== {args.tag}  best-by-recent-r2 ===")
    print(f"R^2 all clusters (TSC 0-7.7):  {r2(val_pred, val_labels):+.4f}  "
          f"(n={len(val_labels)})")
    print(f"R^2 recent (TSC <= 2 Gyr):     "
          f"{r2(val_pred[recent_mask], val_labels[recent_mask]):+.4f}  "
          f"(n={int(recent_mask.sum())})")
    very_recent_mask = val_labels <= 1.0
    print(f"R^2 very recent (TSC <= 1):    "
          f"{r2(val_pred[very_recent_mask], val_labels[very_recent_mask]):+.4f}  "
          f"(n={int(very_recent_mask.sum())})")

    np.savez(f"{OUT}/preds_{args.tag}.npz",
             y=val_labels, yhat=val_pred, history=np.array(history),
             aug_used=bool(args.aug_samples))
    print(f"saved {OUT}/preds_{args.tag}.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="diffusion_radio_64_v2.h5")
    p.add_argument("--labels", type=str,
                   default="TSC_Cutimages/TSC_eachhalo_snap99.hdf5")
    p.add_argument("--label-key", type=str, default="tsc_gyr")
    p.add_argument("--aug-samples", type=str, default=None)
    p.add_argument("--aug-tsc-lo", type=float, default=0.3)
    p.add_argument("--aug-tsc-hi", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ch", type=int, default=32)
    p.add_argument("--huber-delta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0,
                   help="training-randomness seed (NOT split seed)")
    p.add_argument("--split-seed", type=int, default=0,
                   help="cluster-level split seed; keep identical across runs")
    p.add_argument("--tag", type=str, default="baseline")
    p.add_argument("--out-dir", type=str, default="cnn_aug_out")
    main(p.parse_args())
