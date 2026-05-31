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

OUT = "cnn_aug_out"


class RadioMapsCNN(Dataset):
    """Projection-level grayscale maps with 8x rot/flip augmentation."""

    def __init__(self, imgs, labels, train=False, seed=0):
        self.imgs = imgs                            # (M, 1, S, S) in [-1, 1]
        self.labels = labels                        # (M,) in Gyr
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = self.imgs[i]
        if self.train:
            k = self.rng.integers(4)
            x = np.rot90(x, k, axes=(1, 2))
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=2)
            x = np.ascontiguousarray(x)
        return (torch.from_numpy(x),
                torch.tensor(self.labels[i], dtype=torch.float32))


class CNN64(nn.Module):
    """4-block CNN at 64x64 single-channel → scalar regression."""

    def __init__(self, ch=32):
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

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


def load_real(data_path, labels_path, label_key, split_seed=0, val_frac=0.15):
    """Cluster-level split matching the diffusion model's seed-0 split.
    Returns (tr_i, tr_l), (val_i, val_l); NaN-label clusters are dropped."""
    with h5py.File(data_path, "r") as f:
        imgs = f["images"][:]                       # (N, 3, S, S)
        halo = f["meta/halo_id"][:]
    with h5py.File(labels_path, "r") as f:
        lhid = f["halo_id"][:]
        lval = f[label_key][:]
    lmap = {int(h): float(v) for h, v in zip(lhid, lval)}
    labels = np.array([lmap.get(int(h), np.nan) for h in halo], dtype=np.float32)

    rng = np.random.default_rng(split_seed)
    uniq = np.unique(halo)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_frac)
    val_h = set(uniq[:n_val].tolist())
    tr_m = np.array([h not in val_h for h in halo])

    def expand(mask):
        l = labels[mask]
        valid = ~np.isnan(l)
        sel = imgs[mask][valid]
        return (sel.reshape(-1, 1, *sel.shape[2:]).astype(np.float32),
                np.repeat(l[valid], 3).astype(np.float32))

    return expand(tr_m), expand(~tr_m)


def r2(pred, true):
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main(args):
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    (tr_i, tr_l), (val_i, val_l) = load_real(
        args.data, args.labels, args.label_key, split_seed=args.split_seed)
    n_real = len(tr_i)
    print(f"real train: {n_real}  val: {len(val_i)}  size: {tr_i.shape[-1]}")
    print(f"real train TSC range: [{tr_l.min():.2f}, {tr_l.max():.2f}]")

    if args.aug_samples:
        d = np.load(args.aug_samples)
        aug_i = d["samples"].astype(np.float32)
        aug_l = d["tsc"].astype(np.float32)
        # filter aug to a TSC band (default = match where conditioning is clean)
        keep = (aug_l >= args.aug_tsc_lo) & (aug_l <= args.aug_tsc_hi)
        aug_i, aug_l = aug_i[keep], aug_l[keep]
        print(f"aug: {len(aug_i)} samples in TSC [{args.aug_tsc_lo}, "
              f"{args.aug_tsc_hi}]  (from {len(d['samples'])} total)")
        tr_i = np.concatenate([tr_i, aug_i], axis=0)
        tr_l = np.concatenate([tr_l, aug_l], axis=0)
        print(f"mixed train: {len(tr_i)}  (real {n_real} + aug {len(aug_i)})")

    dl = DataLoader(RadioMapsCNN(tr_i, tr_l, train=True, seed=args.seed),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=4, drop_last=True)

    model = CNN64(ch=args.ch).to(dev)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CNN64 params: {nparam:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    val_x = torch.from_numpy(val_i).to(dev)
    recent = val_l <= 2.0

    history = []
    best_r2 = -np.inf
    best_state = None
    for ep in range(args.epochs):
        model.train()
        loss_tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            pred = model(x)
            loss = F.huber_loss(pred, y, delta=args.huber_delta)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_tot += loss.item() * x.size(0)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).cpu().numpy()
        r2_all = r2(val_pred, val_l)
        r2_recent = r2(val_pred[recent], val_l[recent]) if recent.sum() > 5 else float("nan")
        history.append((ep, loss_tot/len(tr_i), r2_all, r2_recent))
        # select best by recent-merger R^2 (the primary readout)
        if r2_recent > best_r2:
            best_r2 = r2_recent
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}  loss {loss_tot/len(tr_i):.4f}  "
                  f"r2_all={r2_all:+.3f}  r2_recent={r2_recent:+.3f}")

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        val_pred = model(val_x).cpu().numpy()

    print(f"\n=== {args.tag}  best-by-recent-r2 ===")
    print(f"R^2 all clusters (TSC 0-7.7):  {r2(val_pred, val_l):+.4f}  (n={len(val_l)})")
    print(f"R^2 recent (TSC <= 2 Gyr):     "
          f"{r2(val_pred[recent], val_l[recent]):+.4f}  (n={int(recent.sum())})")
    very = val_l <= 1.0
    print(f"R^2 very recent (TSC <= 1):    "
          f"{r2(val_pred[very], val_l[very]):+.4f}  (n={int(very.sum())})")

    np.savez(f"{OUT}/preds_{args.tag}.npz",
             y=val_l, yhat=val_pred, history=np.array(history),
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
    main(p.parse_args())
