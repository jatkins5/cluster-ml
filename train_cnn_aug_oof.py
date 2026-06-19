#!/usr/bin/env python3
"""
Out-of-fold (OOF) version of the diffusion-augmentation CNN experiment.

`train_cnn_aug.py` uses a single 15% cluster-level val split and selects the
best epoch by recent-merger R² — good for studying the TSC <= 2 Gyr regime,
but its "R² all" is not comparable to the 5-fold OOF R² reported in the
README Comparison table (e.g. the shallow-CNN 0.511).

This script answers the comparable question: under 5-fold GroupKFold (every
cluster predicted exactly once while held out) with the checkpoint selected
by OVERALL R², does mixing synthetic samples into each fold's training set
improve overall performance?

Notes / caveats:
  - Runs entirely in the diffusion-pipeline value space
    (diffusion_radio_*_v2.h5), so the baseline here will NOT equal 0.511 —
    that number used a different model (the shallow CNN) on dataset.h5. This
    is an honest aug-vs-no-aug delta on a consistent footing, not a literal
    reproduction of the 0.511 run.
  - Synthetic samples are mixed into the TRAIN portion of every fold only;
    they are never scored (they belong to no held-out cluster).
  - Best-epoch selection peeks at the fold's val R² to pick the checkpoint,
    matching the "best-epoch checkpoint" methodology of the 0.511 runs.

Usage:
  python train_cnn_aug_oof.py --data diffusion_radio_128_v2.h5 --ch 48 \
      --epochs 80 --tag baseline
  python train_cnn_aug_oof.py --data diffusion_radio_128_v2.h5 --ch 48 \
      --epochs 80 --tag aug \
      --aug-samples diffusion_out_cond_128_ada/samples_cond.npz
"""

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from train_cnn_aug import CNN, RadioMapsCNN, r2


def load_all(data_path: str, labels_path: str, label_key: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load every projection as an individual sample, dropping NaN-label
    clusters. Returns (imgs, labels, halo_per_sample); the 3 projections of
    a cluster share a halo id so GroupKFold keeps them in the same fold."""
    with h5py.File(data_path, "r") as f:
        imgs = f["images"][:]                       # (N, 3, S, S)
        halo = f["meta/halo_id"][:]
    with h5py.File(labels_path, "r") as f:
        label_halo = f["halo_id"][:]
        label_vals = f[label_key][:]
    label_by_halo = {int(h): float(v) for h, v in zip(label_halo, label_vals)}
    cluster_labels = np.array([label_by_halo.get(int(h), np.nan) for h in halo],
                              dtype=np.float32)

    valid = ~np.isnan(cluster_labels)
    imgs = imgs[valid]                              # (n, 3, S, S)
    sample_imgs = imgs.reshape(-1, 1, *imgs.shape[2:]).astype(np.float32)
    sample_labels = np.repeat(cluster_labels[valid], 3).astype(np.float32)
    sample_halo = np.repeat(halo[valid], 3)
    return sample_imgs, sample_labels, sample_halo


def load_aug(aug_path: str, tsc_lo: float, tsc_hi: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (image, TSC) pairs filtered to a clean conditioning band."""
    d = np.load(aug_path)
    aug_imgs = d["samples"].astype(np.float32)
    aug_labels = d["tsc"].astype(np.float32)
    in_band = (aug_labels >= tsc_lo) & (aug_labels <= tsc_hi)
    return aug_imgs[in_band], aug_labels[in_band]


def train_one_fold(train_imgs: np.ndarray, train_labels: np.ndarray,
                   val_imgs: np.ndarray, val_labels: np.ndarray,
                   args: argparse.Namespace, dev: str) -> np.ndarray:
    """Train a fresh CNN on one fold; return val predictions from the
    checkpoint that maximises the selection metric (overall or recent R²)."""
    loader = DataLoader(
        RadioMapsCNN(train_imgs, train_labels, train=True, seed=args.seed),
        batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = CNN(ch=args.ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    val_tensor = torch.from_numpy(val_imgs).to(dev)
    recent_mask = val_labels <= 2.0

    best_score = -np.inf
    best_pred = None
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            loss = F.huber_loss(model(x), y, delta=args.huber_delta)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor).cpu().numpy()
        if args.select_by == "recent":
            score = (r2(val_pred[recent_mask], val_labels[recent_mask])
                     if recent_mask.sum() > 5 else -np.inf)
        else:
            score = r2(val_pred, val_labels)
        # NaN guard so an undefined-variance epoch never wins selection
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_pred = val_pred.copy()

    # fall back to the last epoch if no epoch produced a finite score
    return best_pred if best_pred is not None else val_pred


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    imgs, labels, halo = load_all(args.data, args.labels, args.label_key)
    print(f"{len(imgs)} projection samples from {len(np.unique(halo))} clusters  "
          f"size={imgs.shape[-1]}  TSC range [{labels.min():.2f}, {labels.max():.2f}]")

    aug_imgs = aug_labels = None
    if args.aug_samples:
        aug_imgs, aug_labels = load_aug(args.aug_samples, args.aug_tsc_lo,
                                        args.aug_tsc_hi)
        print(f"aug: {len(aug_imgs)} synthetic samples in "
              f"TSC [{args.aug_tsc_lo}, {args.aug_tsc_hi}] mixed into each "
              f"fold's train set")

    gkf = GroupKFold(n_splits=args.n_folds)
    oof_pred = np.full(len(labels), np.nan, dtype=np.float32)
    fold_r2_all = []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(imgs, labels, groups=halo)):
        tr_imgs, tr_labels = imgs[tr_idx], labels[tr_idx]
        if aug_imgs is not None:
            tr_imgs = np.concatenate([tr_imgs, aug_imgs], axis=0)
            tr_labels = np.concatenate([tr_labels, aug_labels], axis=0)
        val_pred = train_one_fold(tr_imgs, tr_labels,
                                  imgs[val_idx], labels[val_idx], args, dev)
        oof_pred[val_idx] = val_pred
        r2_fold = r2(val_pred, labels[val_idx])
        fold_r2_all.append(r2_fold)
        print(f"fold {fold}: train {len(tr_imgs)} (+aug) val {len(val_idx)}  "
              f"R²_all={r2_fold:+.3f}")

    # pooled OOF metrics (every cluster predicted exactly once)
    recent = labels <= 2.0
    very = labels <= 1.0
    late = labels > 2.0
    print(f"\n=== {args.tag}  OOF (select_by={args.select_by}, "
          f"{args.n_folds}-fold) ===")
    print(f"OOF R² all (TSC 0–7.7):     {r2(oof_pred, labels):+.4f}  "
          f"(n={len(labels)})   per-fold {np.mean(fold_r2_all):+.3f} "
          f"± {np.std(fold_r2_all):.3f}")
    print(f"OOF R² recent (TSC ≤ 2):    {r2(oof_pred[recent], labels[recent]):+.4f}"
          f"  (n={int(recent.sum())})")
    print(f"OOF R² very recent (≤ 1):   {r2(oof_pred[very], labels[very]):+.4f}"
          f"  (n={int(very.sum())})")
    print(f"OOF R² late (TSC > 2):      {r2(oof_pred[late], labels[late]):+.4f}"
          f"  (n={int(late.sum())})")

    out_path = f"{args.out_dir}/oof_{args.tag}.npz"
    np.savez(out_path, y=labels, yhat=oof_pred, halo=halo,
             fold_r2_all=np.array(fold_r2_all),
             aug_used=bool(args.aug_samples), select_by=args.select_by)
    print(f"saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="diffusion_radio_128_v2.h5")
    p.add_argument("--labels", type=str,
                   default="TSC_Cutimages/TSC_eachhalo_snap99.hdf5")
    p.add_argument("--label-key", type=str, default="tsc_gyr")
    p.add_argument("--aug-samples", type=str, default=None)
    p.add_argument("--aug-tsc-lo", type=float, default=0.3)
    p.add_argument("--aug-tsc-hi", type=float, default=2.0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--select-by", type=str, default="overall",
                   choices=["overall", "recent"],
                   help="metric used to pick the best-epoch checkpoint per fold")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ch", type=int, default=48)
    p.add_argument("--huber-delta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="baseline")
    p.add_argument("--out-dir", type=str, default="cnn_aug_oof_128")
    main(p.parse_args())
