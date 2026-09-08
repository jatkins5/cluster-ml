#!/usr/bin/env python3
"""Train the pooled CNN on forward-modelled mocks, then apply it to real LoTSS.

This is the transfer experiment. Everything else in this project trains on
clean sim images, so its R2 is a ceiling rather than an estimate of real-data
performance; here the training set has been through the beam, the noise and
the same 1 Mpc grid as the observations.

Reports two things:
  1. OOF R2 on held-out mock clusters -- how much TSC signal survives
     observational degradation. This is the transfer-realistic ceiling.
  2. Predictions for the real LoVoCCS/LoTSS cutouts, from an ensemble of the
     fold models. There is no ground-truth TSC for those, so this cannot be
     scored; it is reported so the prediction distribution can be compared
     against the mock distribution. A model that has not transferred usually
     shows it here as collapse to the training mean or excursion out of range.
"""
import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from train_cnn_pooled import PooledCNN, augment


class MockDataset(Dataset):
    """One item per (cluster, realization); each carries its 3 projections."""

    def __init__(self, images, labels, train=False):
        self.images = images          # (M, 3, H, W)
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        imgs = self.images[i].copy()
        if self.train:
            imgs = np.stack([augment(imgs[p]) for p in range(imgs.shape[0])])
        return (torch.tensor(imgs[:, None], dtype=torch.float32),
                torch.tensor(self.labels[i], dtype=torch.float32))


def run_fold(tr_x, tr_y, va_x, va_y, epochs, batch_size, device, delta):
    model = PooledCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.HuberLoss(delta=delta)
    tl = DataLoader(MockDataset(tr_x, tr_y, train=True), batch_size=batch_size,
                    shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(MockDataset(va_x, va_y), batch_size=batch_size)
    best, best_state = -np.inf, None
    for ep in range(epochs):
        model.train()
        for imgs, y in tl:
            imgs, y = imgs.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), y)
            loss.backward()
            opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, _ in vl:
                preds.append(model(imgs.to(device)).cpu().numpy())
        r2 = r2_score(va_y, np.concatenate(preds))
        if r2 > best:
            best = r2
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f"    epoch {ep+1:3d}/{epochs}  val R2={r2:+.3f} "
                  f"(best {best:+.3f})")
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict(model, x, device, batch_size=32):
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        b = torch.tensor(x[i:i + batch_size][:, :, None],
                         dtype=torch.float32).to(device)
        out.append(model(b).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mock_dataset_nh4.h5")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--huber-delta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="mocktransfer")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with h5py.File(args.dataset, "r") as f:
        mock = f["mock/images"][:]            # (N, R, 3, H, W)
        halo = f["mock/halo_id"][:]
        tsc = f["mock/pseudo_tsc"][:]
        obs = f["obs/images"][:]              # (n_obs, H, W)
        obs_name = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in f["obs/name"][:]]
        obs_z = f["obs/z"][:]
    N, R, P, H, _ = mock.shape
    print(f"mock {mock.shape}, obs {obs.shape}, device {device}")

    # Flatten realizations into samples, keeping cluster identity for grouping
    # so no cluster is split across folds.
    x = mock.reshape(N * R, P, H, H)
    y = np.repeat(tsc, R).astype(np.float32)
    groups = np.repeat(halo, R)

    # The pooled model expects 3 views; a real cutout has one. Repeating it
    # three times leaves the mean-pool unchanged, so the obs path is a plain
    # single-image forward pass through the same weights.
    obs_x = np.repeat(obs[:, None], P, axis=1)

    oof = np.full(N * R, np.nan, dtype=np.float32)
    obs_preds = []
    gkf = GroupKFold(n_splits=args.folds)
    for k, (tr, va) in enumerate(gkf.split(x, y, groups), 1):
        print(f"  fold {k}/{args.folds}: train {len(tr)}, val {len(va)}")
        model, best = run_fold(x[tr], y[tr], x[va], y[va], args.epochs,
                               args.batch_size, device, args.huber_delta)
        oof[va] = predict(model, x[va], device)
        obs_preds.append(predict(model, obs_x, device))
        print(f"  fold {k} best val R2 = {best:+.3f}")

    # Score per cluster: average the realizations, which is what a real
    # analysis would do with repeat observations of one cluster.
    oof_cluster = oof.reshape(N, R).mean(axis=1)
    r2 = r2_score(tsc, oof_cluster)
    mae = mean_absolute_error(tsc, oof_cluster)
    print(f"\nmock OOF R2 (per realization): {r2_score(y, oof):+.3f}")
    print(f"mock OOF R2 (cluster-mean)  : {r2:+.3f}   MAE {mae:.3f} Gyr")

    obs_pred = np.mean(obs_preds, axis=0)
    print(f"\nreal LoTSS predictions from the {args.folds}-model ensemble:")
    print(f"{'target':<16}{'z':>8}{'pred TSC [Gyr]':>16}")
    for n, z, p in zip(obs_name, obs_z, obs_pred):
        print(f"{n:<16}{z:>8.4f}{p:>16.2f}")
    print(f"\n{'':<16}{'mock (train)':>16}{'real LoTSS':>14}")
    for lab, fn in [("median", np.median), ("mean", np.mean),
                    ("std", np.std), ("min", np.min), ("max", np.max)]:
        print(f"{lab:<16}{fn(tsc):>16.3f}{fn(obs_pred):>14.3f}")

    np.savez(f"{args.out_prefix}_preds.npz", oof_cluster=oof_cluster,
             tsc=tsc, halo_id=halo, obs_pred=obs_pred,
             obs_name=np.array(obs_name), obs_z=obs_z)
    print(f"\nsaved -> {args.out_prefix}_preds.npz")


if __name__ == "__main__":
    main()
