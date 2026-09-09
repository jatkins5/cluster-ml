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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
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


def run_fold(tr_x, tr_y, sel_x, sel_y, epochs, batch_size, device, delta):
    """Train one fold; pick the checkpoint on `sel`, never on the outer fold.

    `sel` is a held-out slice of this fold's *training* clusters. Selecting
    on the outer fold and then reporting its R2 -- what this did until
    2026-09-09 -- was worth a measured +0.32 R2 here, far more than the
    +0.04 it was worth for the clean-sim baselines, because Adam at lr 1e-3
    with no schedule swung the val curve between -0.5 and +0.5 from epoch to
    epoch and "best" simply picked the top of the swing. The optimizer now
    matches train_cnn_pooled.py (AdamW 3e-4 + cosine), which is most of why
    the curve settles down. Pass sel_x=None to take the final epoch.
    """
    model = PooledCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.HuberLoss(delta=delta)
    tl = DataLoader(MockDataset(tr_x, tr_y, train=True), batch_size=batch_size,
                    shuffle=True, num_workers=2, drop_last=True)
    sl = (None if sel_x is None
          else DataLoader(MockDataset(sel_x, sel_y), batch_size=batch_size))
    best, best_state, best_ep = -np.inf, None, epochs
    for ep in range(epochs):
        model.train()
        for imgs, y in tl:
            imgs, y = imgs.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), y)
            loss.backward()
            opt.step()
        sched.step()
        if sl is None:
            continue
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, _ in sl:
                preds.append(model(imgs.to(device)).cpu().numpy())
        r2 = r2_score(sel_y, np.concatenate(preds))
        if r2 > best:
            best, best_ep = r2, ep + 1
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f"    epoch {ep+1:3d}/{epochs}  sel R2={r2:+.3f} "
                  f"(best {best:+.3f} @ {best_ep})")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, best_ep


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
    ap.add_argument("--select", choices=["inner", "final"], default="inner",
                    help="pick the checkpoint on an inner split of the "
                         "training folds (default) or take the last epoch")
    ap.add_argument("--inner-frac", type=float, default=0.15,
                    help="fraction of each training fold held out for "
                         "checkpoint selection (--select inner)")
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
        if args.select == "inner":
            # Grouped, so every realization of a cluster stays on one side.
            gss = GroupShuffleSplit(n_splits=1, test_size=args.inner_frac,
                                    random_state=args.seed + k)
            rel_fit, rel_sel = next(gss.split(tr, groups=groups[tr]))
            fit, sel = tr[rel_fit], tr[rel_sel]
        else:
            fit, sel = tr, None
        print(f"  fold {k}/{args.folds}: train {len(fit)}, "
              f"sel {0 if sel is None else len(sel)}, val {len(va)}")
        model, best, best_ep = run_fold(
            x[fit], y[fit], None if sel is None else x[sel],
            None if sel is None else y[sel], args.epochs, args.batch_size,
            device, args.huber_delta)
        oof[va] = predict(model, x[va], device)
        obs_preds.append(predict(model, obs_x, device))
        print(f"  fold {k}: checkpoint epoch {best_ep}/{args.epochs}, "
              f"sel R2 {best:+.3f}  ->  outer val R2 "
              f"{r2_score(y[va], oof[va]):+.3f}")

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
             obs_name=np.array(obs_name), obs_z=obs_z,
             select=args.select, seed=args.seed, dataset=args.dataset)
    print(f"\nsaved -> {args.out_prefix}_preds.npz")


if __name__ == "__main__":
    main()
