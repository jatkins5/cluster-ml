"""Does adding CAMELS zooms to training improve TSC prediction on TNG?

Evaluation is 5-fold GroupKFold OOF over TNG-Cluster halos only -- TNG
has merger-catalog TSC labels and matches the real target application,
whereas the CAMELS labels are a mass-jump proxy. CAMELS samples are
mixed into each fold's TRAIN set only and are never scored, exactly as
synthetic augmentation was tested in `train_cnn_aug_oof.py`.

The two simulations use different radio-weight normalizations, so
images are per-image standardized by default; without that the CNN sees
a pure amplitude offset between the domains.

Usage:
  python train_cnn_camels.py --tag tng_only
  python train_cnn_camels.py --tag tng_camels --camels dataset_camels_128.h5
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


def standardize(imgs):
    """Zero-mean/unit-std per image, so the two sims share an amplitude
    scale (their radio weights are normalized differently)."""
    flat = imgs.reshape(len(imgs), -1)
    mu = flat.mean(axis=1)[:, None, None, None]
    sd = flat.std(axis=1)[:, None, None, None]
    return (imgs - mu) / np.maximum(sd, 1e-8)


def load_tng(data_path, labels_path, label_key):
    with h5py.File(data_path, "r") as f:
        imgs = f["images"][:]
        halo = f["meta/halo_id"][:]
    with h5py.File(labels_path, "r") as f:
        by_halo = {int(h): float(v)
                   for h, v in zip(f["halo_id"][:], f[label_key][:])}
    lab = np.array([by_halo.get(int(h), np.nan) for h in halo],
                   dtype=np.float32)
    ok = ~np.isnan(lab)
    imgs = imgs[ok]
    samples = imgs.reshape(-1, 1, *imgs.shape[2:]).astype(np.float32)
    return samples, np.repeat(lab[ok], 3), np.repeat(halo[ok], 3)


def load_camels(path, tsc_max, require_detected, label_path=None,
                m200_min=None, tsc_clip=None):
    with h5py.File(path, "r") as f:
        imgs = f["images"][:]
        tsc = f["labels/merger_tsc"][:]
        detected = f["labels/detected"][:]
        zoom = f["meta/zoom_id"][:]
        m200 = f["meta/m200_msun"][:]
    if label_path is not None:
        with h5py.File(label_path, "r") as f:
            sub = dict(zip(f["zoom_id"][:], f["tsc_proxy"][:]))
        tsc = np.array([sub.get(int(z), np.nan) for z in zoom])
        detected = np.isfinite(tsc)
        print(f"CAMELS labels from {label_path}: "
              f"{detected.sum()}/{len(tsc)} zooms matched")
    keep = np.isfinite(tsc) & (imgs.sum(axis=(1, 2, 3)) > 0)
    if require_detected:
        keep &= detected
    if tsc_max is not None:
        keep &= tsc <= tsc_max
    if m200_min is not None:
        keep &= m200 >= m200_min
        print(f"CAMELS mass cut M200 >= {m200_min:.2e}: {keep.sum()} kept")
    if tsc_clip is not None:
        # relic emission fades long before the tail values, so they are not
        # distinguishable from one another but would dominate a squared loss
        tsc = np.minimum(tsc, tsc_clip)
    imgs = imgs[keep]
    samples = imgs.reshape(-1, 1, *imgs.shape[2:]).astype(np.float32)
    return samples, np.repeat(tsc[keep], 3).astype(np.float32), \
        np.repeat(zoom[keep], 3)


def fit(tr_imgs, tr_labels, val_imgs, val_labels, args, dev,
        epochs=None, lr=None, init_state=None):
    """Train a CNN, keeping the checkpoint with the best val R².

    Returns (best val predictions, best state_dict)."""
    epochs = epochs if epochs is not None else args.epochs
    lr = lr if lr is not None else args.lr
    loader = DataLoader(
        RadioMapsCNN(tr_imgs, tr_labels, train=True, seed=args.seed),
        batch_size=args.batch_size, shuffle=True, num_workers=4,
        drop_last=True)
    model = CNN(ch=args.ch).to(dev)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    val_tensor = torch.from_numpy(val_imgs).to(dev)

    best_score, best_pred, best_state = -np.inf, None, None
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            loss = F.huber_loss(model(x), y, delta=args.huber_delta)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor).cpu().numpy()
        score = r2(val_pred, val_labels)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_pred = val_pred.copy()
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
    if best_pred is None:
        best_pred = val_pred
        best_state = {k: v.detach().clone()
                      for k, v in model.state_dict().items()}
    return best_pred, best_state


def predict(state, imgs, args, dev):
    model = CNN(ch=args.ch).to(dev)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(imgs).to(dev)).cpu().numpy()


def report(tag, pred, labels, extra=""):
    recent, very, late = labels <= 2.0, labels <= 1.0, labels > 2.0
    print(f"\n=== {tag} ===  {extra}")
    print(f"R² all:          {r2(pred, labels):+.4f}  (n={len(labels)})")
    print(f"R² recent (≤2):  {r2(pred[recent], labels[recent]):+.4f}  "
          f"(n={int(recent.sum())})")
    print(f"R² very recent:  {r2(pred[very], labels[very]):+.4f}  "
          f"(n={int(very.sum())})")
    print(f"R² late (>2):    {r2(pred[late], labels[late]):+.4f}  "
          f"(n={int(late.sum())})")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    imgs, labels, halo = load_tng(args.data, args.labels, args.label_key)
    print(f"TNG: {len(imgs)} projections from {len(np.unique(halo))} "
          f"clusters, size={imgs.shape[-1]}, "
          f"TSC [{labels.min():.2f}, {labels.max():.2f}]")
    print(f"  raw pixel stats: mean={imgs.mean():.4f} std={imgs.std():.4f} "
          f"max={imgs.max():.3f}")

    cam_imgs = cam_labels = cam_zoom = None
    if args.camels:
        cam_imgs, cam_labels, cam_zoom = load_camels(
            args.camels, args.camels_tsc_max, args.require_detected,
            args.camels_labels, args.camels_m200_min, args.camels_tsc_clip)
        print(f"CAMELS: {len(cam_imgs)} projections from "
              f"{len(np.unique(cam_zoom))} zooms, "
              f"TSC [{cam_labels.min():.2f}, {cam_labels.max():.2f}]")
        print(f"  raw pixel stats: mean={cam_imgs.mean():.4f} "
              f"std={cam_imgs.std():.4f} max={cam_imgs.max():.3f}")

    if not args.no_standardize:
        imgs = standardize(imgs)
        if cam_imgs is not None:
            cam_imgs = standardize(cam_imgs)
        print("per-image standardization applied to both domains")

    # --- transfer: train on CAMELS only, evaluate on every TNG cluster ---
    if args.mode == "transfer":
        if cam_imgs is None:
            raise SystemExit("--mode transfer requires --camels")
        # hold out CAMELS zooms purely for checkpoint selection, so no TNG
        # data influences training or model choice
        uz = np.unique(cam_zoom)
        rng = np.random.default_rng(args.seed)
        val_zooms = set(rng.choice(uz, max(1, int(0.15 * len(uz))),
                                   replace=False).tolist())
        vm = np.array([z in val_zooms for z in cam_zoom])
        print(f"CAMELS split for checkpoint selection: "
              f"{(~vm).sum()} train / {vm.sum()} val projections")
        cam_val_pred, state = fit(cam_imgs[~vm], cam_labels[~vm],
                                  cam_imgs[vm], cam_labels[vm], args, dev)
        # CAMELS-internal score: separates "the proxy label is unlearnable"
        # from "it is learnable but does not transfer to TNG"
        print(f"CAMELS held-out R² (same-domain): "
              f"{r2(cam_val_pred, cam_labels[vm]):+.4f}  "
              f"(n={int(vm.sum())})")
        pred = predict(state, imgs, args, dev)
        report(f"{args.tag} (CAMELS-trained -> all TNG)", pred, labels,
               "zero TNG clusters seen in training")
        out = f"{args.out_dir}/transfer_{args.tag}.npz"
        np.savez(out, y=labels, yhat=pred, halo=halo)
        print(f"saved {out}")
        return

    # --- pretrain on CAMELS once, then fine-tune per TNG fold ---
    pre_state = None
    if args.mode == "finetune":
        if cam_imgs is None:
            raise SystemExit("--mode finetune requires --camels")
        uz = np.unique(cam_zoom)
        rng = np.random.default_rng(args.seed)
        val_zooms = set(rng.choice(uz, max(1, int(0.15 * len(uz))),
                                   replace=False).tolist())
        vm = np.array([z in val_zooms for z in cam_zoom])
        # pretraining touches no TNG data, so one shared pretrained model
        # is safe for every fold
        print(f"pretraining on CAMELS ({(~vm).sum()} train projections, "
              f"{args.pretrain_epochs} epochs)...")
        _, pre_state = fit(cam_imgs[~vm], cam_labels[~vm],
                           cam_imgs[vm], cam_labels[vm], args, dev,
                           epochs=args.pretrain_epochs)
        pre_pred = predict(pre_state, imgs, args, dev)
        print(f"  pretrained model on TNG before fine-tuning: "
              f"R²={r2(pre_pred, labels):+.4f}")

    gkf = GroupKFold(n_splits=args.n_folds)
    oof = np.full(len(labels), np.nan, dtype=np.float32)
    fold_r2 = []
    for fold, (tr, val) in enumerate(gkf.split(imgs, labels, groups=halo)):
        tr_imgs, tr_labels = imgs[tr], labels[tr]
        if args.mode == "pooled" and cam_imgs is not None:
            tr_imgs = np.concatenate([tr_imgs, cam_imgs], axis=0)
            tr_labels = np.concatenate([tr_labels, cam_labels], axis=0)
        pred, _ = fit(tr_imgs, tr_labels, imgs[val], labels[val], args, dev,
                      lr=args.finetune_lr if args.mode == "finetune" else None,
                      init_state=pre_state)
        oof[val] = pred
        fold_r2.append(r2(pred, labels[val]))
        print(f"fold {fold}: train {len(tr_imgs)} val {len(val)}  "
              f"R²={fold_r2[-1]:+.3f}")

    report(f"{args.tag} OOF ({args.n_folds}-fold, TNG held-out only)",
           oof, labels,
           f"per-fold {np.mean(fold_r2):+.3f} ± {np.std(fold_r2):.3f}")

    out = f"{args.out_dir}/oof_{args.tag}.npz"
    np.savez(out, y=labels, yhat=oof, halo=halo,
             fold_r2=np.array(fold_r2), camels_used=bool(args.camels),
             mode=args.mode)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset_128_r200.h5")
    p.add_argument("--labels",
                   default="TSC_Cutimages/TSC_eachhalo_snap99.hdf5")
    p.add_argument("--label-key", default="tsc_gyr")
    p.add_argument("--camels", default=None)
    p.add_argument("--camels-tsc-max", type=float, default=None,
                   help="drop CAMELS zooms above this TSC (match TNG range)")
    p.add_argument("--require-detected", action="store_true",
                   help="use only CAMELS zooms with a detected mass jump")
    p.add_argument("--camels-labels", default=None,
                   help="HDF5 of zoom_id/tsc_proxy replacing merger_tsc")
    p.add_argument("--camels-m200-min", type=float, default=None,
                   help="drop CAMELS zooms below this M200 in Msun")
    p.add_argument("--camels-tsc-clip", type=float, default=None,
                   help="clip (not drop) CAMELS TSC at this many Gyr")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--mode", default="pooled",
                   choices=["pooled", "transfer", "finetune"],
                   help="pooled: CAMELS mixed into each train fold; "
                        "transfer: CAMELS-trained model scored on all TNG; "
                        "finetune: CAMELS pretrain then per-fold TNG tuning")
    p.add_argument("--pretrain-epochs", type=int, default=80)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ch", type=int, default=48)
    p.add_argument("--huber-delta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="tng_only")
    p.add_argument("--out-dir", default="cnn_camels_128")
    main(p.parse_args())
