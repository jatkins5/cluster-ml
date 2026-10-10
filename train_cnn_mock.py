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
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from forward_model_lotss import target_key
from train_cnn_pooled import JointPooledCNN, PooledCNN, augment

# Provenance (review R4-A): the code revision that produced each result,
# marked "+dirty" if this file differs from that commit.
def _source_rev():
    import os
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "--", "train_cnn_mock.py"],
                               cwd=here).returncode != 0
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


SOURCE_REV = _source_rev()


class MockDataset(Dataset):
    """One item per (cluster, realization); each carries its 3 projections."""

    def __init__(self, images, labels, scalars=None, train=False, xray=None):
        self.images = images          # (M, 3, H, W)
        self.labels = labels
        self.scalars = scalars        # (M, n_scalar) or None
        self.xray = xray              # (M, 3, H, W) or None: joint model
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        imgs = self.images[i].copy()
        xr = None if self.xray is None else self.xray[i].copy()
        if self.train:
            # Same transform for a projection's radio and X-ray views.
            ks = np.random.randint(8, size=imgs.shape[0])
            imgs = np.stack([augment(imgs[p], ks[p])
                             for p in range(imgs.shape[0])])
            if xr is not None:
                xr = np.stack([augment(xr[p], ks[p])
                               for p in range(xr.shape[0])])
        s = (torch.zeros(0) if self.scalars is None
             else torch.tensor(self.scalars[i], dtype=torch.float32))
        xt = (torch.zeros(0) if xr is None
              else torch.tensor(xr[:, None], dtype=torch.float32))
        return (torch.tensor(imgs[:, None], dtype=torch.float32), xt, s,
                torch.tensor(self.labels[i], dtype=torch.float32))


def run_fold(tr_x, tr_y, sel_x, sel_y, epochs, batch_size, device, delta,
             tr_s=None, sel_s=None, n_scalar=0, tr_xr=None, sel_xr=None):
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
    joint = tr_xr is not None
    if joint and n_scalar:
        raise ValueError("joint radio+X-ray model does not take scalars")
    model = (JointPooledCNN() if joint
             else PooledCNN(n_scalar=n_scalar)).to(device)
    fwd = (lambda im, xr, sc: model(im, xr)) if joint else \
          (lambda im, xr, sc: model(im, sc))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.HuberLoss(delta=delta)
    tl = DataLoader(MockDataset(tr_x, tr_y, tr_s, train=True, xray=tr_xr),
                    batch_size=batch_size,
                    shuffle=True, num_workers=2, drop_last=True)
    sl = (None if sel_x is None
          else DataLoader(MockDataset(sel_x, sel_y, sel_s, xray=sel_xr),
                          batch_size=batch_size))
    best, best_state, best_ep = -np.inf, None, epochs
    for ep in range(epochs):
        model.train()
        for imgs, xr, sc, y in tl:
            imgs, y = imgs.to(device), y.to(device)
            sc = sc.to(device) if n_scalar else None
            xr = xr.to(device) if joint else None
            opt.zero_grad()
            loss = crit(fwd(imgs, xr, sc), y)
            loss.backward()
            opt.step()
        sched.step()
        if sl is None:
            continue
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, xr, sc, _ in sl:
                sc = sc.to(device) if n_scalar else None
                xr = xr.to(device) if joint else None
                preds.append(fwd(imgs.to(device), xr, sc).cpu().numpy())
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


def group_folds(groups, n_splits):
    """GroupKFold's balanced greedy assignment, made hardware-independent.

    sklearn's GroupKFold orders groups with np.argsort on group sizes; with
    the many ties here (3 or 6 samples per group) numpy's SIMD sort breaks
    them differently on different CPU families, so the folds depended on
    which GPU node a run landed on (found in review round 4). Here ties are
    broken by group id with a stable lexsort, so every run, seed and node
    gets the same partition."""
    ug, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.lexsort((ug, -cnt))                # size desc, then group id
    load = np.zeros(n_splits)
    fold_of_group = np.empty(len(ug), dtype=int)
    for g in order:
        f = int(np.argmin(load))
        fold_of_group[g] = f
        load[f] += cnt[g]
    fold = fold_of_group[inv]
    idx = np.arange(len(groups))
    for f in range(n_splits):
        yield idx[fold != f], idx[fold == f]


@torch.no_grad()
def predict(model, x, device, scalars=None, batch_size=32, xray=None):
    if len(x) == 0:
        return np.zeros(0, dtype=np.float32)
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        b = torch.tensor(x[i:i + batch_size][:, :, None],
                         dtype=torch.float32).to(device)
        s = (None if scalars is None else
             torch.tensor(scalars[i:i + batch_size],
                          dtype=torch.float32).to(device))
        if xray is not None:
            xb = torch.tensor(xray[i:i + batch_size][:, :, None],
                              dtype=torch.float32).to(device)
            out.append(model(b, xb).cpu().numpy())
        else:
            out.append(model(b, s).cpu().numpy())
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
    ap.add_argument("--mass", action="store_true",
                    help="condition on log10 M500c -- the sim's true mass "
                         "while training, the LoVoCCS weak-lensing mass at "
                         "inference -- so the image only has to supply what "
                         "mass does not already explain (checklist B2).")
    ap.add_argument("--wl-masses", default="lovoccs_wl_masses.csv")
    ap.add_argument("--catalog", default="Radio_Data/TNG-Cluster_Catalog.hdf5")
    ap.add_argument("--fold-manifest", default=None,
                    help="npz from make_fold_manifest.py: fixed parent-halo outer "
                         "folds and inner-selection halos shared across arms")
    ap.add_argument("--mass-seed", type=int, default=None,
                    help="seed for the simulated mass-measurement noise, fixed "
                         "across training seeds and comparison arms")
    ap.add_argument("--mass-noise", action="store_true",
                    help="perturb training masses by the per-cluster error "
                         "actually measured for the real sample (median 25%% "
                         "on M200c). Without it the model learns to trust a "
                         "mass more precisely than the real ones deserve.")
    ap.add_argument("--no-image", action="store_true",
                    help="zero the images: the mass-only baseline, on the "
                         "same folds, protocol and architecture.")
    ap.add_argument("--xray-dataset", default=None,
                    help="realistic X-ray mocks (build_xray_realistic.py) to "
                         "pair with the radio mocks in a joint model")
    args = ap.parse_args()
    if args.xray_dataset and (args.mass or args.no_image):
        raise SystemExit("--xray-dataset does not combine with --mass/--no-image")
    if args.mass_noise and args.mass_seed is None:
        raise SystemExit("--mass-noise needs --mass-seed: the simulated mass "
                         "measurement must be fixed across seeds and arms")
    if args.no_image and not args.mass:
        raise SystemExit("--no-image without --mass leaves no input at all")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with h5py.File(args.dataset, "r") as f:
        mock = f["mock/images"][:]            # (N, R, 3, H, W)
        halo = f["mock/halo_id"][:]
        tsc = f["mock/pseudo_tsc"][:]
        # Multi-snapshot files (merge_snapshots.py): halo_id is a unique row
        # id, group_id the z=0 cluster that folds must keep together, and
        # each row's own-snapshot M500 is stored with it.
        group_id = f["mock/group_id"][:] if "mock/group_id" in f else halo
        snap = f["mock/snap"][:] if "mock/snap" in f else np.full(len(halo), 99)
        own_logm = f["mock/log_m500"][:] if "mock/log_m500" in f else None
        if "obs" not in f:
            # Sim-only datasets (e.g. the realistic-depth X-ray mocks, for
            # which no real cutouts exist yet): score the mocks, skip the
            # real-data predictions.
            H0 = mock.shape[-1]
            obs = np.zeros((0, H0, H0), dtype=np.float32)
            obs_name, obs_z = [], np.zeros(0)
        else:
            obs = f["obs/images"][:]          # (n_obs, H, W)
            obs_name = [s.decode() if isinstance(s, bytes) else str(s)
                        for s in f["obs/name"][:]]
            obs_z = f["obs/z"][:]
        if "obs" not in f:
            obs_key = []
        elif "obs/key" in f:
            obs_key = [s.decode() if isinstance(s, bytes) else str(s)
                       for s in f["obs/key"][:]]
        else:   # datasets built before the key was stored
            obs_key = [target_key(n) for n in obs_name]
    N, R, P, H, _ = mock.shape
    print(f"mock {mock.shape}, obs {obs.shape}, device {device}")
    if own_logm is not None:
        logm = own_logm.astype(np.float64)
    else:
        with h5py.File(args.catalog, "r") as f:
            cmap = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
        logm = np.array([cmap[h] for h in halo], dtype=np.float64)
    if len(np.unique(snap)) > 1:
        print("  snapshots: " + ", ".join(f"{s}: {(snap == s).sum()} rows"
                                         for s in np.unique(snap))
              + f"; {len(np.unique(group_id))} distinct clusters")

    # Flatten realizations into samples, keeping cluster identity for grouping
    # so no cluster is split across folds.
    x = mock.reshape(N * R, P, H, H)
    y = np.repeat(tsc, R).astype(np.float32)
    groups = np.repeat(group_id, R)

    # The pooled model expects 3 views; a real cutout has one. Repeating it
    # three times leaves the mean-pool unchanged, so the obs path is a plain
    # single-image forward pass through the same weights.
    obs_x = np.repeat(obs[:, None], P, axis=1)

    xr_all = obs_xr = None
    if args.xray_dataset:
        with h5py.File(args.xray_dataset, "r") as f:
            xm = f["mock/images"][:]
            xh = f["mock/halo_id"][:]
        order = {h: i for i, h in enumerate(xh)}
        missing = [h for h in halo if h not in order]
        if missing:
            raise SystemExit(f"{len(missing)} radio clusters have no X-ray mock")
        xm = xm[[order[h] for h in halo]]          # radio order
        if xm.shape[1] < R:
            raise SystemExit(f"X-ray has {xm.shape[1]} realizations, radio {R}")
        # Realization r of the radio mock is paired with realization r of the
        # X-ray mock: independent noise draws of the same cluster.
        xr_all = xm[:, :R].reshape(N * R, P, xm.shape[-2], xm.shape[-1])
        print(f"joint model: radio {x.shape} + X-ray {xr_all.shape} "
              f"from {args.xray_dataset}")
        # Real inference needs both modalities for the same cluster: keep the
        # radio targets that also have processed Chandra data, matched by the
        # normalised name key, and pair their cutouts.
        xobs = None
        with h5py.File(args.xray_dataset, "r") as f:
            if "obs" in f:
                xk = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in f["obs/key"][:]]
                xobs = dict(zip(xk, f["obs/images"][:]))
        if xobs is None:
            print("  X-ray set has no real observations: mocks only")
            obs_x = obs_x[:0]
            obs_name, obs_z, obs_xr = [], obs_z[:0], None
        else:
            both = [i for i, k in enumerate(obs_key) if k in xobs]
            obs_x = obs_x[both]
            obs_name = [obs_name[i] for i in both]
            obs_z = obs_z[both]
            obs_xr = np.stack([np.repeat(xobs[obs_key[i]][None], P, axis=0)
                               for i in both]) if both else None
            print(f"  real clusters with both radio and X-ray: {len(both)} "
                  f"({', '.join(obs_name)})")

    # ---- mass conditioning (checklist B2) ----
    sim_s = obs_s = None
    n_scalar = 0
    rng = np.random.default_rng(args.seed)
    # Review round 3: the simulated mass MEASUREMENT must not change with the
    # training seed -- a real target has one measured mass shared by every
    # ensemble member. --mass-seed fixes the measurement draw independently
    # of model initialisation (legacy behaviour when unset).
    mrng = (np.random.default_rng(args.mass_seed) if args.mass_seed is not None
            else rng)
    if args.mass:
        n_scalar = 1
        sim_logm = logm.copy()

        wl = pd.read_csv(args.wl_masses)
        wlm = dict(zip(wl["key"], wl["log_m500c"]))
        wle = dict(zip(wl["key"], wl["frac_err"]))
        keep = np.array([k in wlm for k in obs_key])
        dropped = [n for n, kp in zip(obs_name, keep) if not kp]
        if dropped:
            print(f"  no weak-lensing mass for {len(dropped)}: "
                  f"{', '.join(dropped)}")
        obs_x = obs_x[keep]
        obs_name = [n for n, kp in zip(obs_name, keep) if kp]
        obs_z = obs_z[keep]
        obs_logm = np.array([wlm[k] for k, kp in zip(obs_key, keep) if kp])
        # The fit reports a fractional error on M200c; d(log10 M) = f/ln10.
        err_dex = np.array([wle[k] for k, kp in zip(obs_key, keep) if kp]) \
            / np.log(10.0)
        print(f"  conditioning on mass: {len(obs_logm)} real targets, "
              f"log M500c {obs_logm.min():.2f}-{obs_logm.max():.2f}, "
              f"median error {np.median(err_dex):.3f} dex")

        if args.mass_noise:
            # One draw per cluster, fixed for the run: the point is to blur
            # the mass the model is given, not to augment it every epoch.
            sig = mrng.choice(err_dex, size=N)
            sim_logm = sim_logm + mrng.normal(0.0, sig)
            print(f"  training masses blurred by {sig.mean():.3f} dex "
                  f"on average")
        sim_s = np.repeat(sim_logm, R)[:, None].astype(np.float32)
        obs_s = obs_logm[:, None].astype(np.float32)

    if args.no_image:
        x[...] = 0.0
        obs_x = np.zeros_like(obs_x)
        print("  images zeroed: this is the mass-only baseline")

    oof = np.full(N * R, np.nan, dtype=np.float32)
    # Review F3: the score real data actually get -- ONE projection, repeated
    # across the view axis exactly as a real cutout is -- for every held-out
    # sample and each projection, plus the outer fold of every sample, so
    # equal-size held-out ensembles can be formed across seeds (group_folds
    # folds do not depend on the seed).
    oof_single = np.full((N * R, P), np.nan, dtype=np.float32)
    fold_of = np.full(N * R, -1, dtype=np.int16)
    obs_preds = []
    manifest = None
    if args.fold_manifest:
        # review R4-B: persisted parent-halo folds AND inner-selection halos,
        # so arms on different datasets hold out the same halos
        mf = np.load(args.fold_manifest)
        mpos = {h: i for i, h in enumerate(mf["group_id"])}
        missing = sorted(set(np.unique(groups)) - set(mpos))
        if missing:
            raise SystemExit(f"{len(missing)} parent halos not in {args.fold_manifest}")
        if int(mf["folds"]) != args.folds:
            raise SystemExit(f"manifest has {int(mf['folds'])} folds, run asks {args.folds}")
        gi = np.array([mpos[g] for g in groups])
        manifest = (mf["fold"][gi], mf["inner_sel"][:, gi])
        idx_all = np.arange(len(groups))
        splits = [(idx_all[manifest[0] != f], idx_all[manifest[0] == f])
                  for f in range(args.folds)]
    else:
        splits = list(group_folds(groups, args.folds))
    for k, (tr, va) in enumerate(splits, 1):
        if args.select == "inner" and manifest is not None:
            is_sel = manifest[1][k - 1][tr]
            fit, sel = tr[~is_sel], tr[is_sel]
        elif args.select == "inner":
            # Grouped, so every realization of a cluster stays on one side.
            gss = GroupShuffleSplit(n_splits=1, test_size=args.inner_frac,
                                    random_state=args.seed + k)
            rel_fit, rel_sel = next(gss.split(tr, groups=groups[tr]))
            fit, sel = tr[rel_fit], tr[rel_sel]
        else:
            fit, sel = tr, None
        print(f"  fold {k}/{args.folds}: train {len(fit)}, "
              f"sel {0 if sel is None else len(sel)}, val {len(va)}")
        # Standardize the scalar on the training fold only, and put the real
        # masses on that same scale -- a shift between the two would show up
        # as a systematic offset in the real predictions.
        if n_scalar:
            mu, sd = sim_s[fit].mean(), sim_s[fit].std() + 1e-8
            f_s, v_s = (sim_s[fit] - mu) / sd, (sim_s[va] - mu) / sd
            s_s = None if sel is None else (sim_s[sel] - mu) / sd
            o_s = (obs_s - mu) / sd
        else:
            f_s = v_s = s_s = o_s = None
        model, best, best_ep = run_fold(
            x[fit], y[fit], None if sel is None else x[sel],
            None if sel is None else y[sel], args.epochs, args.batch_size,
            device, args.huber_delta, tr_s=f_s, sel_s=s_s, n_scalar=n_scalar,
            tr_xr=None if xr_all is None else xr_all[fit],
            sel_xr=None if (xr_all is None or sel is None) else xr_all[sel])
        oof[va] = predict(model, x[va], device, scalars=v_s,
                          xray=None if xr_all is None else xr_all[va])
        fold_of[va] = k - 1
        for pj in range(P):
            xs = np.repeat(x[va][:, pj:pj + 1], P, axis=1)
            xrs = (None if xr_all is None
                   else np.repeat(xr_all[va][:, pj:pj + 1], P, axis=1))
            oof_single[va, pj] = predict(model, xs, device, scalars=v_s, xray=xrs)
        obs_preds.append(predict(model, obs_x, device, scalars=o_s,
                                 xray=obs_xr if xr_all is not None else None))
        print(f"  fold {k}: checkpoint epoch {best_ep}/{args.epochs}, "
              f"sel R2 {best:+.3f}  ->  outer val R2 "
              f"{r2_score(y[va], oof[va]):+.3f}")

    # Score per cluster: average the realizations, which is what a real
    # analysis would do with repeat observations of one cluster.
    oof_cluster = oof.reshape(N, R).mean(axis=1)
    r2 = r2_score(tsc, oof_cluster)
    mae = mean_absolute_error(tsc, oof_cluster)
    print(f"\nmock OOF R2 (per realization): {r2_score(y, oof):+.3f}")
    ys = np.repeat(y[:, None], P, axis=1)
    print(f"mock OOF R2 (ONE view, one realization -- as real data): "
          f"{r2_score(ys.ravel(), oof_single.ravel()):+.3f}")
    print(f"mock OOF R2 (cluster-mean)  : {r2:+.3f}   MAE {mae:.3f} Gyr")
    if len(np.unique(snap)) > 1:
        for s in np.unique(snap):
            m = snap == s
            print(f"  snapshot {s}: R2 (cluster-mean) "
                  f"{r2_score(tsc[m], oof_cluster[m]):+.3f}  n={m.sum()}")

    # The confound this experiment exists to measure: how much of the
    # prediction is mass, and does anything survive at fixed mass?
    rho = lambda a, b: stats.spearmanr(a, b)[0]
    rk = lambda v: stats.rankdata(v)
    def resid(a, b):
        A = np.column_stack([b, np.ones_like(b)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    partial = stats.pearsonr(resid(rk(oof_cluster), rk(logm)),
                             resid(rk(tsc), rk(logm)))[0]
    print(f"  rho(pred, M500) {rho(oof_cluster, logm):+.3f}   "
          f"rho(pred, TSC) {rho(oof_cluster, tsc):+.3f}   "
          f"partial(pred, TSC | M500) {partial:+.3f}")
    q = np.quantile(logm, [1 / 3, 2 / 3])
    for lab, b in [("low", logm <= q[0]),
                   ("mid", (logm > q[0]) & (logm <= q[1])),
                   ("high", logm > q[1])]:
        print(f"    mass tercile {lab:<5} n={b.sum():3d}  "
              f"R2 {r2_score(tsc[b], oof_cluster[b]):+.3f}")

    obs_pred = np.mean(obs_preds, axis=0)
    if len(obs_pred) == 0:
        print("\n(no real observations in this dataset)")
    print(f"\nreal LoTSS predictions from the {args.folds}-model ensemble:")
    print(f"{'target':<16}{'z':>8}{'pred TSC [Gyr]':>16}")
    for n, z, p in zip(obs_name, obs_z, obs_pred):
        print(f"{n:<16}{z:>8.4f}{p:>16.2f}")
    print(f"\n{'':<16}{'mock (train)':>16}{'real LoTSS':>14}")
    for lab, fn in [("median", np.median), ("mean", np.mean),
                    ("std", np.std), ("min", np.min), ("max", np.max)]:
        real = fn(obs_pred) if len(obs_pred) else np.nan
        print(f"{lab:<16}{fn(tsc):>16.3f}{real:>14.3f}")

    np.savez(f"{args.out_prefix}_preds.npz", oof_cluster=oof_cluster,
             tsc=tsc, halo_id=halo, group_id=group_id, snap=snap,
             oof_sample=oof, oof_single=oof_single, fold_of=fold_of,
             y_sample=y, obs_pred_folds=np.array(obs_preds),
             obs_pred=obs_pred,
             obs_name=np.array(obs_name), obs_z=obs_z,
             select=args.select, seed=args.seed, dataset=args.dataset,
             mass=args.mass, no_image=args.no_image,
             mass_noise=args.mass_noise, mass_seed=-1 if args.mass_seed is None else args.mass_seed,
             fold_manifest=str(args.fold_manifest or ""), source_rev=SOURCE_REV,
             sim_logm_input=(sim_s[::R, 0] if sim_s is not None else np.zeros(0)),
             obs_logm=(obs_s[:, 0] if obs_s is not None else np.zeros(0)),
             sim_logm=logm)
    print(f"\nsaved -> {args.out_prefix}_preds.npz")


if __name__ == "__main__":
    main()
