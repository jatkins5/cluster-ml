"""Review round 4 analyses (CODEX_CLAUDE_REVIEW.md, Codex round 3).

0. Alignment: for every configuration, halo ids, snapshots, labels and
   saved outer folds must be identical across seeds before averaging.
1. Image beyond mass: mass-only vs image+mass on the same halos, seeds and
   fixed blurred-mass measurement (--mass-seed 0). Reported as "matched"
   only if every run of both arms used one fold partition, otherwise as
   DESCRIPTIVE (the round-4 runs predate the persisted fold manifest).
2. Joint vs X-ray on the same placements (1 Mpc joint and X-ray-only both
   use xray_fix_s9991.h5), single-view and pooled-view; same labelling.
3. View penalty at EQUAL ensemble size: pooled-view and single-view both as
   5-seed ensembles (and both as single models).
Uncertainty: paired halo bootstrap, resampling whole z=0 clusters (all their
snapshots, realizations and views together), 4000 draws; plus the paired
seed spread. Scored on snapshot-99 clusters throughout.
"""
import os

import numpy as np
from sklearn.metrics import r2_score

SEEDS = range(42, 47)
rng = np.random.default_rng(0)
NB = 4000
PARTS = {}
PRED_DIR = os.environ.get("PRED_DIR", "cnn_preds")


def load(pat):
    ds = {s: np.load(f"{PRED_DIR}/{pat.format(s)}", allow_pickle=True)
          for s in SEEDS if os.path.exists(f"{PRED_DIR}/{pat.format(s)}")}
    if sorted(ds) != list(SEEDS):
        raise SystemExit(f"{pat}: expected seeds {list(SEEDS)}, found {sorted(ds)}")
    d0 = next(iter(ds.values()))
    for s, d in ds.items():
        for k in ("halo_id", "snap", "tsc", "y_sample", "group_id"):
            if k in d0 and not np.array_equal(d[k], d0[k]):
                raise SystemExit(f"{pat}: seed {s} differs in {k}")
        # validity of averaging: every sample predicted out-of-fold in every run
        if (d["fold_of"] < 0).any():
            raise SystemExit(f"{pat}: seed {s} has samples without an outer fold")
        for k in ("oof_sample", "oof_single", "oof_cluster"):
            if not np.isfinite(d[k]).all():
                raise SystemExit(f"{pat}: seed {s} has non-finite {k}")
        # no parent halo may be split across outer folds
        R = len(d["oof_sample"]) // len(d["tsc"])
        g = np.repeat(d["group_id"], R)
        nf = {}
        for gg, ff in zip(g, d["fold_of"]):
            nf.setdefault(gg, set()).add(int(ff))
        split = [gg for gg, v in nf.items() if len(v) > 1]
        if split:
            raise SystemExit(f"{pat}: seed {s} splits {len(split)} halos across folds")
    # Folds of runs before the group_folds fix depended on the GPU node's CPU
    # (numpy SIMD sort tie-breaking). Each run's predictions are still OOF
    # under its own partition, so seed averages never use a model that saw
    # the sample; report how many distinct partitions were used.
    parts = {d["fold_of"].tobytes() for d in ds.values()}
    PARTS[pat] = len(parts)
    return ds


def sample_view(ds, mode):
    """(y, pred, group) per scored sample on snapshot-99 clusters.
    mode: '1view-ens', '1view-1', 'pool-ens', 'pool-1' (1 = mean over the
    per-seed scores is handled by the caller; here 1 returns a list)."""
    d0 = next(iter(ds.values()))
    N = len(d0["tsc"])
    R = len(d0["oof_sample"]) // N
    snap = np.repeat(d0["snap"], R)
    grp = np.repeat(d0["group_id"], R)
    m = snap == 99
    y = d0["y_sample"]
    if mode.startswith("1view"):
        P = d0["oof_single"].shape[1]
        Y = np.repeat(y[m][:, None], P, 1).ravel()
        G = np.repeat(grp[m][:, None], P, 1).ravel()
        preds = [d["oof_single"][m].ravel() for d in ds.values()]
    else:
        Y, G = y[m], grp[m]
        preds = [d["oof_sample"][m] for d in ds.values()]
    return Y, G, preds


def r2_ens(Y, preds):
    return r2_score(Y, np.mean(preds, axis=0))


def assert_same_mass(inputs):
    if not all(np.array_equal(inputs[0], q) for q in inputs[1:]):
        raise SystemExit("blurred mass inputs differ between runs: not a matched comparison")


def partitions(ds):
    return {d["fold_of"].tobytes() for d in ds.values()}


def paired(lab, a, b, mode):
    # a comparison is MATCHED only if every run of both arms used one
    # partition; otherwise it is reported, labelled descriptive
    status = ("matched" if len(partitions(a) | partitions(b)) == 1
              else "DESCRIPTIVE: arms/seeds used different fold partitions")
    Ya, Ga, pa = sample_view(a, mode)
    Yb, Gb, pb = sample_view(b, mode)
    assert np.array_equal(Ya, Yb) and np.array_equal(Ga, Gb), "arms not aligned"
    ea, eb = np.mean(pa, 0), np.mean(pb, 0)
    d = r2_score(Yb, eb) - r2_score(Ya, ea)
    ug = np.unique(Ga)
    idx = {g: np.where(Ga == g)[0] for g in ug}
    boots = []
    for _ in range(NB):
        sel = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        boots.append(r2_score(Yb[sel], eb[sel]) - r2_score(Ya[sel], ea[sel]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    seeds = sorted(set(a) & set(b))
    ka, kb = list(a), list(b)
    sd = [r2_score(Yb, pb[kb.index(t)]) - r2_score(Ya, pa[ka.index(t)]) for t in seeds]
    print(f"  {lab:<44} {mode:<10} A {r2_score(Ya, ea):.3f}  B {r2_score(Yb, eb):.3f}  "
          f"B-A {d:+.3f}  halo-bootstrap 95% [{lo:+.3f}, {hi:+.3f}]  "
          f"per-seed single-model B-A {np.mean(sd):+.3f} ± {np.std(sd, ddof=1):.3f}  [{status}]")


def main():
    print("======== 0. alignment checks")
    cfg = {
        "mass only (fixed measurement)": "r4_massonly_s{}_preds.npz",
        "image+mass 3 Mpc (fixed measurement)": "r4_imgmass_b3000_s{}_preds.npz",
        "X-ray, real count model": "fix_x_s{}_preds.npz",
        "joint, physical 1 Mpc": "fix_phys_b1000_joint_s{}_preds.npz",
        "radio, physical 3 Mpc": "fix_phys_b3000_sfimg_s{}_preds.npz",
    }
    L = {}
    for lab, pat in cfg.items():
        L[lab] = load(pat)
        print(f"  {lab:<40} {len(L[lab])} seeds aligned (halo, snap, label); all OOF; "
              f"{PARTS[pat]} distinct fold partition(s)")
    # the matched arms must also share the fixed mass measurement
    mi = [d["sim_logm_input"] for d in L["mass only (fixed measurement)"].values()] + \
         [d["sim_logm_input"] for d in L["image+mass 3 Mpc (fixed measurement)"].values()]
    assert_same_mass(mi)
    print("  blurred mass inputs identical across all 10 runs")

    print("\n======== 1. image beyond mass (A = mass only, B = image + mass)")
    for mode in ("1view-ens", "pool-ens"):
        paired("3 Mpc physical radio", L["mass only (fixed measurement)"],
               L["image+mass 3 Mpc (fixed measurement)"], mode)

    print("\n======== 2. joint vs X-ray, same placements (A = X-ray, B = joint 1 Mpc)")
    for mode in ("1view-ens", "pool-ens"):
        paired("1 Mpc physical radio + X-ray", L["X-ray, real count model"],
               L["joint, physical 1 Mpc"], mode)

    print("\n======== 3. view penalty at equal ensemble size")
    for lab in ("radio, physical 3 Mpc", "X-ray, real count model", "joint, physical 1 Mpc"):
        out = []
        for mode in ("pool-ens", "1view-ens"):
            Y, G, p = sample_view(L[lab], mode)
            out.append((r2_ens(Y, p), np.mean([r2_score(Y, q) for q in p])))
        print(f"  {lab:<28} pooled-view: 5-model {out[0][0]:.3f}, 1-model {out[0][1]:.3f}  |  "
              f"single-view: 5-model {out[1][0]:.3f}, 1-model {out[1][1]:.3f}")


if __name__ == "__main__":
    main()
