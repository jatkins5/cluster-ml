#!/usr/bin/env python3
"""Read all cnn_aug_out/preds_*.npz, group by tag prefix, print summary."""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def r2(p, y, mask=None):
    if mask is not None:
        p, y = p[mask], y[mask]
    if len(y) < 3 or np.var(y) == 0:
        return float("nan")
    return 1.0 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def main(args):
    files = sorted(glob.glob(f"{args.dir}/preds_*.npz"))
    if not files:
        print(f"no preds_*.npz in {args.dir}"); return

    rows = []
    for f in files:
        d = np.load(f)
        tag = os.path.basename(f).replace("preds_", "").replace(".npz", "")
        y, p = d["y"], d["yhat"]
        rows.append({
            "tag": tag, "n": len(y),
            "r2_all": r2(p, y),
            "r2_recent": r2(p, y, y <= 2.0),
            "r2_very_recent": r2(p, y, y <= 1.0),
            "r2_late": r2(p, y, y > 2.0),
        })

    # group by prefix before the _sN suffix
    by_prefix = {}
    for r in rows:
        prefix = r["tag"].rsplit("_s", 1)[0] if "_s" in r["tag"] else r["tag"]
        by_prefix.setdefault(prefix, []).append(r)

    print(f"\n{'group':12s} {'n_seeds':>8s} {'R^2 all':>16s} "
          f"{'R^2 recent (<=2)':>20s} {'R^2 very recent (<=1)':>24s} {'R^2 late (>2)':>18s}")
    print("-" * 100)
    for prefix in sorted(by_prefix):
        rs = by_prefix[prefix]
        keys = ["r2_all", "r2_recent", "r2_very_recent", "r2_late"]
        means = {k: np.nanmean([r[k] for r in rs]) for k in keys}
        stds = {k: np.nanstd([r[k] for r in rs]) for k in keys}
        print(f"{prefix:12s} {len(rs):>8d}  "
              f"{means['r2_all']:+.3f}±{stds['r2_all']:.3f}     "
              f"{means['r2_recent']:+.3f}±{stds['r2_recent']:.3f}        "
              f"{means['r2_very_recent']:+.3f}±{stds['r2_very_recent']:.3f}           "
              f"{means['r2_late']:+.3f}±{stds['r2_late']:.3f}")

    # comparison plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    pred_cmap = {}
    for prefix, rs in by_prefix.items():
        ys = np.concatenate([np.load(f"{args.dir}/preds_{r['tag']}.npz")["y"] for r in rs])
        ps = np.concatenate([np.load(f"{args.dir}/preds_{r['tag']}.npz")["yhat"] for r in rs])
        pred_cmap[prefix] = (ys, ps)

    for k, (lo, hi, title) in enumerate([(0, 8, "all"), (0, 2, "TSC <= 2 (recent)"),
                                          (0, 1, "TSC <= 1 (very recent)")]):
        a = ax[k]
        for prefix, (ys, ps) in pred_cmap.items():
            m = (ys >= lo) & (ys <= hi)
            a.scatter(ys[m], ps[m], s=15, alpha=0.4, label=prefix)
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_xlabel("true merger-TSC (Gyr)")
        a.set_ylabel("predicted")
        a.set_title(title)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.dir}/comparison.png", dpi=120)
    print(f"saved {args.dir}/comparison.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="cnn_aug_out")
    main(p.parse_args())
