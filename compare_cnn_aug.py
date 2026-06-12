#!/usr/bin/env python3
"""Read all cnn_aug_out/preds_*.npz, group by tag prefix, print summary."""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def r2(pred: np.ndarray, true: np.ndarray,
       mask: np.ndarray | None = None) -> float:
    """R^2 of pred vs true, optionally restricted to a boolean mask.
    Returns NaN when fewer than 3 points or zero variance."""
    if mask is not None:
        pred, true = pred[mask], true[mask]
    if len(true) < 3 or np.var(true) == 0:
        return float("nan")
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def main(args: argparse.Namespace) -> None:
    pred_files = sorted(glob.glob(f"{args.dir}/preds_*.npz"))
    if not pred_files:
        print(f"no preds_*.npz in {args.dir}"); return

    # One row of stratified R^2 metrics per saved prediction file.
    rows: list[dict] = []
    for path in pred_files:
        data = np.load(path)
        tag = os.path.basename(path).replace("preds_", "").replace(".npz", "")
        true, pred = data["y"], data["yhat"]
        rows.append({
            "tag": tag, "n": len(true),
            "r2_all": r2(pred, true),
            "r2_recent": r2(pred, true, true <= 2.0),
            "r2_very_recent": r2(pred, true, true <= 1.0),
            "r2_late": r2(pred, true, true > 2.0),
        })

    # group rows that share a tag prefix before the per-seed "_sN" suffix
    by_prefix: dict[str, list[dict]] = {}
    for row in rows:
        prefix = row["tag"].rsplit("_s", 1)[0] if "_s" in row["tag"] else row["tag"]
        by_prefix.setdefault(prefix, []).append(row)

    print(f"\n{'group':12s} {'n_seeds':>8s} {'R^2 all':>16s} "
          f"{'R^2 recent (<=2)':>20s} {'R^2 very recent (<=1)':>24s} {'R^2 late (>2)':>18s}")
    print("-" * 100)
    for prefix in sorted(by_prefix):
        group_rows = by_prefix[prefix]
        metric_keys = ["r2_all", "r2_recent", "r2_very_recent", "r2_late"]
        means = {k: np.nanmean([row[k] for row in group_rows]) for k in metric_keys}
        stds = {k: np.nanstd([row[k] for row in group_rows]) for k in metric_keys}
        print(f"{prefix:12s} {len(group_rows):>8d}  "
              f"{means['r2_all']:+.3f}±{stds['r2_all']:.3f}     "
              f"{means['r2_recent']:+.3f}±{stds['r2_recent']:.3f}        "
              f"{means['r2_very_recent']:+.3f}±{stds['r2_very_recent']:.3f}           "
              f"{means['r2_late']:+.3f}±{stds['r2_late']:.3f}")

    # comparison plot: pool every seed's predictions per prefix
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    scatter_by_prefix: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for prefix, group_rows in by_prefix.items():
        true_all = np.concatenate(
            [np.load(f"{args.dir}/preds_{row['tag']}.npz")["y"] for row in group_rows])
        pred_all = np.concatenate(
            [np.load(f"{args.dir}/preds_{row['tag']}.npz")["yhat"] for row in group_rows])
        scatter_by_prefix[prefix] = (true_all, pred_all)

    for panel, (lo, hi, title) in enumerate([(0, 8, "all"), (0, 2, "TSC <= 2 (recent)"),
                                              (0, 1, "TSC <= 1 (very recent)")]):
        a = ax[panel]
        for prefix, (true_all, pred_all) in scatter_by_prefix.items():
            in_range = (true_all >= lo) & (true_all <= hi)
            a.scatter(true_all[in_range], pred_all[in_range], s=15, alpha=0.4, label=prefix)
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
