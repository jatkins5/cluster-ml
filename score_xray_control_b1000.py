#!/usr/bin/env python3
"""Frozen scoring for the matched 1 Mpc X-ray control (review round 59).

Primary endpoint (fixed before the run): joint minus X-ray single-observation
5-model ensemble R2 on snapshot 99 (352 parents x 3 realisations x 3
projections; predictions averaged over the five seed models per observation),
stored pseudo-TSC target, 4000-draw paired parent bootstrap with the existing
fixed seed, all nine observations of a parent resampled together. Computed by
compare_syn.contrast(control, joint, matched=True, allowed=("dataset",
"xray_dataset"), joint_x="b_is_joint"), which enforces 5/5 outer-fold and
inner-selection agreement, identical rows, identical defining sources and
configuration, and the joint's X-ray input = the control dataset (path and
hash). The interval conditions on the fitted models. Secondary, labelled:
pooled-view mean and seed SD, mean single-model single-view R2, and Spearman
of the pooled-view prediction with M500, for both arms; the same contrast on
both snapshots. The preflight record must exist and have passed. Any failed
check stops the script; nothing is reported as matched.

  score_xray_control_b1000.py --out <file.json>
"""
import argparse
import json
import os
import sys

import compare_syn as CS

CTRL = "runs/xray_control_b1000/preds/xray_b1000_s{}_preds.npz"
JOINT = "syn_phys_b1000_joint_s{}_preds.npz"
PREFLIGHT = "runs/xray_control_b1000/preflight/verified.json"
KW = dict(matched=True, allowed=("dataset", "xray_dataset"), joint_x="b_is_joint")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    pf = json.load(open(PREFLIGHT))
    if pf.get("status") != "passed":
        raise SystemExit("preflight did not pass; not scoring")
    dv, iv, st = CS.contrast(CTRL, JOINT, **KW)
    dv2, iv2, _ = CS.contrast(CTRL, JOINT, snap99=False, **KW)
    sc, sj = CS.scores(CS.load(CTRL)), CS.scores(CS.load(JOINT))
    lab = ("pooled_mean", "pooled_sd", "single_view_5model", "single_view_1model_mean", "rho_pred_M")
    out = dict(
        primary=dict(endpoint="joint - X-ray control, single-observation 5-model ensemble R2, snapshot 99",
                     control=sc[2], joint=sj[2], delta=dv, lo=float(iv[0]), hi=float(iv[1]),
                     bootstrap="4000 paired parent draws, rng seed 0 (compare_syn.contrast)", status=st,
                     interval_note="conditional on the fitted models; no retraining uncertainty"),
        secondary=dict(control=dict(zip(lab, sc)), joint=dict(zip(lab, sj)),
                       both_snapshots=dict(delta=dv2, lo=float(iv2[0]), hi=float(iv2[1]))),
        inputs=dict(control=CTRL, joint=JOINT, preflight=PREFLIGHT),
        reading=("a positive difference with an interval spanning zero is unresolved, not a demonstrated "
                 "gain or equivalence"))
    with open(a.out, "x") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"matched 1 Mpc: X-ray control 1v-5 {sc[2]:.3f}, joint {sj[2]:.3f}; "
          f"joint - control {dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]   ({st})")
    print(f"  both snapshots {dv2:+.3f} [{iv2[0]:+.3f}, {iv2[1]:+.3f}]")
    print(f"  control pooled {sc[0]:.3f} +- {sc[1]:.3f}, 1v-1 {sc[3]:.3f}, rho(M) {sc[4]:+.2f}; "
          f"joint pooled {sj[0]:.3f} +- {sj[1]:.3f}, 1v-1 {sj[3]:.3f}, rho(M) {sj[4]:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
