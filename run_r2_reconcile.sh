#!/bin/bash
#SBATCH -p batch -n 1 -c 2 --mem=8G -t 00:20:00
#SBATCH -J r2rec
#SBATCH -o logs/r2_reconcile_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u - <<'PY'
import numpy as np
import compare_syn as C
from sklearn.metrics import r2_score
print("realistic models, pseudo-TSC truth, snapshot 99:  pooled (per-seed cluster-mean R2) | 1v-5 | 1v-1")
for lab, pat in (("paper X-ray (fix_x)", "fix_x_s{}_preds.npz"),
                 ("E5-H X-ray (manifest, single-view selection)", "e5_h_s{}_preds.npz"),
                 ("label-expt X-ray, pseudo arm", "lbl_x_pseudo_s{}_preds.npz"),
                 ("paper radio 3 Mpc (fix_)", "fix_phys_b3000_sfimg_s{}_preds.npz"),
                 ("new-model radio 3 Mpc (syn_)", "syn_phys_b3000_sfimg_s{}_preds.npz"),
                 ("label-expt radio 3 Mpc, pseudo arm", "lbl_r_pseudo_s{}_preds.npz")):
    s = C.scores(C.load(pat))
    print(f"  {lab:<46} {s[0]:.3f} ± {s[1]:.3f} | {s[2]:.3f} | {s[3]:.3f}")
print("descriptive paired contrasts (different folds/selection), 1v-5:")
for lab, a, b in (("X-ray: label-expt pseudo - paper fix_x", "fix_x_s{}_preds.npz", "lbl_x_pseudo_s{}_preds.npz"),
                  ("radio 3 Mpc: label-expt pseudo - syn", "syn_phys_b3000_sfimg_s{}_preds.npz", "lbl_r_pseudo_s{}_preds.npz")):
    dv, iv, st = C.contrast(a, b, allowed=("fold_manifest", "select_view", "tsc_dir", "label_curves", "label",
                                           "loss", "contrastive_pretrain", "cl_epochs", "cl_temp", "cl_batch"))
    print(f"  {lab:<42} {dv:+.3f} [{iv[0]:+.3f}, {iv[1]:+.3f}]  ({st})")
print("clean (no forward model) X-ray, pooled CNN, per-seed R2:")
import glob
for pat in ("pooled_xray_scaled_inner_s{}.npz", "xray_scaled_inner_s{}.npz"):
    fs = sorted(glob.glob("cnn_preds/" + pat.format("*")))
    if fs:
        r = [r2_score(np.load(f)["label"], np.load(f)["oof_pred"]) for f in fs]
        print(f"  {pat}: {np.mean(r):.3f} ± {np.std(r, ddof=1):.3f} (n={len(r)})")
PY
