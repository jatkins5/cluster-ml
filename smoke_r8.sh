#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 01:00:00
#SBATCH -J smoker8
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/smoke_r8_%j.out
# Review round 8 implementation checks (fail fast):
#  1. per-view scalars: a single-view observation's prediction cannot change
#     when ANOTHER projection's scalars change (dataset + predict code paths)
#  2. join coverage of the optical features for the X-ray training rows
#  3. 1-epoch X-ray + structure + mass + redshift run on the shared manifest
#  4. pilot v2 (grouped inner tuning, radial-mixing control with its
#     constructed check, saved outputs)
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
PY="./venv/bin/python -u"
T=smoke_r8_tmp; rm -rf $T; mkdir -p $T

echo "======== 1. single-view invariance to other projections' scalars"
$PY - <<'EOF'
import numpy as np, torch
from torch.utils.data import DataLoader
from train_cnn_mock import ViewScalarCNN, MockDataset, predict
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
M, P, H, K, G = 6, 3, 128, 4, 2
x = np.random.rand(M, P, H, H).astype(np.float32)
vs = np.random.randn(M, P, K).astype(np.float32)
gs = np.random.randn(M, G).astype(np.float32)
y = np.random.rand(M).astype(np.float32)
m = ViewScalarCNN(K, G).to(dev).eval()
def single_view(vs_):
    ds = MockDataset(x, y, gs, view="single", view_scalars=vs_)   # expanded eval
    out = []
    with torch.no_grad():
        for im, _, sc, v, _ in DataLoader(ds, batch_size=8):
            out.append(m(im.to(dev), v.to(dev), sc.to(dev)).cpu().numpy())
    return np.concatenate(out).reshape(M, P)
def oof_single_path(vs_, pj):   # the trainer's test-time path
    return predict(m, np.repeat(x[:, pj:pj + 1], P, 1), dev, scalars=gs,
                   view_scalars=np.repeat(vs_[:, pj:pj + 1], P, 1))
a = single_view(vs); a0 = oof_single_path(vs, 0)
vs2 = vs.copy(); vs2[:, 1] += 5.0                      # change projection 1 only
b = single_view(vs2); b0 = oof_single_path(vs2, 0)
assert np.allclose(a[:, 0], b[:, 0]) and np.allclose(a[:, 2], b[:, 2]), "other-view leak"
assert np.allclose(a0, b0), "test-time path leaks other projections' scalars"
assert not np.allclose(a[:, 1], b[:, 1]), "own view scalars unused"
assert np.allclose(a[:, 0], a0, atol=1e-5), "dataset and test-time paths disagree"
pooled_a = predict(m, x, dev, scalars=gs, view_scalars=vs)
pooled_b = predict(m, x, dev, scalars=gs, view_scalars=vs2)
assert not np.allclose(pooled_a, pooled_b), "pooled view ignores view scalars"
print("  single-view predictions of projections 0 and 2 unchanged by projection 1's "
      "scalars (dataset and test-time paths agree); own scalars and pooled use them")
EOF

echo "======== 2. optical join coverage for the X-ray training rows"
$PY - <<'EOF'
import h5py, numpy as np, pandas as pd
v = pd.read_csv("optical_features_idealised.csv")
keys = set(zip(v.group_id, v.snap, v.proj))
assert len(keys) == len(v), "duplicate keys"
with h5py.File("xray_fix_s9991.h5") as f:
    g, s = f["mock/group_id"][:], f["mock/snap"][:]
need = {(int(a), int(b), p) for a, b in zip(g, s) for p in range(3)}
miss = need - keys
assert not miss, f"{len(miss)} missing, e.g. {sorted(miss)[:3]}"
print(f"  all {len(need)} (parent, snapshot, projection) keys of xray_fix_s9991 present")
EOF

echo "======== 3. 1-epoch X-ray + structure + mass + redshift (shared manifest)"
$PY train_cnn_mock.py --dataset xray_fix_s9991.h5 --fold-manifest folds_e0.npz \
    --train-view pooled --select-view single --mass --mass-noise --mass-seed 0 \
    --redshift-scalar --view-scalars optical_features_idealised.csv \
    --epochs 1 --seed 42 --out-prefix $T/xopt | grep -E "global scalar|per-view|fold [0-9]/|OOF R2"
$PY - <<'EOF'
import numpy as np
d = np.load("smoke_r8_tmp/xopt_preds.npz", allow_pickle=True)
assert int(d["n_view"]) == 4 and int(d["n_scalar"]) == 2, (d["n_view"], d["n_scalar"])
assert np.isfinite(d["oof_single"]).all() and np.isfinite(d["oof_sample"]).all()
assert len(d["obs_pred"]) == 0
print(f"  saved: n_view {int(d['n_view'])} {list(d['view_cols'])}, n_scalar {int(d['n_scalar'])}, "
      f"finite predictions, real-target predictions disabled")
EOF

echo "======== 4. pilot v2"
$PY train_optical_pilot.py
echo "ALL ROUND-8 SMOKE CHECKS PASSED"
