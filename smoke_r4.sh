#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 01:30:00
#SBATCH -J smoker4
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/smoke_r4_%j.out
# Review R4-A/R4-B smoke checks of the repaired trainer (1-epoch runs, new
# output names, nothing existing is touched).
cd /oscar/data/idellant/cluster-ml
PY="./venv/bin/python -u"
set -e
echo "======== 1. predict() under an enabled-grad caller, all three model types"
$PY - <<'EOF'
import numpy as np, torch
from train_cnn_mock import predict, group_folds
from train_cnn_pooled import PooledCNN, JointPooledCNN
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(True)
x = np.random.rand(6, 3, 128, 128).astype(np.float32)
s = np.random.rand(6, 1).astype(np.float32)
for name, m, kw in [("image", PooledCNN().to(dev), {}),
                    ("image+scalar", PooledCNN(n_scalar=1).to(dev), {"scalars": s}),
                    ("joint", JointPooledCNN().to(dev), {"xray": x})]:
    p = predict(m, x, dev, **kw)
    print(f"  {name:<13} predict ok, shape {p.shape}, finite {np.isfinite(p).all()}")
    m.train()
    b = torch.tensor(x[:, :, None]).to(dev)
    out = (m(b, torch.tensor(s).to(dev)) if name == "image+scalar"
           else m(b, torch.tensor(x[:, :, None]).to(dev)) if name == "joint" else m(b))
    out.sum().backward()
    g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print(f"  {name:<13} backward ok outside predict, grad norm-ish {g:.3g}")
g = np.repeat(np.arange(40), 3); g[:6] = 0
f1 = [tuple(va) for _, va in group_folds(g, 5)]
f2 = [tuple(va) for _, va in group_folds(g, 5)]
print(f"  group_folds deterministic: {f1 == f2}")
EOF
echo "======== 2. --mass-noise without --mass-seed must refuse"
if $PY train_cnn_mock.py --dataset injected_fix_phys_b3000_s9991.h5 --mass --mass-noise \
      --epochs 1 --out-prefix cnn_preds/smoke_r4_refuse 2>&1 | grep -q "needs --mass-seed"; then
  echo "  refused as intended"; else echo "  FAILED: did not refuse"; fi
echo "======== 3. manifest: same folds for snapshot-99 halos in a 99-only and a 99+91 run"
$PY make_fold_manifest.py --datasets injected_fix_phys_b3000_s9991.h5 --output folds_smoke.npz
$PY train_cnn_mock.py --dataset injected_fix_phys_b3000.h5 --fold-manifest folds_smoke.npz \
    --epochs 1 --seed 42 --out-prefix cnn_preds/smoke_r4_99only | grep -E "fold [0-9]/|OOF R2" | head -7
$PY train_cnn_mock.py --dataset injected_fix_phys_b3000_s9991.h5 --fold-manifest folds_smoke.npz \
    --mass --mass-noise --mass-seed 0 --epochs 1 --seed 43 \
    --out-prefix cnn_preds/smoke_r4_9991_mass | grep -E "fold [0-9]/|OOF R2" | head -7
$PY train_cnn_mock.py --dataset injected_fix_phys_b1000_s9991.h5 --xray-dataset xray_fix_s9991.h5 \
    --fold-manifest folds_smoke.npz --epochs 1 --seed 44 \
    --out-prefix cnn_preds/smoke_r4_joint | grep -E "fold [0-9]/|OOF R2" | head -7
$PY - <<'EOF'
import numpy as np
a = np.load("cnn_preds/smoke_r4_99only_preds.npz")
b = np.load("cnn_preds/smoke_r4_9991_mass_preds.npz")
c = np.load("cnn_preds/smoke_r4_joint_preds.npz")
def fold_by_group(d):
    R = len(d["oof_sample"]) // len(d["tsc"])
    g = np.repeat(d["group_id"], R); s = np.repeat(d["snap"], R)
    out = {}
    for gg, ss, ff in zip(g, s, d["fold_of"]):
        if ss == 99:
            out.setdefault(int(gg), set()).add(int(ff))
    return out
A, B, C = fold_by_group(a), fold_by_group(b), fold_by_group(c)
print(f"  no split halos: {all(len(v) == 1 for d in (A, B, C) for v in d.values())}")
common = sorted(set(A) & set(B) & set(C))
same = all(A[h] == B[h] == C[h] for h in common)
print(f"  snapshot-99 halos with identical folds across 99-only / 99+91 mass / joint: "
      f"{same} ({len(common)} halos)")
print(f"  source_rev recorded: {a['source_rev']}, {b['source_rev']}, {c['source_rev']}")
print(f"  mass inputs saved: {b['sim_logm_input'].shape}, mass_seed {int(b['mass_seed'])}")
EOF
echo "======== 4. gallery under the M1 guards"
$PY paper/scripts/make_figures.py xray_gallery 2>&1 | tail -3
