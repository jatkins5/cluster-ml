"""Mass-only reference (images zeroed) on snapshots 99+91, scored on the
snapshot-99 clusters, to judge what the fixed radio images add beyond mass."""
import glob
import numpy as np
from sklearn.metrics import r2_score
r = []
for p in sorted(glob.glob("cnn_preds/s9991_massonly_s*_preds.npz")):
    d = np.load(p); m = d["snap"] == 99
    r.append(r2_score(d["tsc"][m], d["oof_cluster"][m]))
print(f"mass only (99+91 training, snap-99 clusters): R2 {np.mean(r):.3f} +- {np.std(r, ddof=1):.3f} (n={len(r)} seeds)")
