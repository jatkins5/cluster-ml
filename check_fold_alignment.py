"""Are the saved outer folds identical across training seeds? The 5-model
held-out ensembles assume GroupKFold folds do not depend on the seed."""
import glob, numpy as np
for pat in ("r4_massonly", "r4_imgmass_b3000", "fix_x", "fix_phys_b1000_joint",
            "fix_phys_b3000_sfimg", "fix_phys_b3000_sfimgmassn"):
    fs = sorted(glob.glob(f"cnn_preds/{pat}_s*_preds.npz"))
    F = [np.load(f)["fold_of"] for f in fs]
    same = [np.array_equal(F[0], f) for f in F]
    diff = [int((F[0] != f).sum()) for f in F]
    print(f"{pat:<28} seeds {len(F)}  identical to first: {same}  differing samples: {diff}"
          f"  (-1 entries: {[int((f < 0).sum()) for f in F]})")
