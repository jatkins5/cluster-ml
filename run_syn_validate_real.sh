#!/bin/bash
#SBATCH -p batch -n 1 -c 1 --mem=8G -t 00:15:00
#SBATCH -J synval
#SBATCH -o logs/syn_validate_real_%j.out
# Validators only (no bootstrap, no scientific analysis) on the real artifacts.
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u - <<'PY'
import compare_syn as C
def show(lab, a, b, kind="plain", matched=False, allowed=(), joint_x=None):
    da, db = C.load(a), C.load(b)
    C.same_rows(da, db, lab)
    st = C.split_status(da, db)
    if matched:
        C.check_matched(da, db, st, kind, lab)
    print(f"  {lab}: {C.fmt(st)}; {C.check_contract(da, db, allowed, lab, matched, joint_x)}")
show("MATCHED radio+mass vs mass only", "syn_massonly_s{}_preds.npz", "syn_imgmass_b3000_s{}_preds.npz",
     "mass", True, ("no_image",))
show("MATCHED joint 3 Mpc vs X-ray control", "runs/followup_20261103T180830Z/preds/xray_b3000_s{}_preds.npz",
     "syn_phys_b3000_joint_s{}_preds.npz", matched=True, allowed=("dataset", "xray_dataset"), joint_x="b_is_joint")
show("descriptive joint 1 Mpc vs fix_x", "fix_x_s{}_preds.npz", "syn_phys_b1000_joint_s{}_preds.npz",
     allowed=("dataset", "xray_dataset"))
for t in ("phys_b1000_sfimg", "phys_b3000_sfimg", "phys_b3000_joint"):
    show(f"descriptive old->new {t}", f"fix_{t}_s{{}}_preds.npz", f"syn_{t}_s{{}}_preds.npz",
         allowed=("dataset", "xray_dataset"))
show("descriptive old->new mass only", "r4_massonly_s{}_preds.npz", "syn_massonly_s{}_preds.npz",
     allowed=("dataset",))
for a, b in (("pooled_nh4_144_inner_s{}.npz", "pooled_syn_inner_s{}.npz"),
             ("pooled_s9991_inner_s{}.npz", "pooled_syn_s9991_inner_s{}.npz"),
             ("pooled_syn_inner_s{}.npz", "pooled_shape_syn_inner_s{}.npz"),
             ("pooled_syn_inner_s{}.npz", "joint_syn_clean_inner_s{}.npz")):
    C.clean_pair(a, b)
    print(f"  clean pair {a} vs {b}: {C.CLEAN_STATUS[(a, b)]}")
print("REAL-ARTIFACT VALIDATION PASSED")
PY
