#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 01:30:00
#SBATCH -J smokee0
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/smoke_e0_%j.out
# Fail-fast checks for the E0 implementation and the R4/R5 safeguards
# (review rounds 4-5). Any failed check or upstream command stops the job
# with a nonzero exit. 1-epoch runs, new output names only.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
PY="./venv/bin/python -u"
T=smoke_e0_tmp
rm -rf $T && mkdir -p $T

echo "======== 0. shared E0 manifest (all E0 arms reuse this one file)"
[ -f folds_e0.npz ] || $PY make_fold_manifest.py --datasets xray_fix_s9991.h5 \
    injected_fix_phys_b3000_s9991.h5 injected_fix_phys_b1000_s9991.h5 --output folds_e0.npz
sha256sum folds_e0.npz

echo "======== 1. comparison safeguards reject malformed inputs"
$PY - <<'EOF'
import os, numpy as np
os.environ["PRED_DIR"] = "smoke_e0_tmp"
import compare_round4 as c
c.PRED_DIR = "smoke_e0_tmp"
N, R, P = 6, 2, 3
def write(tag, seeds, nan=False, split=False):
    tsc = np.random.rand(N); ys = np.repeat(tsc, R)     # trainer's label contract
    for s in seeds:
        g = np.arange(N); fold = np.repeat(g % 3, R)
        if split:
            fold[1] = (fold[0] + 1) % 3          # halo 0's 2nd realization elsewhere
        single = np.random.rand(N * R, P).astype(np.float32)
        if nan:
            single[0, 0] = np.nan
        np.savez(f"smoke_e0_tmp/{tag}_s{s}_preds.npz", halo_id=g, group_id=g,
                 snap=np.full(N, 99), tsc=tsc, y_sample=ys,
                 fold_of=fold, oof_sample=np.random.rand(N * R),
                 oof_single=single, oof_cluster=np.random.rand(N))
write("good", range(42, 47)); write("missing", [42, 43, 44, 45])
write("nan", range(42, 47), nan=True); write("split", range(42, 47), split=True)
c.load("good_s{}_preds.npz")
for tag in ("missing", "nan", "split"):
    try:
        c.load(tag + "_s{}_preds.npz")
    except SystemExit as e:
        print(f"  {tag:<8} rejected: {e}")
    else:
        raise AssertionError(f"{tag} input was NOT rejected")
try:
    c.assert_same_mass([np.zeros(4), np.ones(4)])
except SystemExit as e:
    print(f"  unequal mass rejected: {e}")
else:
    raise AssertionError("unequal mass inputs were NOT rejected")
EOF

echo "======== 2. E0 single-view training/selection, 1 epoch each, shared manifest"
run () { $PY train_cnn_mock.py --fold-manifest folds_e0.npz --epochs 1 "$@" \
           | grep -E "fold [0-9]/|OOF R2" ; }
run --dataset xray_fix_s9991.h5 --train-view single --select-view single --seed 42 \
    --out-prefix $T/x_single
run --dataset xray_fix_s9991.h5 --train-view pooled --select-view single --seed 42 \
    --out-prefix $T/x_pooled
run --dataset injected_fix_phys_b1000_s9991.h5 --xray-dataset xray_fix_s9991.h5 \
    --train-view single --select-view single --seed 43 --out-prefix $T/joint_single
run --dataset injected_fix_phys_b3000_s9991.h5 --mass --mass-noise --mass-seed 0 \
    --train-view single --select-view single --seed 44 --out-prefix $T/mass_9991
run --dataset injected_fix_phys_b3000.h5 --mass --mass-noise --mass-seed 0 \
    --train-view single --select-view single --seed 45 --out-prefix $T/mass_99only

echo "======== 3. outputs satisfy the contract"
$PY - <<'EOF'
import json, numpy as np
T = "smoke_e0_tmp"
mf = np.load("folds_e0.npz")
for tag in ("x_single", "x_pooled", "joint_single", "mass_9991", "mass_99only"):
    d = np.load(f"{T}/{tag}_preds.npz", allow_pickle=True)
    assert np.isfinite(d["oof_single"]).all() and np.isfinite(d["oof_sample"]).all(), tag
    assert (d["fold_of"] >= 0).all(), tag
    # outer folds = the manifest's, per parent halo
    pos = {h: i for i, h in enumerate(mf["group_id"])}
    assert all(mf["fold"][pos[g]] == f for g, f in zip(d["sample_group"], d["fold_of"])), tag
    # saved inner-selection halos = the manifest's, per fold
    for k in range(int(mf["folds"])):
        got = set(d["sel_groups_flat"][d["sel_groups_fold"] == k].tolist())
        want = set(mf["group_id"][mf["inner_sel"][k]].tolist()) & set(d["sample_group"].tolist())
        want -= set(mf["group_id"][mf["fold"] == k].tolist())
        assert got == want, (tag, k)
    prov = json.loads(str(d["provenance"]))
    import hashlib
    assert prov["manifest_sha256"] == hashlib.sha256(open("folds_e0.npz", "rb").read()).hexdigest(), tag
    assert prov["sha256"]["train_cnn_mock.py"], tag
    assert not bool(d["provenance_changed_during_run"]), tag
    print(f"  {tag:<13} views {d['train_view']}/{d['select_view']}  folds+inner match manifest, "
          f"finite, provenance head {prov['head'][:8]} status {prov['status'].strip() or 'clean'!r}")
# keyed mass draws: snapshot-99 objects get the same measurement in both datasets
a, b = np.load(f"{T}/mass_9991_preds.npz"), np.load(f"{T}/mass_99only_preds.npz")
ma = dict(zip(zip(a["group_id"], a["snap"]), a["sim_logm_input"]))
mb = dict(zip(zip(b["group_id"], b["snap"]), b["sim_logm_input"]))
common = set(ma) & set(mb)
assert common and all(ma[k] == mb[k] for k in common), "mass draws depend on dataset rows"
print(f"  keyed mass draws identical for {len(common)} snapshot-99 objects across 99-only and 99+91")
EOF
echo "======== 4. E0 driver on 1-epoch arms (5 seeds each) and its rejections"
for S in 42 43 44 45 46; do
  for V in pooled single; do
    $PY train_cnn_mock.py --dataset xray_fix_s9991.h5 --fold-manifest folds_e0.npz \
        --train-view $V --select-view single --epochs 1 --seed $S \
        --out-prefix $T/e0d_${V}_s$S > /dev/null
  done
done
SHA=$(sha256sum folds_e0.npz | cut -d' ' -f1)
$PY compare_e0.py --pred-dir $T --a "e0d_pooled_s{}_preds.npz" --b "e0d_single_s{}_preds.npz" \
    --expect-manifest $SHA --boot 200
if $PY compare_e0.py --pred-dir $T --a "e0d_pooled_s{}_preds.npz" --b "e0d_single_s{}_preds.npz" \
    --expect-manifest 0000 --boot 10 > $T/wrong_sha.txt 2>&1; then
  echo "FAILED: wrong manifest hash accepted"; exit 1
else
  echo "  wrong manifest hash rejected: $(tail -1 $T/wrong_sha.txt)"
fi
if $PY compare_e0.py --pred-dir $T --a "e0d_pooled_s{}_preds.npz" --b "e0d_pooled_s{}_preds.npz" \
    --expect-manifest $SHA --boot 10 > $T/same_view.txt 2>&1; then
  echo "FAILED: arms with the same training view accepted"; exit 1
else
  echo "  same-view arms rejected: $(tail -1 $T/same_view.txt)"
fi
echo "ALL E0 SMOKE CHECKS PASSED"
