#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 01:30:00
#SBATCH -J smoker9
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/smoke_r9_%j.out
# Review round 9 implementation checks (fail fast). Diagnostics only: the
# 1-epoch scores are not performance results.
#  1. masked controls: hidden structure inputs cannot change a control
#     prediction, richness can; missing-feature policy; the R8 single-view
#     projection check still holds
#  2. 1-epoch fixture: all four E1X arms x 2 seeds through the real trainer
#  3. compare_e1x.py accepts the valid fixture
#  4. compare_e1x.py rejects: changed richness, placement redshift, noisy
#     mass, feature-file hash, wrong mask, undeclared config change, missing
#     seed, source change during the run
#  5. pilot toy check: a no-op shuffle now fails the radial-mixing assertion
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
PY="./venv/bin/python -u"
T=smoke_r9_tmp; rm -rf $T; mkdir -p $T

echo "======== 1. masked-control invariance and missing-feature policy"
$PY - <<'EOF'
import numpy as np, torch
from torch.utils.data import DataLoader
from train_cnn_mock import ViewScalarCNN, MockDataset, predict, prep_view_scalars
torch.manual_seed(0); rs = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
M, P, H, G = 12, 3, 128, 2
cols = ["richness", "gap12", "off_bcg_x", "off_bcg_gal", "asym"]
mask = np.array([False, True, True, True, True])
x = rs.random((M, P, H, H)).astype(np.float32)
gs = rs.standard_normal((M, G)).astype(np.float32)
raw = rs.standard_normal((M, P, 5)); raw[rs.random(raw.shape) < 0.1] = np.nan
fit = np.arange(8)
m = ViewScalarCNN(5, G).to(dev).eval()
def preds(raw_):
    vsn = prep_view_scalars(raw_, fit, mask, cols)[0]
    single = np.stack([predict(m, np.repeat(x[:, p:p + 1], P, 1), dev, scalars=gs,
                               view_scalars=np.repeat(vsn[:, p:p + 1], P, 1))
                       for p in range(P)], 1)
    return vsn, single, predict(m, x, dev, scalars=gs, view_scalars=vsn)
v0, s0, p0 = preds(raw)
hid = raw.copy(); hid[..., 1:] = rs.standard_normal(hid[..., 1:].shape) * 50
hid[0, 0, 2] = np.nan                                   # missingness pattern too
v1, s1, p1 = preds(hid)
assert np.array_equal(v0, v1), "hidden structure inputs changed the model input"
assert np.allclose(s0, s1, rtol=0, atol=1e-6) and np.allclose(p0, p1, rtol=0, atol=1e-6), \
    "hidden structure inputs changed a control prediction"
assert (v0[..., 1:] == 0).all()
ric = raw.copy(); ric[..., 0] += rs.standard_normal(ric.shape[:2])   # non-uniform: a constant shift is normalised away
v2, s2, p2 = preds(ric)
assert not np.allclose(s0, s2) and not np.allclose(p0, p2), "richness unused"
print("  masked structure: model input bitwise unchanged, single/pooled predictions unchanged; "
      "richness changes them")
bad = raw.copy(); bad[fit, :, 0] = np.nan
try:
    prep_view_scalars(bad, fit, mask, cols); raise SystemExit("unmasked all-missing accepted")
except ValueError as e:
    print(f"  unmasked feature with no fitting values rejected: {e}")
bad = raw.copy(); bad[fit, :, 3] = np.nan
v3 = prep_view_scalars(bad, fit, mask, cols)[0]
assert np.isfinite(v3).all() and (v3[..., 3] == 0).all()
print("  masked feature with no fitting values: allowed, constant 0, finite")
# R8 projection check, now with 5 slots
# perturb the MODEL INPUT of projection 1's richness (a raw change would also
# move the fitted mean/std shared by all projections)
a = prep_view_scalars(raw, fit, mask, cols)[0]
b = a.copy(); b[:, 1, 0] += 5.0
def single_view(vsn):
    ds = MockDataset(x, np.zeros(M, np.float32), gs, view="single", view_scalars=vsn)
    out = []
    with torch.no_grad():
        for im, _, sc, v, _ in DataLoader(ds, batch_size=8):
            out.append(m(im.to(dev), v.to(dev), sc.to(dev)).cpu().numpy())
    return np.concatenate(out).reshape(M, P)
sa, sb = single_view(a), single_view(b)
assert np.allclose(sa[:, 0], sb[:, 0]) and np.allclose(sa[:, 2], sb[:, 2])
assert not np.allclose(sa[:, 1], sb[:, 1])
print("  single-view: projection 1's richness changes only projection 1's prediction")
EOF

echo "======== 2. 1-epoch fixture: four arms x seeds 42, 43"
S4="gap12,off_bcg_x,off_bcg_gal,asym"
for S in 42 43; do
  COMMON="--dataset xray_fix_s9991.h5 --fold-manifest folds_e0.npz --train-view pooled
          --select-view single --mass --mass-noise --mass-seed 0 --redshift-scalar --seed $S
          --view-scalars optical_features_idealised.csv
          --view-cols richness,$S4 --epochs 1"
  $PY train_cnn_mock.py $COMMON --no-image --view-mask $S4 --out-prefix $T/e1x_ctrl_s$S | grep -E "masked|OOF R2 \(ONE"
  $PY train_cnn_mock.py $COMMON --no-image --out-prefix $T/e1x_ctrlopt_s$S | grep -E "OOF R2 \(ONE"
  $PY train_cnn_mock.py $COMMON --view-mask $S4 --out-prefix $T/e1x_xctrl_s$S | grep -E "OOF R2 \(ONE"
  $PY train_cnn_mock.py $COMMON --out-prefix $T/e1x_xctrlopt_s$S | grep -E "OOF R2 \(ONE"
done

MAN=f08136fd6d1a32083c8ea6287fa6a64ff350e8bf2ca5e0a2a7187cf33461df4d
echo "======== 3. compare_e1x.py on the valid fixture"
$PY compare_e1x.py --expect-manifest $MAN --pred-dir $T --seeds 42,43 --boot 200

echo "======== 4. compare_e1x.py rejections"
$PY - <<'EOF'
import copy, json
import numpy as np
import compare_e1x as C
MAN = "f08136fd6d1a32083c8ea6287fa6a64ff350e8bf2ca5e0a2a7187cf33461df4d"
T = "smoke_r9_tmp"
runs = C.load(T, [42, 43])
C.check(runs, MAN)
d0 = runs["ctrl"][42]
print(f"  fixture: global cols {list(d0['global_cols'])}, view cols {list(d0['view_cols'])}; "
      f"inputs hashed {sorted(json.loads(str(d0['provenance']))['inputs_sha256'])}")
def prov_edit(d, f):
    p = json.loads(str(d["provenance"])); f(p); d["provenance"] = np.array(json.dumps(p))
def expect_fail(name, mutate, arm="xctrlopt", seed=43):
    r = copy.deepcopy(runs)
    mutate(r[arm][seed])
    try:
        C.check(r, MAN)
    except SystemExit as e:
        print(f"  rejected ({name}): {e}")
        return
    raise SystemExit(f"NOT rejected: {name}")
def ric(d): d["view_scalar_raw"] = d["view_scalar_raw"].copy(); d["view_scalar_raw"][0, 1, 0] += 1
def zed(d): d["global_scalar_raw"] = d["global_scalar_raw"].copy(); d["global_scalar_raw"][5, 1] += 0.01
def mass(d):
    for k in ("global_scalar_raw", "sim_logm_input"):
        d[k] = d[k].copy(); d[k][0, ...] += 0.1
def fhash(d): prov_edit(d, lambda p: p["inputs_sha256"].__setitem__("view_scalars", "0" * 64))
def mask(d):
    d["view_mask"] = np.array([False, True, True, True, False])
    prov_edit(d, lambda p: p["config"].__setitem__("view_mask", "gap12,off_bcg_x,off_bcg_gal"))
def cfg(d): prov_edit(d, lambda p: p["config"].__setitem__("epochs", 2))
def changed(d): d["provenance_changed_during_run"] = np.array(True)
def seedname(d): prov_edit(d, lambda p: p["config"].__setitem__("seed", 42))
expect_fail("changed richness", ric)
expect_fail("changed placement redshift", zed)
expect_fail("changed noisy mass", mass)
expect_fail("changed feature-file hash", fhash)
expect_fail("wrong mask", mask, arm="xctrl")
expect_fail("undeclared config change", cfg)
expect_fail("inputs changed during run", changed)
expect_fail("seed differs from filename", seedname)
try:
    C.load(T, [42, 43, 44]); raise SystemExit("NOT rejected: missing seed")
except SystemExit as e:
    if str(e).startswith("NOT"): raise
    print(f"  rejected (missing seed): {e}")
EOF

echo "======== 5. pilot radial-mixing check rejects a no-op shuffle"
$PY - <<'EOF'
import numpy as np
src = open("train_optical_pilot.py").read()
fn = src[src.index("def radial_mix"):src.index("def fit_red_sequence")]
chk = src[src.index("# constructed check (review R8)"):src.index("# ---------------------------------------------------------------- models")]
ns = {"np": np}
exec(fn, ns); exec(chk, ns)                       # real shuffle: passes
ns2 = {"np": np}
exec(fn, ns2); ns2["radial_mix"] = lambda m, rs: m.copy()
try:
    exec(chk, ns2); raise SystemExit("no-op shuffle NOT rejected")
except AssertionError as e:
    print(f"  no-op shuffle rejected: {e}")
EOF
echo "ALL ROUND-9 SMOKE CHECKS PASSED"
