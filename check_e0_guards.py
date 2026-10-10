"""Targeted malformed-metadata checks for compare_e0.py (review R7): copies
of the real E0 outputs with one metadata field altered must be rejected;
the unaltered copies must pass."""
import json, os, shutil, subprocess, sys, numpy as np
src, T = "cnn_preds", "e0_guard_tmp"
shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
SHA = subprocess.run(["sha256sum", "folds_e0.npz"], capture_output=True, text=True).stdout.split()[0]

def copy_arm(arm, mutate=None):
    for s in range(42, 47):
        d = dict(np.load(f"{src}/e0_x_{arm}_s{s}_preds.npz", allow_pickle=True))
        if mutate:
            mutate(s, d)
        np.savez(f"{T}/e0_x_{arm}_s{s}_preds.npz", **d)

def run():
    r = subprocess.run([sys.executable, "compare_e0.py", "--pred-dir", T,
                        "--a", "e0_x_pooled_s{}_preds.npz", "--b", "e0_x_single_s{}_preds.npz",
                        "--expect-manifest", SHA, "--boot", "20"], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1]

def prov_edit(fn):
    def m(s, d):
        if s == 44:
            p = json.loads(str(d["provenance"])); fn(p); d["provenance"] = json.dumps(p)
    return m

copy_arm("pooled"); copy_arm("single")
rc, last = run(); assert rc == 0, last; print("  unaltered copies pass")
cases = {
    "source hash differs": lambda: copy_arm("single", prov_edit(lambda p: p["sha256"].__setitem__("train_cnn_mock.py", "0" * 64))),
    "seed mislabelled": lambda: copy_arm("single", prov_edit(lambda p: p["config"].__setitem__("seed", 43))),
    "change flag missing": lambda: copy_arm("single", lambda s, d: d.pop("provenance_changed_during_run") if s == 44 else None),
    "change flag true": lambda: copy_arm("single", lambda s, d: d.__setitem__("provenance_changed_during_run", True) if s == 44 else None),
}
for name, make in cases.items():
    make(); rc, last = run()
    assert rc != 0, f"{name}: NOT rejected"
    print(f"  {name:<20} rejected: {last}")
    copy_arm("single")
shutil.rmtree(T)
print("ALL GUARD CHECKS PASSED")
