"""Small scientific-integrity checks; no production images or training."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

import analyze_radio_xray_offset as offset
import pilot_relic_association as relic


class FollowupChecks(unittest.TestCase):
    def test_parent_resampling_ignores_duplicate_views(self):
        a = offset.mean_interval(np.array([1., 3.]), [0, 1], np.random.default_rng(0))
        b = offset.mean_interval(np.repeat([1., 3.], 9), np.repeat([0, 1], 9),
                                 np.random.default_rng(0))
        self.assertEqual(a, b)

    def test_relic_components_and_bright_cell_removal(self):
        plane = np.array([[x, y, z] for x in [-500., 500.]
                          for y in np.arange(-250., 251., 50.) for z in [-50., 0., 50.]])
        weights = np.full(len(plane), 2e20)
        weights[0] = 1e25
        result = relic.segment(plane, weights, np.full(len(plane), 3.), 500., 14., 50.)
        structs, pair = result[3], result[5]
        self.assertEqual(len(structs), 2)
        self.assertEqual(len(pair), 2)
        self.assertEqual(sum(s["removed_cells"] for s in structs), 1)
        self.assertAlmostEqual(relic.largest_extent(np.array([[0., 0, 0], [3, 0, 0], [7, 0, 0]])), 7.)
        self.assertTrue(all(s["lls_kpc"] >= 300 for s in pair))

    def test_history_selection_excludes_multiple_events(self):
        ages = np.array([.2, .3, .6, .8, 1.1, 1.2, .3, .7])
        events = dict(T_coll=14.-ages, HaloID=np.array([0, 1, 2, 3, 4, 5, 6, 6]),
                      Mass_ratio=np.full(8, .2))
        sample = relic.select_sample(events, np.arange(7), np.r_[ages[:6], .3], 14.)
        self.assertEqual(set(sample.loc[sample.selected, "halo_id"]), set(range(6)))

    def test_offset_end_to_end_and_fold_rejection(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                self.run_offset_fixture()
            finally:
                os.chdir(previous)

    def run_offset_fixture(self):
        run = Path("run")
        (run / "preflight").mkdir(parents=True)
        (run / "preds").mkdir()
        Path("cnn_preds").mkdir()
        nr, n = 2, 40
        groups = np.arange(n)
        y = np.linspace(.6, 1.3, n)
        mass = np.linspace(14.76, 14.84, n)
        z = np.repeat(np.linspace(.075, .085, n)[:, None], nr, axis=1)
        paths = ["injected_syn_phys_b3000_s9991.h5", "xray_fix_b3000_s9991.h5"]
        for p in paths:
            with h5py.File(p, "w") as f:
                for k, v in dict(group_id=groups, log_m500=mass, z=z,
                                  exposure_ks=np.full((n, nr), 30.),
                                  images=np.zeros((n, nr, 3, 2, 2))).items():
                    f.create_dataset(f"mock/{k}", data=v)
                f.create_dataset("obs/name", data=np.array([b"A119"]))
                f.create_dataset("obs/key", data=np.array([b"A119"]))
                f.create_dataset("obs/exposure_ks", data=[30.])
        pd.DataFrame([dict(key="A119", log_m500c=14.8)]).to_csv("lovoccs_wl_masses.csv", index=False)
        verified = dict(radio=paths[0], xray=paths[1], sha256={p: "fixture" for p in paths})
        (run / "preflight/verified.json").write_text(json.dumps(verified))
        for seed in offset.SEEDS:
            base = dict(sample_group=np.repeat(groups, nr), sample_snap=np.full(n*nr, 99),
                        sample_realization=np.tile(np.arange(nr), n), sample_halo=np.repeat(groups, nr),
                        y_sample=np.repeat(y, nr), group_id=groups, snap=np.full(n, 99), halo_id=groups,
                        tsc=y, fold_of=np.repeat(groups % 5, nr), seed=seed,
                        sel_groups_flat=np.array([0, 1]), sel_groups_fold=np.array([1, 2]),
                        select="inner", train_view="pooled", select_view="pooled", loss="huber",
                        select_criterion="inner:pooled:r2", provenance_changed_during_run=False,
                        obs_name=np.array(["A119"]), obs_z=np.array([.08]))
            np.savez(run / f"preflight/splits_s{seed}.npz",
                     **{k: base[k] for k in ["sample_group", "fold_of", "sel_groups_flat", "sel_groups_fold"]})
            for arm, delta, dataset in [("radio", .3, paths[0]), ("xray", .1, paths[1]), ("joint", .05, paths[0])]:
                config = dict(dataset=dataset, epochs=60, folds=5, batch_size=32, huber_delta=.5, inner_frac=.15)
                prov = dict(config=config, sha256={"model": "same"},
                            inputs_sha256=dict(dataset="fixture", xray_dataset="fixture"))
                d = dict(base, provenance=json.dumps(prov), oof_cluster=y+delta,
                         oof_sample=np.repeat(y, nr)+delta,
                         oof_single=np.repeat((np.repeat(y, nr)+delta)[:, None], 3, axis=1),
                         obs_pred_folds=np.full((5, 1), .8+delta))
                path = (run / f"preds/xray_b3000_s{seed}_preds.npz" if arm == "xray" else
                        Path(f"cnn_preds/syn_phys_b3000_{'sfimg' if arm == 'radio' else 'joint'}_s{seed}_preds.npz"))
                np.savez(path, **d)
        for flag in [[], ["--include-joint"]]:
            with patch("sys.argv", ["analysis", "--run-dir", str(run)] + flag), contextlib.redirect_stdout(io.StringIO()):
                offset.main()
        summary = json.loads((run / "joint_comparison/summary.json").read_text())
        self.assertAlmostEqual(summary["simulated_single_view_offset"]["mean"], .2)
        self.assertAlmostEqual(summary["observed_median_offset"], .2)
        self.assertAlmostEqual(summary["joint_comparison"][0]["mae_gain_positive_favors_joint"]["mean"], .05)
        sensitivity = pd.read_csv(run / "offset/observed_age_sensitivity.csv")
        good = sensitivity[sensitivity.status == "estimated"]
        self.assertGreater(len(good), 0)
        self.assertTrue(np.allclose(good.observed_minus_simulated, 0))
        r = offset.load_predictions("cnn_preds/syn_phys_b3000_sfimg_s42_preds.npz")
        x = offset.load_predictions(run / "preds/xray_b3000_s42_preds.npz")
        x["fold_of"][0] = (x["fold_of"][0]+1) % 5
        with self.assertRaisesRegex(ValueError, "fold_of"):
            offset.check_pair(r, x)


if __name__ == "__main__":
    unittest.main()
