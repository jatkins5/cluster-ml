#!/usr/bin/env python3
"""Small scientific/identity checks; no survey data scan."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from spatial_pilot import (ARMS, feature_pair, moments, paired_order, partitions,
                           peak_position, radio_weights, score, xray_to_radio_position)


class GeometryChecks(unittest.TestCase):
    def test_double_relics_do_not_cancel_radius(self):
        weights = np.zeros((11, 11))
        weights[5, 2] = weights[5, 8] = 1
        np.testing.assert_allclose(moments(weights, 10, (0, 0)), [0, 30])
        np.testing.assert_allclose(moments(weights, 10, (0, 40)), [40, 40])

    def test_physical_axis_conventions(self):
        # 3D point (x,y,z)=(10,20,30); SOXS rows/cols = vertical/horizontal.
        raw = ([20, 10], [30, 20], [10, 30])
        expected = ([10, 20], [20, 30], [10, 30])
        for p in range(3):
            np.testing.assert_array_equal(xray_to_radio_position(raw[p], p), expected[p])

    def test_joint_rotations_preserve_moments(self):
        w = np.zeros((11, 11))
        w[3, 7], w[6, 2] = 2, 1
        expected = moments(w, 10, [10, 20])
        # np.rot90 maps centred row/col (r,c) -> (-c,r).
        np.testing.assert_allclose(moments(np.rot90(w), 10, (-20, 10)), expected)
        np.testing.assert_allclose(moments(np.fliplr(w), 10, (10, -20)), expected)

    def test_noise_only_and_isolated_pixels(self):
        im = np.zeros((16, 16))
        im[5, 5] = np.arcsinh(10)
        self.assertEqual(radio_weights(im).sum(), 0)
        im[5:7, 5:7] = np.arcsinh(10)
        self.assertAlmostEqual(radio_weights(im).sum(), 28)
        _, valid = peak_position(np.zeros_like(im), 10, 40, 60)
        self.assertFalse(valid)

    def test_peak_and_registration_on_synthetic_image(self):
        n = 33
        yy, xx = np.indices((n, n))
        x = np.arcsinh(100 * np.exp(-((yy - 18)**2 + (xx - 12)**2) / 2))
        r = np.zeros((n, n))
        r[19:22, 12:15] = np.arcsinh(10)
        origin, registered, peak = feature_pair(r, x, 0, 10, 10, 100, 140)
        np.testing.assert_allclose(peak, [-40, 20])
        self.assertEqual(origin[0], 1)
        self.assertEqual(origin[1], 1)
        np.testing.assert_allclose(registered[2:], moments(radio_weights(r), 10, [-40, 20]))
        self.assertNotEqual(origin[2], registered[2])


class IdentityChecks(unittest.TestCase):
    def test_grouped_partitions_and_reject_leak(self):
        ids = np.arange(12)
        mf = dict(group_id=ids, fold=ids % 3, folds=3,
                  inner_sel=np.array([ids % 3 == (k + 1) % 3 for k in range(3)]))
        groups = np.tile(np.repeat(ids, 2), 2)
        splits, _ = partitions(groups, mf)
        self.assertEqual(len(splits), 3)
        for fit, sel, test in splits:
            self.assertFalse(set(groups[fit]) & set(groups[test]))
            self.assertFalse(set(groups[sel]) & set(groups[test]))
        mf["inner_sel"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "outer-test"):
            partitions(groups, mf)

    def test_pairing_reorders_but_rejects_redshift_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="spatial-fixture-") as tmp:
            paths = [Path(tmp) / name for name in ("radio.h5", "xray.h5")]
            for path, ix in zip(paths, ([0, 1], [1, 0])):
                ix = np.array(ix)
                with h5py.File(path, "w") as f:
                    f.attrs["box_kpc"], f.attrs["recipe"] = 3000, "physical"
                    f.attrs["count_model"] = "real"
                    for key, value in dict(group_id=[7, 7], snap=[99, 91],
                                           halo_id=[7, 91000000007],
                                           pseudo_tsc=[1., 2.], z=[[.05], [.08]]).items():
                        f.create_dataset("mock/" + key, data=np.asarray(value)[ix])
                    f.create_dataset("mock/images", shape=(2, 1, 3, 128, 128), dtype="f4")
            with h5py.File(paths[0], "r") as r, h5py.File(paths[1], "r+") as x:
                np.testing.assert_array_equal(paired_order(r, x), [1, 0])
                x["mock/z"][0] = .1
                with self.assertRaisesRegex(ValueError, "paired z"):
                    paired_order(r, x)

    def test_fold_imputation_never_uses_test_values(self):
        from train_cnn_mock import prep_view_scalars
        data = np.array([[[1., 1., 10., np.nan]], [[1., 1., 20., 40.]],
                         [[1., 1., 9999., 9999.]]])
        _, med, mu, sd = prep_view_scalars(data, [0, 1], np.zeros(4, bool),
                                          ["a", "b", "c", "d"])
        np.testing.assert_allclose(med, [1, 1, 15, 40])
        _, med2, mu2, sd2 = prep_view_scalars(
            data, [0, 1], np.array([False, False, True, True]), ["a", "b", "c", "d"])
        np.testing.assert_allclose(med, med2)
        np.testing.assert_allclose(mu, mu2)
        np.testing.assert_allclose(sd, sd2)

    def test_scorer_detects_incomplete_predictions(self):
        with tempfile.TemporaryDirectory(prefix="spatial-score-") as tmp:
            out = Path(tmp)
            y = np.array([0., 1., 2., 3.])
            for arm, error in zip(ARMS, (0.3, 0.2, 0.1)):
                np.savez(out / f"{arm}_s42.npz", arm=arm, seed=42,
                         prediction=np.repeat((y + error)[:, None], 3, axis=1), y=y,
                         sample_group=np.arange(4), sample_snap=np.full(4, 99),
                         sample_halo=np.arange(4), sample_realization=np.zeros(4),
                         fold_of=np.arange(4), inner_groups=np.arange(4),
                         inner_folds=np.arange(4), provenance_sha256="same",
                         image_preprocessing="same")
            report = score(out, [42], draws=20)
            self.assertAlmostEqual(report["contrasts"]["registered_minus_origin"]
                                   ["delta_mae_gyr"], -0.1)
            data = dict(np.load(out / "registered_s42.npz"))
            data["prediction"][0, 0] = np.nan
            np.savez(out / "registered_s42.npz", **data)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                score(out, [42], draws=20)


class ModelChecks(unittest.TestCase):
    def test_per_observation_forward_and_gradient(self):
        import torch
        from train_cnn_mock import JointViewScalarCNN, predict
        torch.set_num_threads(1)
        torch.manual_seed(1)
        model = JointViewScalarCNN(4, embed_dim=8).eval()
        radio = torch.randn(2, 1, 1, 16, 16)
        xray = torch.randn(2, 1, 1, 32, 32)
        features = torch.randn(2, 1, 4)
        one = model(radio, xray, features)
        many = model(radio.repeat(1, 3, 1, 1, 1), xray.repeat(1, 3, 1, 1, 1),
                     features.repeat(1, 3, 1))
        torch.testing.assert_close(one, many)
        one.sum().backward()
        self.assertTrue(torch.isfinite(model.fuse[0].weight.grad).all())
        result = predict(model, radio[:, :, 0].numpy(), "cpu",
                         xray=xray[:, :, 0].numpy(), view_scalars=features.numpy())
        np.testing.assert_allclose(result, one.detach().numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
