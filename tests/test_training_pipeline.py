import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
import config
from preprocessing.pipeline import build_sequence_index, EnvironmentalPipeline
from training.losses import MaskedWeightedLoss, MaskedAccuracy
from models.convlstm import build_convlstm


class TrainingTests(unittest.TestCase):
    def test_loss_ignores_missing_padding_and_handles_empty_targets(self):
        loss = MaskedWeightedLoss([1., 2., 3.])
        y = tf.constant([[[1., 0., 0.]]])
        p = tf.constant([[[.7, .2, .1]]])
        original = float(loss(y, p))
        padded_y = tf.concat([y, tf.zeros_like(y)], axis=1)
        padded_p = tf.concat([p, tf.ones_like(p) / 3], axis=1)
        self.assertAlmostEqual(original, float(loss(padded_y, padded_p)), places=6)
        self.assertEqual(float(loss(tf.zeros_like(y), p)), 0.)

    def test_accuracy_accumulates_valid_pixels_instead_of_batch_means(self):
        metric = MaskedAccuracy()
        metric.update_state(tf.constant([[1., 0., 0.]]), tf.constant([[.8, .1, .1]]))
        metric.update_state(tf.constant([[1., 0., 0.]] * 9), tf.constant([[.1, .8, .1]] * 9))
        self.assertAlmostEqual(float(metric.result()), .1, places=6)

    def test_sequence_windows_are_future_targets_without_split_crossing(self):
        dates = np.arange(np.datetime64("2021-01-01"), np.datetime64("2026-01-01"))
        labels = np.zeros((len(dates), 1, 1), dtype=np.uint8)
        index = build_sequence_index(dates, labels)
        for row in index.itertuples():
            first, last = config.SPLITS[row.split]
            self.assertGreaterEqual(row.input_start, first)
            self.assertLessEqual(row.target_end, last)
            self.assertEqual(row.target_start_idx, row.input_start_idx + 5)
            self.assertEqual(row.target_valid_pixels, 5)
        labels[:] = 255
        self.assertTrue(build_sequence_index(dates, labels).empty)

    def test_each_future_head_receives_last_input_and_model_reloads(self):
        tf.keras.utils.set_random_seed(42)
        model = build_convlstm()
        zero = np.zeros((1, 5, 120, 84, 14), dtype=np.float32)
        altered = zero.copy()
        altered[:, -1] = 1.
        a = model(zero, training=False).numpy()
        b = model(altered, training=False).numpy()
        self.assertEqual(a.shape, (1, 5, 120, 84, 3))
        self.assertTrue(np.all(np.max(np.abs(a - b), axis=(0, 2, 3, 4)) > 1e-6))
        np.testing.assert_allclose(b.sum(-1), 1., atol=1e-6)
        model.compile(optimizer="adam", loss=MaskedWeightedLoss(), metrics=[MaskedAccuracy()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            np.testing.assert_allclose(restored(altered, training=False), b, atol=1e-6)

    @unittest.skipUnless((config.SEQUENCE_DIR / "manifest.json").exists(), "Processed local dataset not prepared")
    def test_real_shared_transform_and_corrected_generator_match(self):
        from training.data_generator import HABDataGenerator
        pipeline = EnvironmentalPipeline()
        generator = HABDataGenerator("train", shuffle=False, pipeline=pipeline)
        X, y = generator[0]
        start = int(generator.indices.iloc[0].input_start_idx)
        np.testing.assert_array_equal(X[0], pipeline.transform(np.arange(start, start + 5)))
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue(np.all(y[:, :, ~pipeline.ocean] == 0))
        self.assertEqual(X.shape, (1, 5, 120, 84, 16 if pipeline.stats.get("seasonal_features", False) else 14))
        self.assertEqual(y.shape, (1, 5, 120, 84, 3))


if __name__ == "__main__":
    unittest.main()
