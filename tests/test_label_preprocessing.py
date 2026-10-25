"""Scientific regression checks for training-only fitting and missing labels."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from preprocessing import create_chl_climatology as climatology
from preprocessing import create_final_hab_labels as labels
from preprocessing import data_utils as data
from preprocessing.report_data_quality import longest_missing_run, summarize_source


class LabelPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.dates = np.arange(np.datetime64("2021-01-01"), np.datetime64("2026-01-01"))
        rng = np.random.default_rng(42)
        self.values = rng.lognormal(size=(len(self.dates), 2, 3)).astype(np.float32)
        self.mask = np.ones(self.values.shape, dtype=np.uint8)
        self.ocean = np.array([[True, True, True], [True, True, False]])
        self.training = data.training_selection(self.dates)

    def fit(self, values=None, mask=None):
        return climatology.calculate_climatology(
            self.values if values is None else values,
            self.mask if mask is None else mask,
            self.dates, self.ocean)

    def test_held_out_values_and_masks_cannot_change_fitted_statistics(self):
        # This cell is observed only after the training period.
        self.values[self.training, 0, 1] = np.nan
        self.mask[self.training, 0, 1] = 0
        original = self.fit()
        _, original_labels, original_p75, original_p90 = labels.calculate_labels(
            self.values, self.mask, self.dates,
            original["monthly_chl_climatology_2021_2025.npy"], original["stable_ocean_mask.npy"])
        changed_values, changed_mask = self.values.copy(), self.mask.copy()
        changed_values[~self.training] = 1e6
        changed_mask[~self.training] = 0
        changed = self.fit(changed_values, changed_mask)
        _, changed_labels, changed_p75, changed_p90 = labels.calculate_labels(
            changed_values, changed_mask, self.dates,
            changed["monthly_chl_climatology_2021_2025.npy"], changed["stable_ocean_mask.npy"])
        for name in original:
            np.testing.assert_array_equal(original[name], changed[name])
        np.testing.assert_array_equal(original_p75, changed_p75)
        np.testing.assert_array_equal(original_p90, changed_p90)
        np.testing.assert_array_equal(original_labels[self.training], changed_labels[self.training])
        self.assertEqual(original["stable_ocean_mask.npy"][0, 1], 0)
        self.assertTrue(np.all(original_labels[:, 0, 1] == 255))

    def test_sparse_month_missing_and_non_ocean_cells_stay_unlabelled(self):
        january = np.flatnonzero(self.training & (data.month_numbers(self.dates) == 1))
        self.mask[january[2:], 0, 0] = 0
        self.values[january[2:], 0, 0] = np.nan
        self.mask[1100, 1, 1] = 0
        self.values[1101, 1, 1] = np.nan
        fitted = self.fit()
        self.assertTrue(np.isnan(fitted["monthly_chl_climatology_2021_2025.npy"][0, 0, 0]))
        anomaly, result, p75, p90 = labels.calculate_labels(
            self.values, self.mask, self.dates,
            fitted["monthly_chl_climatology_2021_2025.npy"], fitted["stable_ocean_mask.npy"])
        self.assertTrue(np.all(result[data.month_numbers(self.dates) == 1, 0, 0] == 255))
        self.assertTrue(np.all(result[:, 1, 2] == 255))
        self.assertEqual(result[1100, 1, 1], 255)
        self.assertEqual(result[1101, 1, 1], 255)
        for month in range(1, 13):
            training_values = anomaly[self.training & (data.month_numbers(self.dates) == month)]
            training_values = training_values[np.isfinite(training_values)]
            np.testing.assert_allclose([p75[month - 1], p90[month - 1]],
                                       np.percentile(training_values, [75, 90]))
        self.assertEqual(set(np.unique(result)), {0, 1, 2, 255})

    def test_mask_shape_and_empty_training_fail_explicitly(self):
        with self.assertRaises(ValueError):
            self.fit(mask=self.mask[:, :, :2])
        self.mask[0, 0, 0] = 255
        with self.assertRaises(ValueError):
            self.fit()
        with self.assertRaises(ValueError):
            data.training_selection(self.dates, "2030-01-01", "2031-01-01")
        with self.assertRaises(ValueError):
            data.training_selection(self.dates, "2023-01-01", "2021-01-01")

    def test_missing_training_month_cannot_create_held_out_labels(self):
        june_training = self.training & (data.month_numbers(self.dates) == 6)
        self.mask[june_training] = 0
        self.values[june_training] = np.nan
        fitted = self.fit()
        _, result, p75, p90 = labels.calculate_labels(
            self.values, self.mask, self.dates,
            fitted["monthly_chl_climatology_2021_2025.npy"], fitted["stable_ocean_mask.npy"])
        self.assertTrue(np.isnan(p75[5]) and np.isnan(p90[5]))
        self.assertTrue(np.all(result[data.month_numbers(self.dates) == 6] == 255))
        with self.assertRaisesRegex(ValueError, "minimum training observation"):
            self.fit(mask=np.zeros_like(self.mask))

    def test_dates_reject_gaps_duplicates_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dates_2021_2025.txt"
            for dates in ("2021-01-01\n2021-01-03\n", "2021-01-01\n2021-01-01\n", "NaT\n"):
                path.write_text(dates)
                with self.assertRaises(ValueError):
                    data.load_dates(directory)

    def write_fixture(self, directory):
        np.savetxt(directory / "dates_2021_2025.txt", self.dates, fmt="%s")
        np.save(directory / "latitude_kerala_2021_2025.npy", [12., 11.])
        np.save(directory / "longitude_kerala_2021_2025.npy", [74., 75., 76.])
        for name in ("chlorophyll", "sst", "current_u", "current_v"):
            np.save(directory / f"{name}_2021_2025.npy", self.values)
            np.save(directory / f"{name}_mask_2021_2025.npy", self.mask)

    def test_ocean_support_cannot_learn_from_held_out_years(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            marine = np.full(self.values.shape, np.nan, dtype=np.float32)
            mask = np.zeros(self.mask.shape, dtype=np.uint8)
            marine[self.training, 0, 0], mask[self.training, 0, 0] = 0, 1
            marine[~self.training, 0, 1], mask[~self.training, 0, 1] = 1, 1
            for name in ("sst", "current_u", "current_v"):
                np.save(directory / f"{name}_2021_2025.npy", marine)
                np.save(directory / f"{name}_mask_2021_2025.npy", mask)
            ocean = data.derive_ocean_support(directory, self.values.shape, self.training)
            self.assertEqual(ocean.sum(), 1)
            self.assertTrue(ocean[0, 0])
            self.assertFalse(ocean[0, 1])

    def test_commands_generate_bundle_and_reject_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            directory = Path(directory)
            self.write_fixture(directory)
            output = directory / "labels"
            arguments = ["command", "--processed-dir", str(directory), "--output-dir", str(output)]
            with patch("sys.argv", arguments):
                climatology.main()
                labels.main()
            result = np.load(output / "hab_labels_2021_2025.npy")
            self.assertEqual(result.shape, self.values.shape)
            self.assertTrue((output / "label_metadata.json").exists())
            with (output / "stable_ocean_mask.npy").open("ab") as stream:
                stream.write(b"changed")
            with patch("sys.argv", arguments), self.assertRaisesRegex(ValueError, "artifact"):
                labels.main()
            with patch("sys.argv", arguments):
                climatology.main()
                labels.main()
            previous_labels_hash = data.file_sha256(output / "hab_labels_2021_2025.npy")
            np.save(directory / "chlorophyll_2021_2025.npy", self.values * 2)
            with patch("sys.argv", arguments), self.assertRaisesRegex(ValueError, "inputs have changed"):
                labels.main()
            self.assertEqual(previous_labels_hash, data.file_sha256(output / "hab_labels_2021_2025.npy"))

    def test_quality_counts_use_fixed_ocean_denominator_and_preserve_values(self):
        values = np.array([[[1., 10.]], [[np.nan, 20.]], [[0., 30.]], [[2., 40.]]])
        mask = np.array([[[1, 1]], [[1, 1]], [[1, 1]], [[0, 1]]])
        original = values.copy()
        dates = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-01-05"))
        metrics, rows = summarize_source("chlorophyll", values, mask, dates, np.array([[True, False]]))
        self.assertEqual(metrics["ocean_coverage_percent"], 25.)
        self.assertEqual(metrics["grid_coverage_percent"], 62.5)
        self.assertEqual(metrics["days_without_ocean_observations"], 3)
        self.assertEqual(metrics["longest_run_without_ocean_observations_days"], 3)
        self.assertEqual(metrics["integrity_issues"]["masked_nonfinite_pixels"], 1)
        self.assertEqual(metrics["integrity_issues"]["masked_nonpositive_chlorophyll_pixels"], 1)
        self.assertEqual(metrics["integrity_issues"]["finite_unmasked_pixels"], 1)
        np.testing.assert_array_equal(values, original)
        self.assertEqual(longest_missing_run([0, 0, 1, 0]), 2)
        self.assertEqual(rows[0]["valid_outside_ocean_pixels"], 1)

    def test_output_guard_preserves_existing_checkpoint_labels(self):
        with self.assertRaises(ValueError):
            data.validate_output_directory(data.PROCESSED_DIR, data.PROCESSED_DIR)


if __name__ == "__main__":
    unittest.main()
