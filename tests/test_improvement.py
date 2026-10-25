"""Scientific safeguards for experiments and optional improvements."""
import unittest
import tempfile
import json
from datetime import datetime, timezone, timedelta
from prediction.service import decorate_bundle
from pathlib import Path
import pandas as pd
import numpy as np
from preprocessing.features import seasonal_features, add_seasonal_features
from experiments.train import class_weights
from prediction.decision import apply_rule
from experiments.common import cached_samples, baseline_pipeline, BASELINE_DIR
from experiments.regional import summarize_rows
from experiments.finalize import claim_test_evaluation
from prediction.coverage import assess_coverage
from prediction.regional import regional_summary, binary_warning_metrics


class ImprovementTests(unittest.TestCase):
    def test_weight_normalization_and_rare_class_order(self):
        counts = np.array([750., 150., 100.])
        baseline = np.sqrt(counts.max() / counts)
        for power in [.5, .75, 1.]:
            weights = np.array(class_weights(counts, power))
            self.assertAlmostEqual(float(weights @ (counts / counts.sum())), float(baseline @ (counts / counts.sum())))
            self.assertTrue(np.all(np.diff(weights) > 0))
        np.testing.assert_allclose(class_weights(counts, .5), baseline)
        with self.assertRaises(ValueError):
            class_weights([1., 0., 3.])

    def test_elevated_boundary_and_strong_override(self):
        p = np.array([[.55, .30, .15], [.50, .25, .25]], np.float32)
        np.testing.assert_array_equal(apply_rule(p, {"type": "strong_override", "strong_threshold": .25}), [0, 2])
        np.testing.assert_array_equal(apply_rule(p, {"type": "score_ratios", "scales": [1., 2., 1.]}), [1, 0])
        with self.assertRaises(ValueError):
            apply_rule(p, {"type": "score_ratios", "scales": [1., 0., 1.]})

    def test_cyclical_features_and_shared_batch_encoding(self):
        dates = ['2024-12-30', '2024-12-31', '2025-01-01', '2025-01-02', '2025-01-03']
        features = seasonal_features(dates, 2, 3)
        self.assertEqual(features.shape, (5, 2, 3, 2))
        np.testing.assert_allclose((features ** 2).sum(-1), 1., atol=1e-6)
        self.assertLess(float(np.linalg.norm(features[1, 0, 0] - features[2, 0, 0])), .6)
        X = np.zeros((5, 2, 3, 14), np.float32)
        single = add_seasonal_features(X, dates)
        batch = add_seasonal_features(X[None], dates)
        np.testing.assert_array_equal(single, batch[0])
        np.testing.assert_array_equal(single[..., :14], X)
        with self.assertRaises(ValueError):
            add_seasonal_features(X, dates[:2])

    def test_coverage_states_and_exact_cell_day_counts(self):
        X = np.zeros((5, 2, 2, 14), np.float32)
        ocean = np.ones((2, 2), bool)
        dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        self.assertEqual(assess_coverage(X, ocean, dates)['observation_status'], 'MISSING DATA')
        X[..., 8:14] = 1
        X[0, 0, 0, 7] = 1
        self.assertEqual(assess_coverage(X, ocean, dates)['observation_status'], 'INSUFFICIENT DATA')
        X[1, 0, 0, 7] = 1
        low = assess_coverage(X, ocean, dates, {'minimum_chlorophyll_mean_percent': 20., 'minimum_chlorophyll_observed_days': 2, 'minimum_environment_observed_days': 1, 'minimum_target_coverage_percent': 5.})
        self.assertEqual(low['observation_status'], 'LOW COVERAGE')
        self.assertEqual(low['prediction_availability'], 'exploratory_only')
        quality = assess_coverage(X, ocean, dates)
        self.assertEqual(quality['observation_status'], 'VALID OBSERVATION')
        self.assertEqual(quality['valid_observations'], 2)
        self.assertEqual(quality['features']['chlorophyll']['observed_input_days'], 2)
        X[..., 8] = 0
        self.assertEqual(assess_coverage(X, ocean, dates)['prediction_availability'], 'unavailable')

    def test_regional_aggregation_and_abstention_is_missed_warning(self):
        risk = np.array([[2, 1], [0, 2]], np.uint8)
        support = np.array([[True, True], [True, False]])
        summary = regional_summary(risk, support, {'regional_elevated_percent': 20., 'regional_strong_percent': 50.})
        self.assertEqual(summary['supported_pixels'], 3)
        self.assertEqual(summary['high_risk_pixels'], 1)
        self.assertEqual(summary['elevated_risk_pixels'], 1)
        self.assertEqual(summary['warning_level'], 'ELEVATED RISK')
        result = binary_warning_metrics([True, True, False, False], [True, False, True, False])
        self.assertEqual(result['correctly_warned_high_risk_days'], 1)
        self.assertEqual(result['missed_high_risk_days'], 1)
        self.assertEqual(result['false_warning_days'], 1)
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['recall'], .5)

    def test_development_rejects_test_cache_before_loading_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'cache.json').write_text(json.dumps({'split': 'test'}))
            with self.assertRaisesRegex(ValueError, 'prohibit 2025'):
                next(cached_samples(folder, None))

    def test_unknown_regional_truth_is_excluded_and_abstentions_are_counted(self):
        frame = pd.DataFrame({'truth_qualified': [True, True, False], 'available': [True, False, True],
                              'input_state': ['observed', 'low_coverage', 'observed'],
                              'input_coverage_percent': [10., 2., 10.],
                              'actual_level': ['STRONG BLOOM RISK', 'STRONG BLOOM RISK', 'UNKNOWN'],
                              'issued_level': ['STRONG BLOOM RISK', 'INSUFFICIENT DATA', 'STRONG BLOOM RISK']})
        result = summarize_rows(frame)
        self.assertEqual(result['unknown_truth_cases'], 1)
        self.assertEqual(result['warnings_without_qualified_truth'], 1)
        self.assertEqual(result['strong_warning']['qualified_days'], 2)
        self.assertEqual(result['strong_warning']['abstained_high_risk_days'], 1)
        self.assertEqual(result['strong_warning']['missed_high_risk_days'], 1)
        self.assertEqual(result['strong_warning']['false_warning_days'], 0)

    def test_final_holdout_claim_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as folder:
            claim_test_evaluation(folder, 'frozen-checksum')
            self.assertEqual(json.loads(Path(folder, 'test_started.json').read_text())['freeze_sha256'], 'frozen-checksum')
            with self.assertRaisesRegex(ValueError, 'already started'):
                claim_test_evaluation(folder, 'another-checksum')

    @unittest.skipUnless((BASELINE_DIR / 'freeze.json').exists(), 'Frozen local baseline not prepared')
    def test_optional_seasonal_preprocessing_preserves_existing_fields(self):
        pipeline = baseline_pipeline()
        indices = np.arange(1096, 1101)  # January 2024 inputs only.
        original = pipeline.transform(indices)
        pipeline.stats['seasonal_features'] = True
        extended = pipeline.transform(indices)
        self.assertEqual(extended.shape, (5, 120, 84, 16))
        np.testing.assert_array_equal(extended[..., :14], original)
        np.testing.assert_array_equal(extended[..., 14:], seasonal_features(pipeline.dates[indices], 120, 84))

    def test_unvalidated_regional_warning_gate_prevents_operational_delivery(self):
        today = datetime.now(timezone.utc).date()
        bundle = {'status': 'success', 'input_dates': [str(today)],
                  'data_quality': {'state': 'observed'}, 'historical_demo': False,
                  'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                  'forecast_dates': [str(today + timedelta(days=1))],
                  'daily_predictions': [{'day': 1, 'warning_level': 'STRONG BLOOM RISK'}],
                  'operational_warning_validated': False}
        blocked = decorate_bundle(bundle)
        self.assertEqual(blocked['status'], 'unavailable')
        demo = decorate_bundle(bundle, allow_demo=True)
        self.assertFalse(demo['operational_warnings_available'])
        self.assertEqual(len(demo['alerts']), 1)
        self.assertFalse(demo['alerts'][0]['operational'])


if __name__ == '__main__':
    unittest.main()
