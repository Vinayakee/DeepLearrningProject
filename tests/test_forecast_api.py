"""Decision rules, stale-cache blocking and HTTP/frontend contracts."""
from datetime import date, datetime, timezone
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from backend.app import app
from prediction.service import data_freshness, summarize_forecast, decorate_bundle, warning_level, validate_current_contract

CALIBRATION = {"strong_probability_threshold": .4, "regional_elevated_percent": 10., "regional_strong_percent": 25.}


class WarningTests(unittest.TestCase):
    def test_freshness_boundary_and_future(self):
        today = date(2026, 1, 10)
        self.assertEqual(data_freshness('2026-01-07', today)['state'], 'fresh')
        self.assertEqual(data_freshness('2026-01-06', today)['state'], 'stale')
        self.assertEqual(data_freshness('2026-01-11', today)['state'], 'future')

    def test_warning_boundaries(self):
        self.assertEqual(warning_level(9.99, CALIBRATION), 'NORMAL')
        self.assertEqual(warning_level(10., CALIBRATION), 'ELEVATED RISK')
        self.assertEqual(warning_level(25., CALIBRATION), 'STRONG BLOOM RISK')

    def test_masked_decisions_and_daily_summary(self):
        p = np.tile([.5, .1, .4], (5, 2, 2, 1))
        mask = np.ones((5, 2, 2), bool)
        mask[:, 0, 0] = False
        risk, days = summarize_forecast(p, mask, CALIBRATION)
        self.assertTrue(np.all(risk[:, 0, 0] == 255))
        self.assertEqual(days[0]['supported_pixels'], 3)
        self.assertEqual(days[0]['high_risk_percentage'], 100.)
        self.assertEqual(days[0]['warning_level'], 'STRONG BLOOM RISK')
        mask[:] = False
        _, empty = summarize_forecast(p, mask, CALIBRATION)
        self.assertEqual(empty[0]['warning_level'], 'UNAVAILABLE')
        self.assertIsNone(empty[0]['max_probability'])
        with self.assertRaises(ValueError):
            summarize_forecast(p * 2, mask, CALIBRATION)

    def test_old_cache_hidden_without_explicit_demo(self):
        bundle = {'input_dates': ['2021-01-01'], 'data_quality': {'state': 'observed'},
                  'historical_demo': False, 'generated_at_utc': '2021-01-01T00:00:00Z',
                  'forecast_dates': ['2021-01-02'], 'daily_predictions': [{'day': 1, 'warning_level': 'STRONG BLOOM RISK'}]}
        blocked = decorate_bundle(bundle)
        self.assertEqual(blocked['status'], 'stale')
        self.assertNotIn('daily_predictions', blocked)
        demo = decorate_bundle(bundle, True)
        self.assertFalse(demo['operational_warnings_available'])
        self.assertFalse(demo['alerts'][0]['operational'])
        self.assertEqual(demo['alerts'][0]['delivery'], 'dashboard demonstration')

    def test_changed_fitted_artifacts_reject_old_model(self):
        training = {"label_metadata_sha256": "fitted-labels", "preprocessing_sha256": "fitted-stats"}
        with patch("prediction.service.file_sha256", return_value="newly-regenerated-artifact"):
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_current_contract(training)

    def test_normal_generates_no_alert(self):
        bundle = {'input_dates': [str(datetime.now(timezone.utc).date())], 'data_quality': {'state': 'observed'},
                  'historical_demo': False, 'forecast_dates': [str(datetime.now(timezone.utc).date())],
                  'daily_predictions': [{'day': 1, 'warning_level': 'NORMAL'}]}
        self.assertEqual(decorate_bundle(bundle)['alerts'], [])


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_home_health_dashboard_and_assets(self):
        self.assertEqual(self.client.get('/').json()['status'], 'success')
        self.assertEqual(self.client.get('/api/health').status_code, 200)
        page = self.client.get('/dashboard/').text
        self.assertIn('historical-demo', page)
        self.assertIn('prediction-map', page)
        self.assertNotIn('../outputs/maps/day_1.png', page)
        for name in ['style.css', 'script.js']:
            self.assertEqual(self.client.get('/dashboard/' + name).status_code, 200)

    def test_invalid_prediction_body(self):
        for body in [{'input_end_date': 'not-a-date'}, {'unknown_field': True}]:
            self.assertEqual(self.client.post('/api/predict', json=body).status_code, 422)

    def test_missing_forecast_and_broken_artifacts(self):
        with patch('prediction.service.cached_forecast', side_effect=FileNotFoundError):
            self.assertEqual(self.client.get('/api/prediction').json()['status'], 'unavailable')
            self.assertEqual(self.client.get('/api/alerts').json()['alerts'], [])
        with patch('prediction.service.cached_forecast', side_effect=ValueError('mismatch')):
            self.assertEqual(self.client.get('/api/prediction').status_code, 503)

    def test_predict_uses_requested_real_date_and_demo_flag(self):
        from unittest.mock import Mock
        service = Mock()
        service.generate.return_value = {'status': 'success'}
        with patch('backend.app.forecast_service', return_value=service), patch('preprocessing.data_utils.file_sha256', return_value='digest'):
            result = self.client.post('/api/predict', json={'input_end_date': '2025-12-31', 'historical_demo': True})
            self.assertEqual(result.status_code, 200)
            service.generate.assert_called_once_with(date(2025, 12, 31), True)
            service.generate.side_effect = ValueError('stale input')
            self.assertEqual(self.client.post('/api/predict', json={}).status_code, 422)

    def test_prediction_lock_and_evaluation_endpoint(self):
        from backend.app import prediction_lock
        prediction_lock.acquire()
        try:
            self.assertEqual(self.client.post('/api/predict', json={}).status_code, 409)
        finally:
            prediction_lock.release()
        self.assertEqual(self.client.get('/api/evaluation').status_code, 200)


if __name__ == '__main__':
    unittest.main()
