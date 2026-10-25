"""Real data/checkpoint integration; skipped on a code-only checkout."""
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
import config
from backend.app import app
from prediction.evaluation import load_trained_model


def corrected_run_ready():
    """Pre-training regression runs must not use older fitted checkpoints."""
    from preprocessing.data_utils import file_sha256
    try:
        training = json.loads((config.RUN_DIR / "training_config.json").read_text())
        calibration = json.loads((config.RUN_DIR / "calibration.json").read_text())
        return (not training.get("smoke") and bool(training.get("epochs_completed"))
                and training["model_sha256"] == file_sha256(config.MODEL_PATH)
                and training["label_metadata_sha256"] == file_sha256(config.LABEL_DIR / "label_metadata.json")
                and training["preprocessing_sha256"] == file_sha256(config.SEQUENCE_DIR / "preprocessing.json")
                and calibration["model_sha256"] == training["model_sha256"]
                and calibration["preprocessing_sha256"] == training["preprocessing_sha256"])
    except (OSError, ValueError, KeyError):
        return False


@unittest.skipUnless(corrected_run_ready(),
                     'Prepare real data, complete training and evaluate before real integration tests.')
class RealIntegrationTests(unittest.TestCase):
    def test_real_model_to_inference_api_maps_and_warning(self):
        client = TestClient(app)
        denied = client.post('/api/predict', json={'input_end_date': '2024-02-15'})
        self.assertEqual(denied.status_code, 422)
        response = client.post('/api/predict', json={'input_end_date': '2024-02-15', 'historical_demo': True})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['input_dates'], ['2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', '2024-02-15'])
        self.assertEqual(result['forecast_dates'], ['2024-02-16', '2024-02-17', '2024-02-18', '2024-02-19', '2024-02-20'])
        self.assertFalse(result['operational_warnings_available'])
        directory = config.FORECAST_DIR / result['forecast_id']
        p = np.load(directory / 'prediction.npy')
        support = np.load(directory / 'support.npy')
        risk = np.load(directory / 'risk_classes.npy')
        self.assertEqual(p.shape, (5, 120, 84, 3))
        np.testing.assert_allclose(p.sum(-1), 1., atol=1e-6)
        self.assertTrue(np.all(risk[~support] == 255))
        for row, url in zip(result['daily_predictions'], result['maps']):
            day = row['day'] - 1
            self.assertEqual(row['high_risk_pixels'], int((risk[day] == 2).sum()))
            self.assertEqual(row['supported_pixels'], int(support[day].sum()))
            image = client.get(url)
            self.assertEqual(image.status_code, 200)
            self.assertTrue(image.content.startswith(b'\x89PNG'))
        self.assertEqual(client.get('/api/prediction').json()['status'], 'stale')
        cached = client.get('/api/prediction?demo=true').json()
        self.assertEqual(cached['forecast_id'], result['forecast_id'])
        alerts = client.get('/api/alerts?demo=true').json()['alerts']
        self.assertTrue(all(not item['operational'] for item in alerts))
        self.assertEqual(len(alerts), sum(row['warning_level'] in ('ELEVATED RISK', 'STRONG BLOOM RISK') for row in result['daily_predictions']))

    def test_missing_inputs_return_unavailable_without_running_model(self):
        from prediction.service import ForecastService
        service = ForecastService()
        channels = service.model.input_shape[-1]
        with tempfile.TemporaryDirectory() as directory, patch('config.FORECAST_DIR', Path(directory)):
            with patch.object(service.pipeline, 'transform', return_value=np.zeros((5, 120, 84, channels), np.float32)), patch.object(service, 'model') as model:
                result = service.generate('2024-02-15', historical_demo=True)
                model.assert_not_called()
            self.assertEqual(result['status'], 'insufficient_data')
            self.assertEqual(result['data_quality']['observation_status'], 'MISSING DATA')
            self.assertEqual(result['prediction_availability'], 'unavailable')
            self.assertEqual(result['maps'], [])
            self.assertEqual(result['alerts'], [])
            self.assertFalse(result['operational_warnings_available'])
            self.assertFalse((Path(directory) / result['forecast_id'] / 'prediction.npy').exists())
            response = TestClient(app).get('/api/prediction?demo=true')
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['status'], 'insufficient_data')

    def test_completed_model_mismatch_rejected(self):
        with patch('prediction.evaluation.file_sha256', return_value='changed-checkpoint'):
            with self.assertRaisesRegex(ValueError, 'Model differs'):
                load_trained_model()


if __name__ == '__main__':
    unittest.main()
