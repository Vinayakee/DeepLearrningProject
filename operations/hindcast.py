"""Generate frozen research forecasts for supplied QC inputs; no live claims."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import config
from experiments.finalize import verify_final
from preprocessing.pipeline import EnvironmentalPipeline
from prediction.evaluation import load_trained_model
from prediction.service import summarize_forecast, render_maps, data_freshness
from preprocessing.data_utils import file_sha256, month_numbers
from upgrade.live import prepare_inputs, observation_summary
from prediction.coverage import assess_coverage
from operations.provenance import validate_manifest


def generate(directory, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; preserve forecasts and choose a new directory.')
    verify_final()
    pipeline = EnvironmentalPipeline()
    validate_manifest(directory)
    X, dates, metadata = prepare_inputs(directory, pipeline)
    targets = [str(dates[-1] + np.timedelta64(i, 'D')) for i in range(1, 6)]
    if any('2025-01-01' <= str(d) <= '2025-12-31' for d in [*dates, *targets]):
        raise ValueError('2025 is frozen; this integration does not repeat its evaluation.')
    calibration = json.loads((config.RUN_DIR / 'calibration.json').read_text())
    quality = assess_coverage(X, pipeline.ocean, dates, calibration['coverage_policy'])
    bundle = {'status': 'insufficient_data', 'forecast_id': output.name,
              'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'historical_demo': True, 'input_dates': [str(d) for d in dates], 'forecast_dates': targets,
              'freshness': data_freshness(str(dates[-1])), 'data_quality': quality,
              'maps': [], 'daily_predictions': [], 'array_sha256': {},
              'model_sha256': file_sha256(config.MODEL_PATH), 'decision_rule': calibration['decision_rule'],
              'operational_warning_validated': False, 'operational_warnings_available': False,
              'source_metadata_sha256': file_sha256(Path(directory) / 'metadata.json'),
              'source_inputs_sha256': file_sha256(Path(directory) / 'inputs.npz'),
              'sources': metadata['sources'], 'satellite_source': 'MODIS', 'lead_time': 'Verified manifest timing; native provenance accuracy must be independently assessed.'}
    output.mkdir(parents=True, exist_ok=False)
    if quality['state'] == 'observed':
        model, _ = load_trained_model()
        scores = model(X[None], training=False).numpy()[0]
        support = pipeline.ocean[None] & np.isfinite(pipeline.climatology[month_numbers(targets) - 1])
        risk, days = summarize_forecast(scores, support, calibration)
        np.save(output / 'prediction.npy', scores); np.save(output / 'risk_classes.npy', risk)
        render_maps(scores, risk, pipeline.latitude, pipeline.longitude, targets, output)
        observed, environment = observation_summary(directory, pipeline, str(dates[-1]), calibration)
        bundle.update(status='success', daily_predictions=days, current_observed_risk=observed, environment=environment,
                      maps=[str((output / f'day_{i}.png').resolve()) for i in range(1, 6)],
                      array_sha256={n: file_sha256(output / n) for n in ['prediction.npy', 'risk_classes.npy']})
    (output / 'bundle.json').write_text(json.dumps(bundle, indent=2))
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = generate(args.input_dir, args.output_dir)
    except FileNotFoundError:
        result = {'status': 'unavailable', 'reason': 'Real staged inputs or matching fitted artifacts are missing.'}
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'status': 'invalid_input', 'reason': str(error)}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
