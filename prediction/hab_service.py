"""Explicitly configured, coverage-gated HAB research forecasts."""
import json
import os
from pathlib import Path
import numpy as np
import config
from preprocessing.data_utils import file_sha256


class HABArtifactError(ValueError):
    """A server-side HAB fitting artifact is missing, inconsistent or unreadable."""


def run_directory():
    value = os.getenv('HAB_OCCURRENCE_RUN_DIR')
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else config.ROOT/path


def status():
    directory = run_directory()
    base = {'target': 'Verified HAB occurrence within a coastal zone',
            'operational_warnings_available': False, 'alerts': [],
            'message': 'Supply verified HAB observations, prepare labels, train, calibrate and evaluate a separate HAB run. Chlorophyll classes are not substituted.'}
    if directory is None:
        return {**base, 'status': 'not_configured', 'next_step': 'Set HAB_OCCURRENCE_RUN_DIR after completing the verified-HAB workflow.'}
    try:
        record = json.loads((directory/'training.json').read_text())
        if record.get('state') != 'trained':
            return {**base, 'status': 'training_incomplete'}
        required = ('model.keras', 'baselines.json', 'decision.json', 'freeze.json', 'evaluation.json')
        if not all((directory/name).is_file() for name in required):
            return {**base, 'status': 'assessment_incomplete'}
        # Status is metadata; forecast/evaluation endpoints perform full integrity checks.
        return {**base, 'status': 'configured', 'target': record['target'],
                'message': 'A HAB research run is configured. Inference verifies artifacts, input coverage and freshness.',
                'epochs_completed': record['epochs_completed'], 'input_days': record['input_shape'][0],
                'output_days': 5, 'input_timing': record['input_timing']}
    except (OSError, ValueError, KeyError):
        return {**base, 'status': 'artifacts_unavailable'}


def evaluated_run(directory):
    from prediction.hab_evaluation import verify_decision
    try:
        decision, data = verify_decision(directory)
        report = json.loads((Path(directory)/'evaluation.json').read_text())
        if (report.get('schema') != 'hab-evaluation-v1' or report.get('model_sha256') != decision['model_sha256']
                or report.get('decision_sha256') != file_sha256(Path(directory)/'decision.json')):
            raise ValueError('HAB evaluation differs from the frozen model/decision.')
        return decision, data, report
    except (OSError, ValueError, KeyError) as error:
        raise HABArtifactError('HAB fitting artifacts are unavailable or inconsistent.') from error


def evaluation():
    directory = run_directory()
    if directory is None:
        return status()
    _, _, report = evaluated_run(directory)
    return {'status': 'success', **report}


def generate(input_end_date=None, historical_demo=False):
    from prediction.hab_evaluation import load_model, zone_quality, calibrated
    from prediction.service import data_freshness
    directory = run_directory()
    if directory is None:
        return status()
    decision, (pipeline, _, _, _, _), _ = evaluated_run(directory)
    end = str(input_end_date or pipeline.dates[-1])
    freshness = data_freshness(end)
    if freshness['state'] == 'future' or (freshness['state'] != 'fresh' and not historical_demo):
        raise ValueError('Archived inputs require explicit historical demonstration mode; future inputs are rejected.')
    found = np.flatnonzero(pipeline.dates == np.datetime64(end, 'D'))
    n = pipeline.spec['input_days']
    if not len(found) or found[0] < n-1:
        raise ValueError('Choose an archived date with a complete preceding input window.')
    indices = np.arange(found[0]-n+1, found[0]+1)
    X = pipeline.transform(indices)
    dates = [str(d) for d in pipeline.dates[indices]]
    quality = zone_quality(X, pipeline, dates)
    available = np.array([q['state'] == 'observed' for q in quality])
    targets = [str(np.datetime64(end, 'D')+np.timedelta64(i, 'D')) for i in range(1, 6)]
    base = {'status': 'success' if available.any() else 'insufficient_data', 'target': pipeline.spec['event_definition'],
            'historical_demo': historical_demo, 'input_dates': dates, 'forecast_dates': targets, 'freshness': freshness,
            'operational_warnings_available': False, 'alerts': [], 'model_sha256': decision['model_sha256'],
            'calibration_applied': bool(available.any()), 'probability_validation': 'Experimental calibration; no operational approval.',
            'lead_time_verified': False,
            'limitations': 'Zone occurrence prediction, not bloom boundaries, severity or toxicity. Archive publication timing is unverified. Unknown observations are not negative labels.',
            'zones': [{**zone, 'data_quality': q} for zone, q in zip(pipeline.spec['zones'], quality)],
            'daily_predictions': []}
    p = None
    if available.any():
        try:
            scores = load_model(directory)(X[None], training=False).numpy()[0]
        except (OSError, ValueError) as error:
            raise HABArtifactError('The HAB checkpoint could not be loaded for inference.') from error
        if scores.shape != (5, len(available)) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise HABArtifactError('Invalid HAB model output.')
        p = calibrated(scores, decision['methods']['model'])
    settings = decision['methods']['model']
    for i, day in enumerate(targets):
        zones = []
        for j, zone in enumerate(pipeline.spec['zones']):
            probability = float(p[i, j]) if available[j] else None
            level = ('UNAVAILABLE' if probability is None else 'HIGH' if probability >= settings['high_threshold']
                     else 'MODERATE' if probability >= settings['low_threshold'] else 'LOW')
            zones.append({'zone_id': zone['id'], 'probability': probability, 'risk_level': level,
                          'reason': 'Experimental calibrated HAB occurrence estimate.' if available[j]
                          else 'Insufficient observed environmental inputs in this zone.'})
        base['daily_predictions'].append({'day': i+1, 'date': day, 'zones': zones})
    base['thresholds'] = {key: settings[key] for key in ('low_threshold', 'high_threshold', 'threshold_objective')}
    return base
