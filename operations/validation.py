"""Field matching and supplied-label independent evaluation; never train/tune."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import config
from upgrade.field_records import load_verified_records
from preprocessing.data_utils import file_sha256
from prediction.metrics import classification_metrics, confusion_matrix
from prediction.regional import warning_level


def load_fields(path, purpose):
    # Validate malformed CSV and optional lab fields without modifying the frozen
    # original record validator or inventing a risk-severity conversion.
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Field CSV row length differs from its header.')
            if row.get('toxin_value', '').strip():
                value = float(row['toxin_value'])
                if not np.isfinite(value) or value < 0 or not row.get('toxin_units', '').strip():
                    raise ValueError('Measured toxin values require finite nonnegative values and documented units.')
    return load_verified_records(path, purpose)


def grid_index(latitude, longitude, lat, lon):
    """Nearest center only inside the actual grid footprint, never coastal snapping."""
    indices = []
    for axis, value in ((latitude, lat), (longitude, lon)):
        axis = np.asarray(axis, float)
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or not np.isfinite(value):
            raise ValueError('Finite coordinate vectors/locations required.')
        steps = np.diff(axis)
        if not (np.all(steps > 0) or np.all(steps < 0)) or not np.allclose(np.abs(steps), np.abs(steps[0]), atol=1e-6):
            raise ValueError('Regular monotonic grid required.')
        half = abs(steps[0]) / 2
        if value < axis.min() - half or value > axis.max() + half:
            return None
        indices.append(int(np.abs(axis - value).argmin()))
    return tuple(indices)


def load_forecast(directory, independent=False):
    directory = Path(directory)
    bundle = json.loads((directory / 'bundle.json').read_text())
    if independent and any(d <= '2025-12-31' for d in bundle['forecast_dates']):
        raise ValueError('Previously inspected 2021–2025 cannot be reused as independent evaluation.')
    if bundle['model_sha256'] != file_sha256(config.MODEL_PATH):
        raise ValueError('Forecast does not use the selected frozen model.')
    latitude = np.load(config.PROCESSED_DIR / 'latitude_kerala_2021_2025.npy')
    longitude = np.load(config.PROCESSED_DIR / 'longitude_kerala_2021_2025.npy')
    if bundle.get('status') != 'success':
        return bundle, None, latitude, longitude
    risk = np.load(directory / 'risk_classes.npy', allow_pickle=False)
    if risk.shape != (len(bundle['forecast_dates']), len(latitude), len(longitude)) or not np.isin(risk, [0, 1, 2, 255]).all():
        raise ValueError('Invalid forecast classes or grid.')
    expected = bundle.get('array_sha256', {}).get('risk_classes.npy')
    if not expected or file_sha256(directory / 'risk_classes.npy') != expected:
        raise ValueError('Forecast array integrity check failed.')
    return bundle, risk, latitude, longitude


def binary_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, bool), np.asarray(predicted, bool)
    tp = int((actual & predicted).sum()); fp = int((~actual & predicted).sum()); fn = int((actual & ~predicted).sum()); tn = int((~actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision, 'recall': recall,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def match_fields(records, forecasts):
    """One verified record per horizon; overlapping origins are not independent."""
    matches = []; unmatched = []; unique = set()
    for record in records:
        count = 0
        for bundle, risk, lat, lon in forecasts:
            if record['date'] not in bundle['forecast_dates']:
                continue
            index = grid_index(lat, lon, record['latitude'], record['longitude'])
            if index is None:
                continue
            horizon = bundle['forecast_dates'].index(record['date'])
            key = (record['record_id'], horizon)
            if key in unique:
                raise ValueError('Duplicate forecast for the same field record/horizon; select one reproducible origin.')
            unique.add(key)
            prediction = int(risk[horizon][index]) if risk is not None else 255
            # Insufficient/low coverage maps are not usable field forecasts.
            if bundle.get('data_quality', {}).get('state') != 'observed':
                prediction = 255
            input_end = bundle['input_dates'][-1]
            if input_end >= record['date']:
                raise ValueError('Forecast input overlaps the field target date.')
            matches.append({'record_id': record['record_id'], 'date': record['date'],
                'latitude': record['latitude'], 'longitude': record['longitude'],
                'forecast_id': bundle.get('forecast_id'), 'model_prediction': None if prediction == 255 else prediction,
                'field_observation': record['confirmed_hab'], 'species': record.get('species'),
                'severity': record.get('severity'), 'toxin_value': record.get('toxin_value'),
                'toxin_units': record.get('toxin_units'), 'verification_method': record['verification_method'],
                'source_reference': record['source_reference'], 'verified_by': record.get('verified_by'), 'horizon_days': horizon + 1,
                'available': prediction != 255, 'coverage': bundle.get('data_quality'),
                'lead_time_verified': bool(bundle.get('source_metadata_sha256'))})
            count += 1
        if not count:
            unmatched.append({'record_id': record['record_id'], 'reason': 'No target-date forecast within the supported grid.'})
    by_horizon = {}
    for day in range(1, 6):
        rows = [m for m in matches if m['horizon_days'] == day]
        usable = [m for m in rows if m['available']]
        by_horizon[str(day)] = {'matched_records': len(rows), 'available_records': len(usable),
            'abstentions': len(rows) - len(usable),
            'available_forecast_metrics': binary_metrics([m['field_observation'] for m in usable], [m['model_prediction'] > 0 for m in usable]) if usable else None,
            'event_detection_including_abstentions': binary_metrics([m['field_observation'] for m in rows], [m['available'] and m['model_prediction'] > 0 for m in rows]) if rows else None}
    return {'status': 'success' if any(m['available'] for m in matches) else 'insufficient_data',
            'target': 'Verified HAB presence compared with predicted proxy class >=1; proxy classes are not confirmed severity.',
            'matches': matches, 'unmatched': unmatched, 'by_horizon': by_horizon,
            'regional_metrics': None, 'regional_reason': 'Point observations do not establish regional event absence/presence fractions.',
            'limitation': 'Overlapping origins and repeated observations are correlated; metrics do not establish operational lead time without acquisition provenance.'}


def evaluate_arrays(actual, predicted, available, ocean, calibration):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    available, ocean = np.asarray(available), np.asarray(ocean, bool)
    if actual.shape != predicted.shape or actual.ndim != 3 or ocean.shape != actual.shape[1:] or available.shape != (len(actual),) or not np.isin(available, [0, 1]).all() or not ocean.any():
        raise ValueError('Require aligned daily labels/forecasts, binary availability and nonempty support.')
    if not np.isin(actual, [0, 1, 2, 255]).all() or not np.isin(predicted, [0, 1, 2, 255]).all():
        raise ValueError('Classes must be 0/1/2 or missing 255.')
    valid = (actual != 255) & (predicted != 255) & ocean & available[:, None, None].astype(bool)
    cm = confusion_matrix(actual[valid], predicted[valid])
    metrics = classification_metrics(cm) if valid.any() else None
    truth, warning = [], []
    unknown_days = 0
    for i, day in enumerate(actual):
        observed = (day != 255) & ocean
        if observed.sum() < .05 * ocean.sum():
            unknown_days += 1; continue
        observed_fraction = (day[observed] == 2).mean() * 100
        supported = (predicted[i] != 255) & ocean
        expected = warning_level(observed_fraction, calibration)
        level = warning_level((predicted[i][supported] == 2).mean() * 100, calibration) if available[i] and supported.any() else 'INSUFFICIENT DATA'
        truth.append(expected); warning.append(level)
    any_truth = np.array([t != 'NORMAL' for t in truth]); any_warning = np.array([w in ('ELEVATED RISK', 'STRONG BLOOM RISK') for w in warning])
    strong_truth = np.array([t == 'STRONG BLOOM RISK' for t in truth]); strong_warning = np.array([w == 'STRONG BLOOM RISK' for w in warning])
    return {'status': 'success' if valid.any() else 'insufficient_data', 'pixel_metrics': metrics,
            'confusion_matrix': cm.tolist(), 'available_pixel_count': int(valid.sum()),
            'target_observation_coverage_percent': float(((actual != 255) & ocean).sum() / (len(actual) * ocean.sum()) * 100) if len(actual) else 0.,
            'forecast_available_days': int(available.sum()), 'total_days': len(actual), 'unknown_regional_truth_days': unknown_days,
            'regional_any_risk': binary_metrics(any_truth, any_warning) if len(truth) else None,
            'regional_strong': binary_metrics(strong_truth, strong_warning) if len(truth) else None,
            'regional_abstentions': sum(w == 'INSUFFICIENT DATA' for w in warning)}


def independent_report(directory):
    """Evaluate supplied actual adjudicated classes, without fitting any threshold."""
    from experiments.finalize import verify_final
    verify_final()
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('independent') is not True or not metadata.get('label_definition') or not metadata.get('label_provenance'):
        raise ValueError('Independent declaration and actual label definition/provenance required.')
    with np.load(directory / 'evaluation.npz', allow_pickle=False) as data:
        dates, lat, lon = data['dates'].astype('datetime64[D]'), data['latitude'], data['longitude']
        actual, predicted, available, ocean = data['actual'], data['predicted'], data['available'], data['ocean']
    if np.isnat(dates).any() or np.any(dates <= np.datetime64('2025-12-31')) or len(dates) != len(actual) or np.any(np.diff(dates).astype(int) <= 0):
        raise ValueError('Unique increasing independent dates after 2025 are required.')
    expected_lat = np.load(config.PROCESSED_DIR / 'latitude_kerala_2021_2025.npy')
    expected_lon = np.load(config.PROCESSED_DIR / 'longitude_kerala_2021_2025.npy')
    if lat.shape != expected_lat.shape or lon.shape != expected_lon.shape or not np.allclose(lat, expected_lat, atol=1e-7, rtol=0) or not np.allclose(lon, expected_lon, atol=1e-7, rtol=0):
        return {'status': 'unsupported_region', 'reason': 'Frozen model/grid supports Kerala only. Supply a subset mask on its actual grid; out-of-grid regions require separate development.'}
    model_hash = file_sha256(config.MODEL_PATH)
    trained_support = np.load(config.LABEL_DIR / 'stable_ocean_mask.npy').astype(bool)
    if ocean.shape != trained_support.shape or not np.isin(ocean, [0, 1]).all() or np.any(ocean.astype(bool) & ~trained_support):
        raise ValueError('Evaluation region must be a binary subset of frozen training support.')
    if metadata.get('model_sha256') != model_hash or metadata.get('decision_rule_sha256') != file_sha256(config.RUN_DIR / 'calibration.json'):
        raise ValueError('Predictions must originate from the frozen model/decision artifacts.')
    # Per-forecast provenance ties supplied predictions to actual cached model
    # outputs; claiming a model hash alone is not verification.
    forecast_dirs = metadata.get('forecast_directories', [])
    if len(forecast_dirs) != len(dates):
        raise ValueError('One provenance-verified horizon-one forecast directory per evaluation date required.')
    for i, path in enumerate(forecast_dirs):
        bundle, risk, _, _ = load_forecast(path, independent=True)
        if bundle['forecast_dates'][0] != str(dates[i]) or bundle['input_dates'][-1] >= str(dates[i]):
            raise ValueError('Forecast target/input date mismatch.')
        supported = bundle.get('data_quality', {}).get('state') == 'observed' and risk is not None
        if bool(available[i]) != supported or (supported and not np.array_equal(predicted[i], risk[0])) or (not supported and np.any(predicted[i] != 255)):
            raise ValueError('Supplied predictions/availability differ from source forecast.')
    calibration = json.loads((config.RUN_DIR / 'calibration.json').read_text())
    result = evaluate_arrays(actual, predicted, available, ocean, calibration)
    result.update(model_sha256=model_hash, labels=metadata['label_definition'], label_provenance=metadata['label_provenance'],
                  input_sha256=file_sha256(directory / 'evaluation.npz'), metadata_sha256=file_sha256(directory / 'metadata.json'),
                  evaluation_period=[str(dates[0]), str(dates[-1])] if len(dates) else None,
                  trained=False, tuned=False, limitation='Proxy-class evaluation is distinct from field-confirmed binary HAB validation.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['field', 'independent'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--forecast-dir', type=Path, action='append', default=[])
    parser.add_argument('--purpose', choices=['development', 'independent'], default='independent')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; preserve reports and choose a new filename.')
    try:
        if args.mode == 'field':
            records = load_fields(args.data, args.purpose)
            forecasts = [load_forecast(p, args.purpose == 'independent') for p in args.forecast_dir]
            result = match_fields(records, forecasts)
            result.update(field_sha256=file_sha256(args.data), purpose=args.purpose)
        else:
            result = independent_report(args.data)
    except FileNotFoundError:
        result = {'status': 'unavailable', 'reason': 'Required real observations, forecasts or fitted artifacts are missing.'}
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {'status': 'invalid_input', 'reason': str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
