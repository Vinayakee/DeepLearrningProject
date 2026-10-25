"""Calibrate on a declared fitting split; freeze before one held-out assessment."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from preprocessing.hab_sequences import load_dataset, write_json
from preprocessing.data_utils import file_sha256
from preprocessing.hab_labels import utc
from prediction.coverage import assess_coverage


def verified_run(directory):
    directory = Path(directory)
    record = json.loads((directory/'training.json').read_text())
    if record.get('schema') != 'hab-training-v1' or record.get('state') != 'trained' or not record.get('epochs_completed'):
        raise ValueError('A completed verified-HAB training run is required.')
    dataset = Path(record['dataset_dir'])
    for path, expected in ((dataset/'manifest.json', record['dataset_manifest_sha256']),
                           (directory/'model.keras', record['model_sha256']),
                           (directory/'baselines.json', record['baselines_sha256'])):
        if file_sha256(path) != expected:
            raise ValueError('HAB model, baselines or dataset manifest changed.')
    return record, load_dataset(dataset)


def load_model(directory):
    import tensorflow as tf
    import models.hab_convlstm  # registers serializable pooling and loss
    return tf.keras.models.load_model(Path(directory)/'model.keras', compile=False)


def zone_quality(X, pipeline, dates):
    return [assess_coverage(X, mask, dates) for mask in pipeline.zones]


def calibrated(scores, parameters):
    return expit(parameters['slope']*logit(np.clip(scores, 1e-7, 1-1e-7))+parameters['intercept'])


def binary_metrics(y, scores, threshold=.5):
    y, scores = np.asarray(y), np.asarray(scores, float)
    if y.shape != scores.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Aligned observed binary labels and finite scores required.')
    prediction = scores >= threshold
    tp, fp = int(((y == 1) & prediction).sum()), int(((y == 0) & prediction).sum())
    fn, tn = int(((y == 1) & ~prediction).sum()), int(((y == 0) & ~prediction).sum())
    precision, recall = tp/max(1, tp+fp), tp/max(1, tp+fn)
    return {'n': len(y), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision, 'recall': recall,
            'f1': 2*tp/max(1, 2*tp+fp+fn), 'csi': tp/max(1, tp+fp+fn),
            'brier': float(np.mean((scores-y)**2)) if len(y) else None,
            'pr_auc': float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else None,
            'roc_auc': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None}


def fit_decision(y, scores, allow_uncalibrated=False):
    if len(np.unique(y)) != 2 or not np.isfinite(scores).all():
        raise ValueError('Coverage-qualified calibration targets require both verified classes.')
    calibrator = LogisticRegression(C=1., random_state=42).fit(logit(np.clip(scores, 1e-7, 1-1e-7))[:, None], y)
    if calibrator.coef_[0, 0] <= 0 and not allow_uncalibrated:
        raise ValueError('Calibration found nonpositive discrimination; do not promote this candidate.')
    fitted = calibrator.coef_[0, 0] > 0
    parameters = {'slope': float(calibrator.coef_[0, 0]) if fitted else 1.,
                  'intercept': float(calibrator.intercept_[0]) if fitted else 0.,
                  'calibration_applied': bool(fitted)}
    p = calibrated(scores, parameters)
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2*precision*recall/np.maximum(precision+recall, 1e-12)
    # Predeclared objectives: high maximizes F1, low retains >=90% fitting recall.
    high = float(thresholds[max(range(len(thresholds)), key=lambda i: (f1[i], precision[i], thresholds[i]))])
    low = min(high, float(thresholds[recall >= .9].max()))
    return {**parameters, 'low_threshold': low, 'high_threshold': high,
            'threshold_objective': 'High: maximum calibration-split F1; Low: highest threshold with >=90% calibration-split recall, capped at High.',
            'fit_count': len(y), 'fit_positive_count': int(np.sum(y)),
            'interpretation': ('Platt fit conditional on verified observation sampling; held-out reliability and population representativeness must be assessed separately.'
                               if fitted else 'Baseline calibration had nonpositive discrimination; raw baseline scores retained and thresholds fitted without probability calibration.')}


def persistence(pipeline, records, row):
    scores = np.full(len(pipeline.zones), np.nan)
    first = str(pipeline.dates[row['input_start']])
    end = str(pipeline.dates[row['target_start']-1])
    cutoff = utc(str(pipeline.dates[row['target_start']])+'T00:00:00Z')
    for j, zone in enumerate(pipeline.spec['zones']):
        eligible = [r for r in records if r['zone_id'] == zone['id'] and first <= r['date'] <= end
                    and utc(r['available_at_utc']) <= cutoff]
        if eligible:
            latest_day = max(r['date'] for r in eligible)
            statuses = {r['confirmed_hab'] for r in eligible if r['date'] == latest_day}
            if len(statuses) == 1:
                scores[j] = float(statuses.pop() == 'true')
    return scores


def collect(directory, split):
    from training.train_hab import baseline_features
    record, (pipeline, labels, events, index, _) = verified_run(directory)
    directory = Path(directory)
    model = load_model(directory)
    baselines = json.loads((directory/'baselines.json').read_text())
    records = json.loads((Path(record['dataset_dir'])/'records.json').read_text())
    output = {key: [] for key in ('truth', 'scores', 'logistic', 'persistence', 'available', 'events', 'origins', 'targets')}
    for row in index:
        if row['split'] != split:
            continue
        first, target = row['input_start'], row['target_start']
        X = pipeline.transform(np.arange(first, target))
        dates = [str(d) for d in pipeline.dates[first:target]]
        available = np.array([q['state'] == 'observed' for q in zone_quality(X, pipeline, dates)])
        scores = model(X[None], training=False).numpy()[0] if available.any() else np.full((5, len(available)), np.nan)
        features = baseline_features(X, pipeline.zones)
        logistic = np.array([np.full(len(available), item['constant']) if 'constant' in item
                             else expit(features@np.asarray(item['coef'])+item['intercept']) for item in baselines['horizons']])
        for key, value in {'truth': labels[target:target+5], 'scores': scores, 'logistic': logistic,
                           'persistence': np.broadcast_to(persistence(pipeline, records, row), scores.shape),
                           'available': np.broadcast_to(available, scores.shape), 'events': events[target:target+5],
                           'origins': dates[-1], 'targets': [str(d) for d in pipeline.dates[target:target+5]]}.items():
            output[key].append(value)
    if not output['truth']:
        raise ValueError(f'No usable windows in {split}.')
    result = {key: np.asarray(value) for key, value in output.items()}
    result['zone_ids'] = np.asarray([zone['id'] for zone in pipeline.spec['zones']])
    result['label_audit'] = record['label_audit'][split]
    # Include records before the first split-safe forecast target; otherwise
    # an ongoing event can be mistaken for a newly starting event.
    result['event_first_recorded_day'] = {}
    for observation in records:
        if observation['confirmed_hab'] == 'true':
            event = observation['event_id']
            previous = result['event_first_recorded_day'].get(event, observation['date'])
            result['event_first_recorded_day'][event] = min(previous, observation['date'])
    return result


def calibrate(directory):
    directory = Path(directory)
    if any((directory/name).exists() for name in ('decision.json', 'freeze.json', 'test_started.json')):
        raise ValueError('Decisions are already fitted/frozen; use a separate run.')
    data = collect(directory, 'calibration')
    valid = (data['truth'] != 255) & data['available']
    decisions = {name: fit_decision(data['truth'][valid], data[key][valid], allow_uncalibrated=name == 'logistic')
                 for name, key in (('model', 'scores'), ('logistic', 'logistic'))}
    record = json.loads((directory/'training.json').read_text())
    decision = {'schema': 'hab-decision-v1', 'methods': decisions, 'fit_split': 'calibration',
                'model_sha256': record['model_sha256'], 'dataset_manifest_sha256': record['dataset_manifest_sha256'],
                'operational_warning_validated': False}
    # Exclusive creation prevents concurrent calibration from replacing a decision.
    with (directory/'decision.json').open('x') as handle:
        json.dump(decision, handle, indent=2, allow_nan=False)
    freeze = {'files': {name: file_sha256(directory/name) for name in
                        ('training.json', 'model.keras', 'baselines.json', 'decision.json')}}
    with (directory/'freeze.json').open('x') as handle:
        json.dump(freeze, handle, indent=2)
    return decision


def verify_decision(directory):
    directory = Path(directory)
    freeze = json.loads((directory/'freeze.json').read_text())
    for name in ('training.json', 'model.keras', 'baselines.json', 'decision.json'):
        if freeze['files'][name] != file_sha256(directory/name):
            raise ValueError('Frozen HAB run changed.')
    decision = json.loads((directory/'decision.json').read_text())
    record, data = verified_run(directory)
    if decision['model_sha256'] != record['model_sha256'] or decision['dataset_manifest_sha256'] != record['dataset_manifest_sha256']:
        raise ValueError('HAB decisions belong to another model/dataset.')
    return decision, data


def reliability(y, scores):
    result = []
    for i in range(10):
        mask = (scores >= i/10) & ((scores < (i+1)/10) if i < 9 else (scores <= 1))
        result.append({'lower': i/10, 'upper': (i+1)/10, 'n': int(mask.sum()),
                       'mean_probability': float(scores[mask].mean()) if mask.any() else None,
                       'observed_rate': float(y[mask].mean()) if mask.any() else None})
    return result


def summarize(data, scores, threshold):
    observed = data['truth'] != 255
    valid = observed & data['available'] & np.isfinite(scores)
    warnings = valid & (scores >= threshold)
    positive = observed & (data['truth'] == 1)
    events = sorted(set(data['events'][positive])-{''})
    detected = sum(bool(np.any(warnings & (data['events'] == event))) for event in events)
    lead = []
    for event in events:
        positions = np.argwhere(positive & (data['events'] == event))
        first = data.get('event_first_recorded_day', {}).get(str(event))
        origins = [data['origins'][i] for i, h, z in positions if warnings[i, h, z]]
        days = int((np.datetime64(first)-np.datetime64(min(origins))).astype(int)) if origins and first else None
        lead.append({'event_id': str(event), 'detected': bool(origins), 'first_recorded_day': first,
                     'first_warning_origin': str(min(origins)) if origins else None,
                     'nominal_days_before_first_recorded_day': days,
                     'timing': 'not_detected' if not origins else 'reference_unavailable' if days is None
                     else 'advance_warning' if days > 0 else 'same_day' if days == 0 else 'ongoing_event_detection'})
    return {'available_target_metrics': binary_metrics(data['truth'][valid], scores[valid], threshold),
            'observed_targets': int(observed.sum()), 'available_targets': int(valid.sum()),
            'abstentions': int((observed & ~valid).sum()),
            'positive_target_recall_including_abstentions': int((warnings & positive).sum())/max(1, int(positive.sum())),
            'events': len(events), 'detected_events': detected, 'event_recall': detected/len(events) if events else None,
            'events_warned_in_advance': sum(row['timing'] == 'advance_warning' for row in lead),
            'advance_warning_event_recall': sum(row['timing'] == 'advance_warning' for row in lead)/len(events) if events else None,
            'false_warning_forecast_zone_days': int((warnings & (data['truth'] == 0)).sum()),
            'event_lead_times': lead,
            'by_horizon': {str(h+1): binary_metrics(data['truth'][:, h][valid[:, h]], scores[:, h][valid[:, h]], threshold) for h in range(5)},
            'reliability_bins': reliability(data['truth'][valid], scores[valid])}


def block_intervals(data, scores, threshold, replicates=500):
    """Resample 14-calendar-day origin blocks with all zones/horizons together."""
    valid = (data['truth'] != 255) & data['available'] & np.isfinite(scores)
    origins = data['origins'].astype('datetime64[D]')
    block = ((origins-origins.min()).astype(int)//14)
    groups = np.unique(block)
    if len(groups) < 2:
        return {'state': 'insufficient_temporal_blocks', 'blocks': len(groups)}
    counts = []
    for group in groups:
        selected = valid & (block == group)[:, None, None]
        m = binary_metrics(data['truth'][selected], scores[selected], threshold)
        counts.append([m[k] for k in ('tp', 'fp', 'fn')])
    rng = np.random.default_rng(42)
    values = []
    for _ in range(replicates):
        tp, fp, fn = np.asarray(counts)[rng.integers(0, len(groups), len(groups))].sum(0)
        values.append([2*tp/max(1, 2*tp+fp+fn), tp/max(1, tp+fn)])
    bounds = np.percentile(values, [2.5, 97.5], axis=0)
    return {'state': 'estimated', 'blocks': len(groups), 'replicates': replicates, 'block_days': 14,
            'f1_95_percent': bounds[:, 0].tolist(), 'recall_95_percent': bounds[:, 1].tolist(),
            'limitation': 'Conditional on the selected model and observation coverage; long events may span blocks.'}


def evaluate(directory):
    directory = Path(directory)
    decision, _ = verify_decision(directory)
    # Irreversible scientific guard within this run; errors require a separate run.
    with (directory/'test_started.json').open('x') as handle:
        json.dump({'decision_sha256': file_sha256(directory/'decision.json')}, handle)
    data = collect(directory, 'test')
    methods = {}
    calibrated_scores = {}
    for name, key in (('model', 'scores'), ('logistic', 'logistic')):
        settings = decision['methods'][name]
        p = calibrated(data[key], settings)
        calibrated_scores[name] = p
        methods[name] = summarize(data, p, settings['high_threshold'])
        methods[name]['decision'] = settings
        methods[name]['temporal_block_intervals'] = block_intervals(data, p, settings['high_threshold'])
        methods[name]['by_zone'] = {
            str(zone): binary_metrics(data['truth'][:, :, j][valid], p[:, :, j][valid], settings['high_threshold'])
            for j, zone in enumerate(data['zone_ids'])
            for valid in [(data['truth'][:, :, j] != 255) & data['available'][:, :, j] & np.isfinite(p[:, :, j])]}
    # Persistence requires a published prior observation, not satellite coverage.
    persistence_data = {**data, 'available': np.isfinite(data['persistence'])}
    methods['persistence'] = summarize(persistence_data, data['persistence'], .5)
    common = data['available'] & np.isfinite(data['persistence']) & (data['truth'] != 255)
    common_metrics = {name: binary_metrics(data['truth'][common], p[common], decision['methods'][name]['high_threshold'])
                      for name, p in calibrated_scores.items()}
    common_metrics['persistence'] = binary_metrics(data['truth'][common], data['persistence'][common], .5)
    report = {'schema': 'hab-evaluation-v1', 'split': 'test', 'methods': methods,
              'common_persistence_cohort': common_metrics, 'label_audit': data['label_audit'],
              'assessed_event_count': methods['model']['events'],
              'events_without_split_safe_target_windows': data['label_audit']['events']-methods['model']['events'],
              'operational_warning_validated': False,
              'model_sha256': decision['model_sha256'], 'decision_sha256': file_sha256(directory/'decision.json'),
              'limitations': 'Observed zone-days only. Repeated horizons overlap. Unobserved days do not establish absence. Lead times use nominal archive dates, not verified product delivery or true event onset. Calibration fitting is not independent validation.'}
    write_json(directory/'evaluation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('calibrate', 'evaluate'))
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(json.dumps((calibrate if args.action == 'calibrate' else evaluate)(args.run_dir), indent=2))


if __name__ == '__main__':
    main()
