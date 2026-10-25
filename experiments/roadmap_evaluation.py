"""2024-only seasonal/persistence comparisons and dependence-aware uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import config
from experiments.common import cached_samples
from experiments.finalize import verify_final
from preprocessing.pipeline import EnvironmentalPipeline
from preprocessing.data_utils import month_numbers, file_sha256
from prediction.metrics import confusion_matrix, classification_metrics
from prediction.decision import apply_rule


def seasonal_predictions(labels, dates):
    """Cell/month majority learned strictly from training proxy labels."""
    result = np.full((12, *labels.shape[1:]), 255, np.uint8)
    months = month_numbers(dates)
    train = (dates >= np.datetime64(config.TRAIN_START)) & (dates <= np.datetime64(config.TRAIN_END))
    for month in range(1, 13):
        selected = labels[train & (months == month)]
        counts = np.stack([(selected == c).sum(0) for c in range(3)])
        valid = counts.sum(0) > 0
        result[month-1, valid] = counts.argmax(0)[valid]
    return result


def block_intervals(counts, repetitions=1000, block_days=14, seed=42):
    """Moving non-circular blocks on a complete origin-day calendar; keep pixels together."""
    counts = np.asarray(counts)
    if counts.ndim != 3 or counts.shape[1:] != (3, 3) or len(counts) < block_days or block_days < 5 or repetitions < 50:
        raise ValueError('Require daily 3x3 counts, >=5-day blocks and >=50 replicates.')
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(counts)-block_days+1, size=int(np.ceil(len(counts)/block_days)))
        indices = (starts[:, None] + np.arange(block_days)).ravel()[:len(counts)]
        m = classification_metrics(counts[indices].sum(0))
        samples.append([m['macro_f1'], m['classes'][2]['precision'], m['classes'][2]['recall']])
    values = np.quantile(samples, [.025, .975], axis=0)
    return {'method': 'Moving contiguous calendar-origin blocks; all pixels and horizons stay together.',
            'block_days': block_days, 'replicates': repetitions, 'seed': seed,
            'intervals_95_percent': {key: values[:, i].tolist() for i, key in enumerate(('macro_f1', 'strong_precision', 'strong_recall'))},
            'limitations': 'Development uncertainty conditional on the selected model and observed labels; longer seasonal dependence and model-selection uncertainty remain.'}


def event_report(frame):
    """Consecutive qualified risk days are proxy episodes; unknown days break episodes."""
    first = frame[frame.horizon == 1].sort_values('target_date')
    if first.target_date.duplicated().any():
        raise ValueError('Require one primary truth row per target day.')
    events, current = [], []
    for row in first.itertuples():
        risk = row.truth_qualified and row.actual_level in ('ELEVATED RISK', 'STRONG BLOOM RISK')
        if current and (not risk or np.datetime64(row.target_date)-np.datetime64(current[-1]) != np.timedelta64(1, 'D')):
            events.append(current); current = []
        if risk:
            current.append(row.target_date)
    if current:
        events.append(current)
    records = []
    for days in events:
        warnings = frame[frame.target_date.isin(days) & frame.available & frame.issued_level.isin(['ELEVATED RISK', 'STRONG BLOOM RISK'])]
        early = warnings[pd.to_datetime(warnings.input_end_date) < pd.Timestamp(days[0])]
        lead = int((pd.Timestamp(days[0])-pd.to_datetime(early.input_end_date).min()).days) if len(early) else None
        records.append({'start': days[0], 'end': days[-1], 'observed_days': len(days),
                        'detected': bool(len(warnings)), 'warned_before_onset': bool(len(early)),
                        'lead_days_before_onset': lead})
    return {'definition': 'Consecutive observed, qualified horizon-one chlorophyll-proxy risk days; unknown/missing days split episodes.',
            'episodes': records, 'episode_count': len(records), 'detected_episodes': sum(r['detected'] for r in records),
            'warned_before_onset': sum(r['warned_before_onset'] for r in records),
            'limitations': 'Not confirmed ecological HAB events. Uses saved hindcast origins, whose source-publication latency was not independently established.'}


def run(output, repetitions=1000):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new versioned evaluation directory.')
    verify_final()
    pipeline = EnvironmentalPipeline()
    cache = config.RUN_DIR / 'validation'
    calibration = json.loads((config.RUN_DIR / 'calibration.json').read_text())
    seasonal = seasonal_predictions(pipeline.labels, pipeline.dates)
    full = np.zeros((3, 3), np.int64)
    horizons = {n: np.zeros((5, 3, 3), np.int64) for n in ('model', 'seasonal', 'persistence')}
    origin_counts = np.zeros((366, 3, 3), np.int64)
    support_total, common_total, brier_sum = 0, 0, np.zeros(5)
    brier_count = np.zeros(5, np.int64)
    reliability = np.zeros((5, 10, 3), np.float64)
    for row, scores, target in cached_samples(cache, pipeline):
        if not ('2024-01-01' <= str(pipeline.dates[row.input_start_idx]) <= str(pipeline.dates[row.target_end_idx]) <= '2024-12-31'):
            raise ValueError('Only complete 2024 windows are accepted.')
        predicted = apply_rule(scores, calibration['decision_rule'])
        seasonal_target = seasonal[month_numbers(pipeline.dates[row.target_start_idx:row.target_end_idx+1])-1]
        previous = np.full(pipeline.ocean.shape, 255, np.uint8)
        for day in pipeline.labels[row.input_start_idx:row.input_end_idx+1]:
            seen = day != 255
            previous[seen] = day[seen]
        valid = target != 255
        common = valid & (seasonal_target != 255) & (previous[None] != 255)
        support_total += int(valid.sum()); common_total += int(common.sum())
        cm = confusion_matrix(target[valid], predicted[valid]); full += cm
        origin = int((pipeline.dates[row.input_end_idx]-np.datetime64('2024-01-01')).astype(int))
        origin_counts[origin] += cm
        for h in range(5):
            v = valid[h]; c = common[h]
            for name, p in [('model', predicted[h]), ('seasonal', seasonal_target[h]), ('persistence', previous)]:
                horizons[name][h] += confusion_matrix(target[h][c], p[c])
            p, truth = scores[h, ..., 2][v], (target[h][v] == 2)
            brier_sum[h] += np.square(p-truth).sum(); brier_count[h] += len(p)
            bins = np.minimum((p*10).astype(int), 9)
            reliability[h, :, 0] += np.bincount(bins, minlength=10)
            reliability[h, :, 1] += np.bincount(bins, weights=p, minlength=10)
            reliability[h, :, 2] += np.bincount(bins, weights=truth, minlength=10)
    frame = pd.read_csv(config.RUN_DIR / 'regional_2024/daily_cases.csv')
    result = {'period': '2024 development; no independent test claim', 'model_sha256': file_sha256(config.MODEL_PATH),
              'cache_sha256': file_sha256(cache / 'probabilities.npy'), 'test_inference_performed': False,
              'model_all_observed_targets': classification_metrics(full),
              'common_cohort': {'valid_pixel_forecasts': common_total, 'all_observed_targets': support_total,
                                'excluded_missing_baseline_support': support_total-common_total,
                                'definition': 'Identical observed targets with train-fitted cell/month climatology and at least one input-window label for persistence; missing baseline inputs abstain.'},
              'comparisons': {n: {'pooled': classification_metrics(cm.sum(0)), 'horizons': [classification_metrics(m) for m in cm]} for n, cm in horizons.items()},
              'uncertainty': block_intervals(origin_counts, repetitions), 'events': event_report(frame),
              'probability_by_horizon': []}
    for h in range(5):
        bins = [{'count': int(c), 'mean_score': float(p/c) if c else None, 'observed_frequency': float(t/c) if c else None} for c, p, t in reliability[h]]
        result['probability_by_horizon'].append({'day': h+1, 'strong_brier': float(brier_sum[h]/brier_count[h]) if brier_count[h] else None,
                                                 'strong_reliability_bins': bins})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-replicates', type=int, default=1000)
    args = parser.parse_args()
    result = run(args.output_dir, args.bootstrap_replicates)
    print(json.dumps({'common_cohort': result['common_cohort'], 'uncertainty': result['uncertainty'],
                      'macro_f1': {k: v['pooled']['macro_f1'] for k, v in result['comparisons'].items()}}, indent=2))


if __name__ == '__main__':
    main()
