"""Paired 2024 frozen-model diagnostics for real native QC Sentinel gap filling."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.common import cached_samples
from experiments.finalize import verify_final
from preprocessing.pipeline import EnvironmentalPipeline, value_transform
from preprocessing.data_utils import file_sha256
from operations.sentinel_pilot import load_daily
from upgrade.satellite import compare_and_fill
from prediction.coverage import assess_coverage
from prediction.service import summarize_forecast
from prediction.metrics import confusion_matrix, classification_metrics
from prediction.evaluation import load_trained_model
from prediction.regional import warning_level, binary_warning_metrics
import config


def run(sentinel, output, start='2024-02-12', end='2024-02-15'):
    output = Path(output)
    if output.exists() or not ('2024-01-01' <= start <= end <= '2024-12-26'):
        raise ValueError('Choose a new output directory and complete 2024 development windows.')
    verify_final()
    pipeline = EnvironmentalPipeline()
    calibration = json.loads((config.RUN_DIR / 'calibration.json').read_text())
    counts = {n: np.zeros((3, 3), np.int64) for n in ('modis', 'modis_sentinel')}
    available_counts = {n: np.zeros((3, 3), np.int64) for n in counts}
    common_counts = {n: np.zeros((3, 3), np.int64) for n in counts}
    records, model = [], None
    for row, baseline, target in cached_samples(config.RUN_DIR / 'validation', pipeline):
        origin = str(pipeline.dates[row.input_end_idx])
        if not start <= origin <= end:
            continue
        indices = np.arange(row.input_start_idx, row.input_end_idx+1)
        sd, sv, sm, metadata, qc = load_daily(sentinel, pipeline.latitude, pipeline.longitude,
                                            forecast_start=pipeline.dates[row.target_start_idx])
        if (pipeline.dates[row.input_end_idx]-sd[-1]).astype(int) > config.MAX_DATA_AGE_DAYS:
            sv[:] = np.nan; sm[:] = False
            qc['source_age_gate'] = 'Outside runtime maximum source age; MODIS fallback.'
        candidate = pipeline.transform(indices)
        raw, raw_mask = pipeline.valid('chlorophyll', indices)
        aligned = np.full(raw.shape, np.nan, np.float32)
        aligned_mask = np.zeros(raw.shape, bool)
        for i, day in enumerate(sd):
            match = np.flatnonzero(pipeline.dates[indices] == day)
            if len(match):
                aligned[match[0]], aligned_mask[match[0]] = sv[i], sm[i]
        values, valid, provenance, agreement = compare_and_fill(raw, raw_mask, aligned, aligned_mask, pipeline.ocean)
        stats = pipeline.stats['channels']['chlorophyll']
        scaled = np.clip((value_transform(np.where(valid, values, 0), 'chlorophyll')-stats['median'])/stats['scale'], -8, 8)
        candidate[..., 0], candidate[..., 7] = np.where(valid, scaled, 0), valid
        if agreement['additional_cell_days']:
            if model is None:
                model, _ = load_trained_model()
            fused_scores = model(candidate[None], training=False).numpy()[0]
        else:
            fused_scores = baseline
        targets = pipeline.dates[row.target_start_idx:row.target_end_idx+1]
        support = pipeline.forecast_support(np.arange(row.target_start_idx, row.target_end_idx+1))
        record = {'input_end_date': origin, 'forecast_dates': [str(t) for t in targets],
                  'agreement': agreement, 'qc': qc, 'source_available_at_utc': metadata['available_at_utc']}
        for name, scores, X in [('modis', baseline, pipeline.transform(indices)), ('modis_sentinel', fused_scores, candidate)]:
            risk, days = summarize_forecast(scores, support, calibration)
            common = (target != 255) & support
            cm = confusion_matrix(target[common], risk[common]); counts[name] += cm
            coverage = assess_coverage(X, pipeline.ocean, pipeline.dates[indices], calibration['coverage_policy'])
            if coverage['state'] == 'observed':
                available_counts[name] += cm
            record[name] = {'coverage': coverage, 'days': days, 'conditional_pixel_metrics': classification_metrics(cm)}
        if all(record[name]['coverage']['state'] == 'observed' for name in counts):
            for name in counts:
                common_counts[name] += np.asarray(record[name]['conditional_pixel_metrics']['confusion_matrix'])
        observed = target[0] != 255
        truth_qualified = observed.sum() >= calibration['coverage_policy']['minimum_target_coverage_percent']*pipeline.ocean.sum()/100
        fraction = float((target[0] == 2).sum()/observed.sum()*100) if observed.any() else None
        record['horizon_one_truth'] = {'qualified': bool(truth_qualified), 'strong_fraction_percent': fraction,
                                       'level': warning_level(fraction, calibration) if truth_qualified else 'UNKNOWN'}
        records.append(record)
    if not records:
        raise ValueError('No matching cached development origins.')
    result = {'period': [start, end], 'model_sha256': file_sha256(config.MODEL_PATH),
              'sentinel_export_sha256': file_sha256(Path(sentinel) / 'observations.npz'),
              'decision_rule': calibration['decision_rule'], 'origins': records,
              'conditional_pixel_metrics': {n: classification_metrics(cm) for n, cm in counts.items()},
              'qualified_input_pixel_metrics': {n: classification_metrics(cm) for n, cm in available_counts.items()},
              'same_qualified_input_cohort': {n: classification_metrics(cm) for n, cm in common_counts.items()},
              'promoted': False, 'training_changed': False, 'test_2025_inference': False,
              'interpretation': 'Bounded development diagnostic, including conditional scores on withheld-input windows. Qualified-input cohorts differ and are not a controlled skill comparison. Too little real satellite support for promotion or retraining decisions.',
              'timing_limit': 'Sentinel native publication is checked; historical MODIS/environment source publication latency remains unverified.'}
    result['regional_horizon_one'] = {}
    qualified = [r for r in records if r['horizon_one_truth']['qualified']]
    for name in counts:
        result['regional_horizon_one'][name] = {}
        for label, levels in [('any_warning', ('ELEVATED RISK', 'STRONG BLOOM RISK')), ('strong_warning', ('STRONG BLOOM RISK',))]:
            truth = np.array([r['horizon_one_truth']['level'] in levels for r in qualified], bool)
            issued = np.array([r[name]['coverage']['state'] == 'observed' and r[name]['days'][0]['warning_level'] in levels for r in qualified], bool)
            result['regional_horizon_one'][name][label] = binary_warning_metrics(truth, issued)
    before, after = [result['conditional_pixel_metrics'][n] for n in counts]
    result['acceptance'] = {'promoted': False,
                            'macro_f1_change': after['macro_f1']-before['macro_f1'],
                            'elevated_f1_change': after['classes'][1]['f1']-before['classes'][1]['f1'],
                            'strong_f1_change': after['classes'][2]['f1']-before['classes'][2]['f1'],
                            'reason': 'One observed Sentinel day and four overlapping development origins cannot validate fusion generalization. No deployed setting is selected from this pilot.'}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sentinel-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--start', default='2024-02-12')
    parser.add_argument('--end', default='2024-02-15')
    args = parser.parse_args()
    result = run(args.sentinel_dir, args.output_dir, args.start, args.end)
    print(json.dumps({'origins': len(result['origins']), 'metrics': result['conditional_pixel_metrics'], 'promoted': False}, indent=2))


if __name__ == '__main__':
    main()
