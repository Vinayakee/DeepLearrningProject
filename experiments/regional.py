"""2024 hindcast warning evaluation; unknown target observations remain unknown."""
import argparse
import numpy as np
import pandas as pd
import config
from prediction.coverage import assess_coverage, DEFAULT_POLICY
from prediction.decision import apply_rule
from prediction.regional import regional_summary, warning_level, binary_warning_metrics
from preprocessing.data_utils import month_numbers
from experiments.common import EXPERIMENT_DIR, BASELINE_DIR, baseline_pipeline, cached_samples, read_json, write_json, verify_baseline
from experiments.development import cache_path

OLD_POLICY = {**DEFAULT_POLICY, 'minimum_chlorophyll_observed_days': 1}


def rows_for_cache(directory, pipeline, rule, calibration, policy, development=True):
    rows = []
    for sample, p, target in cached_samples(directory, pipeline, development=development):
        indices = np.arange(sample.input_start_idx, sample.input_end_idx + 1)
        input_dates = pipeline.dates[indices]
        quality = assess_coverage(pipeline.transform(indices), pipeline.ocean, input_dates, policy)
        dates = pipeline.dates[sample.target_start_idx:sample.target_end_idx + 1]
        support = pipeline.ocean[None] & np.isfinite(pipeline.climatology[month_numbers(dates) - 1])
        risk = apply_rule(p, rule)
        environment = {}
        for name in config.CHANNEL_NAMES:
            values, valid = pipeline.valid(name, np.array([sample.input_end_idx]))
            environment[name] = float(values[valid].mean()) if valid.any() else None
        for i, target_date in enumerate(dates):
            observed = target[i] != 255
            observed_count = int(observed.sum())
            observed_coverage = observed_count / pipeline.ocean.sum() * 100
            truth_qualified = observed_coverage >= policy['minimum_target_coverage_percent']
            actual_fraction = float((target[i] == 2).sum() / observed_count * 100) if observed_count else None
            summary = regional_summary(risk[i], support[i], calibration)
            available = quality['state'] == 'observed'
            mask = support[i]
            rows.append({'input_end_date': str(input_dates[-1]), 'target_date': str(target_date), 'horizon': i + 1,
                         'month': int(month_numbers([target_date])[0]), 'available': available, 'input_state': quality['state'],
                         'input_coverage_percent': quality['chlorophyll_mean_input_coverage_percent'],
                         'input_valid_cell_days': quality['valid_observations'],
                         'input_chlorophyll_observed_days': quality['features']['chlorophyll']['observed_input_days'],
                         'truth_qualified': bool(truth_qualified), 'actual_valid_pixels': observed_count,
                         'actual_coverage_percent': float(observed_coverage), 'actual_strong_percent': actual_fraction,
                         'actual_level': warning_level(actual_fraction, calibration) if truth_qualified else 'UNKNOWN',
                         'predicted_strong_percent': summary['high_risk_percentage'], 'raw_predicted_level': summary['warning_level'],
                         'issued_level': summary['warning_level'] if available else 'INSUFFICIENT DATA',
                         'strong_mean_score': float(p[i, ..., 2][mask].mean()) if mask.any() else None,
                         'strong_max_score': float(p[i, ..., 2][mask].max()) if mask.any() else None,
                         **environment})
    return pd.DataFrame(rows)


def summarize_rows(frame):
    result = {'forecast_cases': len(frame), 'truth_qualified_cases': int(frame.truth_qualified.sum()),
              'unknown_truth_cases': int((~frame.truth_qualified).sum()), 'unavailable_cases': int((~frame.available).sum()),
              'insufficient_data_frequency': float((~frame.available).mean()) if len(frame) else None,
              'input_state_counts': {str(k): int(v) for k, v in frame.input_state.value_counts().items()},
              'mean_input_coverage_percent': float(frame.input_coverage_percent.mean()) if len(frame) else None,
              'warnings_without_qualified_truth': int((~frame.truth_qualified & frame.issued_level.isin(['ELEVATED RISK', 'STRONG BLOOM RISK'])).sum())}
    qualified = frame[frame.truth_qualified]
    for key, levels in [('any_warning', ['ELEVATED RISK', 'STRONG BLOOM RISK']), ('strong_warning', ['STRONG BLOOM RISK'])]:
        truth = qualified.actual_level.isin(levels)
        issued = qualified.issued_level.isin(levels)
        result[key] = binary_warning_metrics(truth, issued)
        result[key]['abstained_high_risk_days'] = int((truth & ~qualified.available).sum())
        warning_rows = qualified[issued]
        result[key]['mean_input_coverage_on_warning_days_percent'] = float(warning_rows.input_coverage_percent.mean()) if len(warning_rows) else None
        available = qualified[qualified.available]
        result[key]['available_input_cohort'] = binary_warning_metrics(available.actual_level.isin(levels), available.issued_level.isin(levels))
    return result


def evaluate_cache(directory, pipeline, rule, calibration, policy, output, development=True):
    frame = rows_for_cache(directory, pipeline, rule, calibration, policy, development)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'daily_cases.csv', index=False)
    result = {'policy': policy, 'decision_rule': rule,
              'daily_horizon_1': summarize_rows(frame[frame.horizon == 1]),
              'all_horizon_cases': summarize_rows(frame),
              'by_horizon': {str(h): summarize_rows(frame[frame.horizon == h]) for h in range(1, 6)},
              'methodology': 'Primary daily counts use horizon 1: one target day per origin. All-horizon counts are distinct forecast cases, not independent days. Actual regional strong fraction uses only observed proxy labels, qualified at >=5% marine support. Predicted fraction uses dense training-supported cells. P75/P90 thresholds are fixed from 2021-2023. Missing target observations are UNKNOWN. Unavailable-input high-risk days remain missed warnings, with abstention counts reported. Hindcast availability is assessed at origin, not using today\'s age; product delivery latency is unknown.'}
    write_json(output / 'regional_metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--names', nargs='+', default=['baseline', 'inverse', 'intermediate'])
    args = parser.parse_args()
    verify_baseline(); pipeline = baseline_pipeline()
    calibration = read_json(BASELINE_DIR / 'calibration.json')
    decisions = read_json(EXPERIMENT_DIR / 'decision_comparison.json')['comparisons']
    report_path = EXPERIMENT_DIR / 'regional_comparison_2024.json'
    results = read_json(report_path) if report_path.exists() else {}
    for name in args.names:
        rules = {'selected_rule': decisions[name]['best']['decision_rule']}
        if name == 'baseline':
            rules['original_rule'] = read_json(BASELINE_DIR / 'freeze.json')['decision_rule']
            rules['original_rule_shared_policy'] = rules['original_rule']
        for stage, rule in rules.items():
            if name + '/' + stage in results:
                continue
            # Original baseline is evaluated with its original coverage safeguard.
            policy = OLD_POLICY if name == 'baseline' and stage == 'original_rule' else DEFAULT_POLICY
            result = evaluate_cache(cache_path(name), pipeline, rule, calibration, policy, EXPERIMENT_DIR / name / 'regional_2024' / stage)
            results[name + '/' + stage] = result
            r = result['daily_horizon_1']['any_warning']
            print(f"{name}/{stage}: qualified days={r['qualified_days']}, TP={r['correctly_warned_high_risk_days']}, FP={r['false_warning_days']}, FN={r['missed_high_risk_days']}, precision={r['precision']:.4f}, recall={r['recall']:.4f}", flush=True)
    write_json(EXPERIMENT_DIR / 'regional_comparison_2024.json', results)
    original = pd.read_csv(EXPERIMENT_DIR / 'baseline/regional_2024/original_rule/daily_cases.csv')
    common_path = EXPERIMENT_DIR / 'regional_common_cohort_2024.json'
    common_comparisons = read_json(common_path) if common_path.exists() else {}
    for name in args.names:
        candidate = pd.read_csv(EXPERIMENT_DIR / name / 'regional_2024/selected_rule/daily_cases.csv')
        if not original[['input_end_date', 'target_date', 'horizon']].equals(candidate[['input_end_date', 'target_date', 'horizon']]):
            raise ValueError('Regional cohorts have different dates; do not compare by row position.')
        common = original.available & candidate.available & original.truth_qualified & candidate.truth_qualified & (original.horizon == 1)
        common_comparisons[name] = {'baseline': summarize_rows(original[common]), 'candidate': summarize_rows(candidate[common])}
    write_json(EXPERIMENT_DIR / 'regional_common_cohort_2024.json', common_comparisons)

if __name__ == '__main__':
    main()
