"""Freeze a configuration using 2024 pixel and regional evidence only."""
import numpy as np
from experiments.common import EXPERIMENT_DIR, BASELINE_DIR, read_json, write_json, utc_now, verify_baseline
from experiments.development import table
from prediction.coverage import DEFAULT_POLICY

CRITERION = {'macro_f1_tie_band': .005,
             'description': 'Macro F1 is primary. Candidates within 0.005 absolute F1 of the best form the near-tie set; choose highest regional any-warning F1, then minimum rare-class F1, lower false-warning frequency, fewer missed strong days, macro F1. Report all class-specific and warning tradeoffs. Seasonal inputs are eligible only after the predeclared controlled experiment retains them.'}


def main():
    verify_baseline()
    if (EXPERIMENT_DIR / 'selection.json').exists():
        raise ValueError('Selection is already frozen and will not be overwritten.')
    decisions = read_json(EXPERIMENT_DIR / 'decision_comparison.json')['comparisons']
    seasonal = read_json(EXPERIMENT_DIR / 'seasonal_comparison.json')
    regions = read_json(EXPERIMENT_DIR / 'regional_comparison_2024.json')
    if not (EXPERIMENT_DIR / 'false_alarm_analysis_2024.json').exists():
        raise ValueError('Finish the false-alarm analysis before selection.')
    names = ['baseline', 'inverse', 'intermediate']
    if seasonal['retained']:
        names += ['seasonal_root']
        if 'seasonal_combination' in decisions:
            names += ['seasonal_combination']
    candidates = {}
    for name in names:
        r = decisions[name]['best']; region = regions[name + '/selected_rule']['daily_horizon_1']
        training = read_json(EXPERIMENT_DIR / name / 'training_config.json')
        if name == 'baseline':
            training = {**training, 'weight_power': .5, 'seasonal_features': False}
        candidates[name] = {'pixel': r, 'regional_daily': region, 'weight_power': training['weight_power'], 'class_weights': training['class_weights'], 'seasonal_features': training['seasonal_features']}
    best_macro = max(r['pixel']['macro_f1'] for r in candidates.values())
    near = [name for name, result in candidates.items() if result['pixel']['macro_f1'] >= best_macro - CRITERION['macro_f1_tie_band']]
    def rank(name):
        r = candidates[name]; regional = r['regional_daily']['any_warning']; strong = r['regional_daily']['strong_warning']
        return (regional['f1'], min(c['f1'] for c in r['pixel']['classes'][1:]), -(regional['false_warning_frequency'] or 0), -strong['missed_high_risk_days'], r['pixel']['macro_f1'])
    chosen = max(near, key=rank)
    selected = candidates[chosen]
    configuration = {'experiment': chosen, 'weight_power': selected['weight_power'], 'class_weights': selected['class_weights'], 'seasonal_features': selected['seasonal_features'],
                     'channels': 16 if selected['seasonal_features'] else 14, 'filters': [8, 4], 'spatial_downsample': 2,
                     'decision_rule': selected['pixel']['decision_rule'], 'coverage_policy': DEFAULT_POLICY,
                     'regional_parameters': {key: read_json(BASELINE_DIR / 'calibration.json')[key] for key in ['regional_elevated_percent', 'regional_strong_percent', 'regional_method', 'regional_fit_days', 'regional_fit_split']},
                     'seed': 42, 'epochs_requested': 30, 'batch_size': 2, 'learning_rate': .001, 'clipnorm': 1.}
    report = {'frozen_at_utc': utc_now(), 'development_period': '2024 only', 'criterion': CRITERION, 'near_tie_candidates': near,
              'selected_configuration': configuration, 'candidates': candidates,
              'test_disclosure': '2025 was inspected in earlier work. It was excluded from this improvement development; the final frozen configuration is evaluated once. This is a previously inspected holdout, not an untouched test.'}
    write_json(EXPERIMENT_DIR / 'selection.json', report)
    lines = ['# Final configuration selection — 2024 only', '', CRITERION['description'], ''] + table([(name, r['pixel']) for name, r in candidates.items()])
    lines += ['', '| Candidate | Warning precision | Warning recall | Warning F1 | False-warning days | Missed strong days | Unavailable origins |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in candidates.items():
        a = r['regional_daily']['any_warning']; st = r['regional_daily']['strong_warning']
        lines.append(f"| {name} | {a['precision']:.4f} | {a['recall']:.4f} | {a['f1']:.4f} | {a['false_warning_days']} | {st['missed_high_risk_days']} | {r['regional_daily']['unavailable_cases']} |")
    lines += ['', f'Selected: **{chosen}**. Near-tie set: {near}. Calendar retained: {seasonal["retained"]}. Exact configuration and boundaries are frozen in selection.json before final retraining. Regional warnings remain a prototype; observed-target sampling and sparse satellite coverage prevent a claim of operational reliability.', '', report['test_disclosure']]
    (EXPERIMENT_DIR / 'selection.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)

if __name__ == '__main__':
    main()
