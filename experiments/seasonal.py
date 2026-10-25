"""Evaluate calendar inputs with weighting and decision boundaries held fixed."""
import argparse
from experiments.common import EXPERIMENT_DIR, read_json, write_json, baseline_pipeline, verify_baseline
from experiments.development import valid_scores, metrics, table

CRITERION = {'minimum_macro_f1_gain': .01, 'maximum_elevated_f1_decrease': .01,
             'description': 'Retain calendar inputs only for >=0.01 absolute validation macro-F1 gain, without >0.01 elevated-F1 loss, under the baseline selected rule held fixed. All inputs/labels/weights/training settings otherwise fixed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare', action='store_true')
    args = parser.parse_args()
    verify_baseline()
    if args.declare:
        if not (EXPERIMENT_DIR / 'decision_comparison.json').exists():
            raise ValueError('Finish weight and boundary comparisons first.')
        write_json(EXPERIMENT_DIR / 'seasonal_criterion.json', CRITERION)
        return
    criterion = read_json(EXPERIMENT_DIR / 'seasonal_criterion.json')
    pipeline = baseline_pipeline()
    baseline_rule = read_json(EXPERIMENT_DIR / 'decision_comparison.json')['comparisons']['baseline']['best']['decision_rule']
    results = {}
    for name in ['baseline', 'seasonal_root']:
        y, p = valid_scores(name, pipeline)
        results[name] = {key: metrics(y, p, rule) for key, rule in {'argmax': {'type': 'argmax'}, 'baseline_boundary_fixed': baseline_rule}.items()}
    original, seasonal = [results[n]['baseline_boundary_fixed'] for n in ['baseline', 'seasonal_root']]
    gain = seasonal['macro_f1'] - original['macro_f1']
    retained = gain >= criterion['minimum_macro_f1_gain'] and seasonal['classes'][1]['f1'] >= original['classes'][1]['f1'] - criterion['maximum_elevated_f1_decrease']
    report = {'period': '2024 only', 'criterion': criterion, 'macro_f1_gain': gain, 'retained': bool(retained), 'results': results,
              'rule_held_fixed': baseline_rule, 'features': 'sin/cos(2*pi*month/12) appended per input date. Availability remains channels 7..13; no future labels/dates as inputs.'}
    write_json(EXPERIMENT_DIR / 'seasonal_comparison.json', report)
    lines = ['# Controlled seasonal-feature experiment', '', criterion['description'], ''] + table([(n + '/' + rule, r) for n, values in results.items() for rule, r in values.items()])
    lines += ['', f'Calendar inputs {"retained" if retained else "rejected"}; fixed-boundary macro-F1 gain {gain:.4f}. Architecture filters, label definition and square-root weights stayed fixed.']
    (EXPERIMENT_DIR / 'seasonal_comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)

if __name__ == '__main__':
    main()
