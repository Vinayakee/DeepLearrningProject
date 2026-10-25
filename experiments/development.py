"""Fixed-rule weight comparisons and then separate 2024 decision experiments."""
import argparse
import numpy as np
from prediction.decision import apply_rule
from prediction.metrics import classification_metrics, confusion_matrix
from experiments.common import EXPERIMENT_DIR, BASELINE_DIR, baseline_pipeline, cached_samples, read_json, write_json, verify_baseline, utc_now


def cache_path(name):
    return EXPERIMENT_DIR / 'diagnostics_2024/predictions' if name == 'baseline' else EXPERIMENT_DIR / name / 'validation'


def valid_scores(name, pipeline):
    actual, scores = [], []
    for _, p, target in cached_samples(cache_path(name), pipeline):
        valid = target != 255
        actual.append(target[valid]); scores.append(p[valid])
    return np.concatenate(actual), np.concatenate(scores)


def metrics(y, p, rule):
    cm = confusion_matrix(y, apply_rule(p, rule))
    result = classification_metrics(cm)
    result['false_warning_pixel_rate'] = float(cm[0, 1:].sum() / max(cm[0].sum(), 1))
    result['missed_risk_pixel_rate'] = float(cm[1:, 0].sum() / max(cm[1:].sum(), 1))
    result['decision_rule'] = rule
    return result


def ranking(result):
    rare = result['classes'][1:]
    return (result['macro_f1'], min(c['f1'] for c in rare), sum(c['precision'] for c in rare), -result['false_warning_pixel_rate'])


def rules_grid():
    yield {'type': 'argmax'}
    for threshold in np.linspace(.1, .9, 33):
        yield {'type': 'strong_override', 'strong_threshold': float(threshold)}
    for elevated in np.arange(-1.5, 2.01, .25):
        for strong in np.arange(-1.5, 2.01, .25):
            yield {'type': 'score_ratios', 'scales': [1., float(np.exp(elevated)), float(np.exp(strong))]}


def table(rows):
    lines = ['| Experiment/rule | Macro F1 | Elevated precision | Elevated recall | Elevated F1 | Strong precision | Strong recall | Strong F1 | Normal F1 | False-warning pixel rate |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in rows:
        e, s = r['classes'][1:]
        lines.append(f"| {name} | {r['macro_f1']:.4f} | {e['precision']:.4f} | {e['recall']:.4f} | {e['f1']:.4f} | {s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} | {r['classes'][0]['f1']:.4f} | {r['false_warning_pixel_rate']:.4f} |")
    return lines


def weight_comparison(pipeline):
    results = {}
    baseline_rule = read_json(BASELINE_DIR / 'freeze.json')['decision_rule']
    for name in ['baseline', 'inverse', 'intermediate']:
        y, p = valid_scores(name, pipeline)
        results[name] = {key: metrics(y, p, rule) for key, rule in {'argmax': {'type': 'argmax'}, 'legacy_fixed': baseline_rule}.items()}
        if name != 'baseline':
            training = read_json(EXPERIMENT_DIR / name / 'training_config.json')
            if training['seasonal_features'] or not training.get('epochs_completed'):
                raise ValueError('Weight experiment is incomplete or changed seasonal inputs.')
            results[name]['training'] = training
    selected = max(results, key=lambda n: ranking(results[n]['argmax']))
    output = {'period': '2024 only', 'criterion': 'Highest macro F1 under the SAME direct-argmax reference rule. Ties: minimum rare-class F1, rare-class precision, lower false-warning pixel rate. Frozen legacy rule also reported without retuning.',
              'selected_weight_experiment': selected, 'results': results}
    write_json(EXPERIMENT_DIR / 'weight_comparison.json', output)
    lines = ['# Controlled class-weight comparison — 2024', '', output['criterion'], '']
    lines += table([(name + '/argmax', r['argmax']) for name, r in results.items()])
    lines += ['', 'Frozen legacy threshold comparison (no threshold change):', '']
    lines += table([(name + '/legacy_fixed', r['legacy_fixed']) for name, r in results.items()])
    lines += ['', f'Selected weighting candidate: {selected}. Architecture, input features, labels, seed, optimizer, maximum epochs and callbacks were fixed. Weights are normalized to the same expected training-pixel loss mass as baseline. Argmax provides a common reference that is not calibrated to one model; production behavior remains unchanged until final selection.']
    (EXPERIMENT_DIR / 'weight_comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def rule_comparison(names, pipeline):
    if not (EXPERIMENT_DIR / 'weight_comparison.json').exists():
        raise ValueError('Establish the controlled weight comparison first.')
    existing = EXPERIMENT_DIR / 'decision_comparison.json'
    comparisons = read_json(existing)['comparisons'] if existing.exists() else {}
    for name in names:
        if name in comparisons:
            print(f'{name}: existing frozen 2024 rule retained; no retuning.', flush=True)
            continue
        y, p = valid_scores(name, pipeline)
        all_results = [metrics(y, p, rule) for rule in rules_grid()]
        best = max(all_results, key=ranking)
        comparisons[name] = {'best': best, 'sweep': all_results}
        print(f"{name}: best 2024 macro F1 {best['macro_f1']:.4f}; rule {best['decision_rule']}", flush=True)
    report = {'period': '2024 only', 'created_at_utc': utc_now(),
              'criterion': 'Validation macro F1; ties broken by minimum rare-class F1, summed rare-class precision, then false-warning pixel rate. Same finite predeclared grid for each checkpoint.',
              'grid': 'argmax, 33 strong override thresholds 0.1..0.9, 225 positive score-ratio boundaries exp(-1.5..2, step .25). Scores themselves are not recalibrated probabilities.',
              'comparisons': comparisons}
    write_json(EXPERIMENT_DIR / 'decision_comparison.json', report)
    lines = ['# Separate decision-rule comparison — 2024', '', report['criterion'], '', report['grid'], '']
    lines += table([(name + '/selected rule', r['best']) for name, r in comparisons.items()])
    lines += ['', 'Each candidate checkpoint was fixed during this rule experiment. These validation-fitted boundaries are frozen for each model; no 2025 data were loaded. Final configuration selection happens after coverage/regional analysis.']
    (EXPERIMENT_DIR / 'decision_comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['weights', 'rules'])
    parser.add_argument('--names', nargs='+', default=['baseline', 'inverse', 'intermediate'])
    args = parser.parse_args()
    verify_baseline()
    pipeline = baseline_pipeline()
    if args.phase == 'weights':
        weight_comparison(pipeline)
    else:
        rule_comparison(args.names, pipeline)


if __name__ == '__main__':
    main()
