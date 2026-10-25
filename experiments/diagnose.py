"""Explain elevated-class failures using 2024 data only."""
import argparse
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from preprocessing.data_utils import month_numbers
from prediction.metrics import classify, classification_metrics, confusion_matrix
import config
from experiments.common import BASELINE_DIR, EXPERIMENT_DIR, verify_baseline, baseline_pipeline, load_model, collect_predictions, cached_samples, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached', action='store_true')
    args = parser.parse_args()
    verify_baseline()
    pipeline = baseline_pipeline()
    directory = EXPERIMENT_DIR / 'diagnostics_2024'
    cache = directory / 'predictions'
    if not args.cached:
        model = load_model(BASELINE_DIR / 'model.keras')
        collect_predictions(model, pipeline, cache)
    actual, scores, cell_observations, window_coverage = [], [], [], []
    examples = {'elevated_to_normal': [], 'elevated_to_strong': [], 'elevated_near_strong_boundary': []}
    threshold = read_json(BASELINE_DIR / 'freeze.json')['decision_rule']['strong_threshold']
    for row, p, target in cached_samples(cache, pipeline):
        valid = target != 255
        actual.append(target[valid]); scores.append(p[valid])
        indices = np.arange(row.input_start_idx, row.input_end_idx + 1)
        _, observed = pipeline.valid('chlorophyll', indices)
        cell_count = observed.sum(0)
        mean_coverage = float(observed.sum() / (5 * pipeline.ocean.sum()) * 100)
        cell_observations.append(np.broadcast_to(cell_count, target.shape)[valid])
        window_coverage.append(np.full(int(valid.sum()), mean_coverage, dtype=np.float32))
        predicted = classify(p, threshold)
        groups = {'elevated_to_normal': (target == 1) & (predicted == 0),
                  'elevated_to_strong': (target == 1) & (predicted == 2),
                  'elevated_near_strong_boundary': (target == 1) & (np.abs(p[..., 2] - threshold) < .01)}
        for name, selection in groups.items():
            points = np.argwhere(selection)
            if not len(points):
                continue
            values = p[selection]
            order = np.argsort(-values[:, 1] if name != 'elevated_near_strong_boundary' else np.abs(values[:, 2] - threshold))[:2]
            for position in order:
                day, y, x = map(int, points[position])
                environment = {}
                for feature in config.CHANNEL_NAMES:
                    v, mask = pipeline.valid(feature, np.array([row.input_end_idx]))
                    environment[feature] = float(v[0, y, x]) if mask[0, y, x] else None
                example = {'input_end': str(row.input_end), 'target_date': str(pipeline.dates[row.target_start_idx + day]),
                           'horizon': day + 1, 'latitude': float(pipeline.latitude[y]), 'longitude': float(pipeline.longitude[x]),
                           'true_class': 1, 'predicted_class': int(predicted[day, y, x]), 'scores': p[day, y, x].tolist(),
                           'input_chlorophyll_observed_days_at_cell': int(cell_count[y, x]),
                           'mean_input_chlorophyll_coverage_percent': mean_coverage, 'last_input_environment': environment}
                examples[name].append(example)
    y = np.concatenate(actual); p = np.concatenate(scores)
    cell = np.concatenate(cell_observations); coverage = np.concatenate(window_coverage)
    rules = {'direct_argmax': p.argmax(-1), 'current_rule': classify(p, threshold)}
    results = {name: classification_metrics(confusion_matrix(y, predicted)) for name, predicted in rules.items()}
    conditional = {}
    curves = {}
    for true_class in range(3):
        s = p[y == true_class]
        conditional[str(true_class)] = {'pixels': int(len(s)), 'score_quantiles_by_output_class': np.percentile(s, [5, 25, 50, 75, 95], axis=0).tolist(),
                                        'fraction_elevated_beats_normal': float((s[:, 1] > s[:, 0]).mean()),
                                        'fraction_elevated_raw_argmax': float((s.argmax(-1) == 1).mean())}
        precision, recall, thresholds = precision_recall_curve(y == true_class, p[:, true_class])
        positions = np.linspace(0, len(thresholds) - 1, min(200, len(thresholds)), dtype=int)
        curves[str(true_class)] = {'average_precision': float(average_precision_score(y == true_class, p[:, true_class])),
                                 'thresholds': thresholds[positions].tolist(), 'precision': precision[positions].tolist(), 'recall': recall[positions].tolist()}
    strata = {}
    for name, keep in {'no_input_chlorophyll_at_cell': cell == 0, 'some_input_chlorophyll_at_cell': cell > 0,
                       'window_coverage_below_5_percent': coverage < 5, 'window_coverage_at_least_5_percent': coverage >= 5}.items():
        strata[name] = classification_metrics(confusion_matrix(y[keep], rules['current_rule'][keep]))
    for name, rows in examples.items():
        rows.sort(key=lambda r: -r['scores'][1] if name != 'elevated_near_strong_boundary' else abs(r['scores'][2] - threshold))
        examples[name] = rows[:5]
    training_counts = read_json(BASELINE_DIR / 'sequences/manifest.json')['splits']['train']['class_counts']
    summary = {'period': '2024 development only', 'threshold': threshold, 'training_unique_label_counts': training_counts,
               'validation_pixel_forecast_class_counts': np.bincount(y, minlength=3).tolist(),
               'predicted_class_counts': {name: np.bincount(v, minlength=3).tolist() for name, v in rules.items()},
               'metrics': results, 'conditional_scores': conditional, 'precision_recall': curves, 'coverage_strata': strata,
               'examples': examples, 'months_unique_label_counts': {str(m): [int((pipeline.labels[(pipeline.dates >= np.datetime64('2024-01-01')) & (pipeline.dates <= np.datetime64('2024-12-31')) & (month_numbers(pipeline.dates) == m)] == c).sum()) for c in range(3)] for m in range(1, 13)},
               'interpretation': 'Raw argmax and elevated-versus-normal scores identify learned score suppression; comparing the current rule quantifies additional post-processing suppression. Coverage associations are observational, not causal evidence. Proxy labels and class imbalance may contribute; diagnostics alone cannot prove causality or a capacity limit.'}
    write_json(directory / 'diagnostics.json', summary)
    lines = ['# Why elevated risk is not detected — 2024 only', '',
             f"Valid pixel forecasts: {len(y):,}. True counts normal/elevated/strong: {summary['validation_pixel_forecast_class_counts']}.",
             f"Training unique-label counts: {training_counts}.", '',
             '| Rule | Predicted normal/elevated/strong | Macro F1 | Elevated precision | Elevated recall | Elevated F1 |', '|---|---|---:|---:|---:|---:|']
    for name, result in results.items():
        c = result['classes'][1]
        lines.append(f"| {name} | {summary['predicted_class_counts'][name]} | {result['macro_f1']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} |")
    q = conditional['1']
    lines += ['', f"On actual elevated pixels, elevated beats normal in {q['fraction_elevated_beats_normal'] * 100:.4f}% of cases, and is raw argmax in {q['fraction_elevated_raw_argmax'] * 100:.4f}%.",
              f"Median output scores on elevated pixels (normal/elevated/strong): {q['score_quantiles_by_output_class'][2]}.", '',
              'The raw score ranking establishes whether elevated is already suppressed before post-processing. The current strong override may additionally take ambiguous/elevated pixels, but cannot explain suppression that is present under direct argmax. Imbalance and missing predictive information are hypotheses to test with controlled training, not proven causes. Labels represent adjacent anomaly quantiles rather than confirmed biological events; intermediate-class overlap is plausible. Nothing here establishes that increasing model capacity is necessary.', '',
              'Class precision–recall curves, coverage-stratified metrics, monthly distributions and actual elevated failure examples with dates, scores, cell observations and environmental inputs are in diagnostics.json. No 2025 predictions were generated.', '', summary['interpretation']]
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
