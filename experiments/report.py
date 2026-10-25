"""Publish actual saved results without retraining, inference or tuning."""
import re
import shutil
import numpy as np
import config
from experiments.common import EXPERIMENT_DIR, BASELINE_DIR, read_json
from experiments.development import table
from experiments.finalize import FINAL_DIR, verify_final


def pixel_false_rate(result):
    cm = np.asarray(result['confusion_matrix'])
    return cm[0, 1:].sum() / cm[0].sum()


def main():
    verify_final()
    evaluation = read_json(FINAL_DIR / 'evaluation.json')
    log = (EXPERIMENT_DIR / 'final_tests.log').read_text()
    matches = re.findall(r'Ran (\d+) tests', log)
    if not matches or not log.rstrip().endswith('OK'):
        raise ValueError('A passing final complete test log is required before reporting completion.')
    count = int(matches[-1]); baseline = read_json(BASELINE_DIR / 'evaluation.json')
    selection = read_json(EXPERIMENT_DIR / 'selection.json'); training = read_json(FINAL_DIR / 'training_config.json')
    decisions = read_json(EXPERIMENT_DIR / 'decision_comparison.json')['comparisons']
    weights = read_json(EXPERIMENT_DIR / 'weight_comparison.json')['results']
    seasonal = read_json(EXPERIMENT_DIR / 'seasonal_comparison.json')
    combination = read_json(EXPERIMENT_DIR / 'combination_fixed_comparison.json')
    calibration = evaluation['calibration']; satellite = read_json(EXPERIMENT_DIR / 'satellite_quality_2024.json')
    original = dict(baseline['validation']['model']); original['false_warning_pixel_rate'] = float(pixel_false_rate(original))
    rows = [('Original deployed baseline', original), ('Baseline/fixed argmax', weights['baseline']['argmax']),
            ('Inverse weights/fixed argmax', weights['inverse']['argmax']), ('Intermediate weights/fixed argmax', weights['intermediate']['argmax']),
            ('Intermediate + selected boundary', decisions['intermediate']['best']),
            ('Baseline + calendar/held boundary', seasonal['results']['seasonal_root']['baseline_boundary_fixed']),
            ('Combined calendar + weights/held boundary', combination['results']['intermediate_boundary_fixed']),
            ('Final retrained configuration', evaluation['validation']['model'])]
    lines = ['# HAB improvement results', '', evaluation['test_disclosure'], '',
             '**Macro F1 and ELEVATED detection improved, but STRONG detection regressed on 2025. Operational warnings are disabled.** Historical maps, model scores and prototype alert entries remain available through explicit demonstration mode.', '',
             '## 2024 controlled development results', ''] + table(rows)
    lines += ['', 'Intermediate weighting was retained over inverse frequency because its fixed-rule macro F1 was higher with fewer normal-pixel false positives. Decision boundaries were then tested separately on fixed checkpoints. Calendar inputs passed the predeclared controlled criterion: a 0.0242 macro-F1 gain with improved elevated F1. The combined candidate was trained only after measuring these individual effects. Final selection and thresholds were frozen before retraining; retraining used the same selected configuration.', '',
              '## 2025 frozen holdout comparison', '', '| Model | Accuracy | Macro F1 | Elevated precision | Elevated recall | Elevated F1 | Strong precision | Strong recall | Strong F1 | Pixel false-warning rate |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in [('Baseline', evaluation['baseline_test']['model']), ('Final', evaluation['test']['model'])]:
        e, s = r['classes'][1:]
        lines.append(f"| {name} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {e['precision']:.4f} | {e['recall']:.4f} | {e['f1']:.4f} | {s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} | {pixel_false_rate(r):.4f} |")
    lines += ['', 'The selected configuration did not generalize its 2024 strong-class gains: 2025 strong recall fell from 51.70% to 4.13%. This is a material failure, despite the higher macro F1 and recovered elevated class. The frozen configuration was retained for honest reporting; no further tuning used these results.', '', evaluation['scoring_note'], '', 'Per-class details and confusion matrices:', '']
    for name, r in [('Baseline', evaluation['baseline_test']['model']), ('Final', evaluation['test']['model'])]:
        lines += [f'### {name}', '', '| Class | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
        lines += [f"| {c['name']} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['support']} |" for c in r['classes']]
        lines += ['', 'Confusion matrix: rows actual, columns predicted normal/elevated/strong.', '```text'] + [str(row) for row in r['confusion_matrix']] + ['```', '']
    lines += ['## Regional warning evaluation', '', 'Counts below are truth-qualified horizon-one days, not rolling pixel forecasts. Actual fractions use observed proxy labels only; days below 5% target coverage remain UNKNOWN. Predicted fractions use the dense training-supported domain. P75/P90 thresholds remain unchanged from training. Input abstentions on high-risk days remain misses. All-horizon counts and the common available-input cohort are saved separately.', '',
              '| Period/model | Qualified days | Actual risk days | Warning days | Correct warnings | False warnings | Misses | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    regions2024 = read_json(EXPERIMENT_DIR / 'regional_comparison_2024.json')
    regional_rows = [('2024 baseline', regions2024['baseline/original_rule']['daily_horizon_1']), ('2024 final', evaluation['validation']['regional']['daily_horizon_1']),
                     ('2025 baseline', evaluation['baseline_test']['regional']['daily_horizon_1']), ('2025 final', evaluation['test']['regional']['daily_horizon_1'])]
    for name, result in regional_rows:
        r = result['any_warning']
        lines.append(f"| {name} | {r['qualified_days']} | {r['actual_high_risk_days']} | {r['warning_days']} | {r['correctly_warned_high_risk_days']} | {r['false_warning_days']} | {r['missed_high_risk_days']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    lines += ['', '**Strong regional warnings must be assessed separately:**', '', '| Period/model | Actual strong days | Correct strong warnings | Missed strong days | False strong warnings | Strong-warning F1 | Unknown target days | Unavailable input origins |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, result in regional_rows:
        r = result['strong_warning']
        lines.append(f"| {name} | {r['actual_high_risk_days']} | {r['correctly_warned_high_risk_days']} | {r['missed_high_risk_days']} | {r['false_warning_days']} | {r['f1']:.4f} | {result['unknown_truth_cases']} | {result['unavailable_cases']} |")
    lines += ['', '| Period/model | False any-warning frequency | Insufficient-input frequency | Mean input coverage | Mean coverage on warning days | Abstained high-risk days |', '|---|---:|---:|---:|---:|---:|']
    for name, result in regional_rows:
        r = result['any_warning']
        alert_coverage = r['mean_input_coverage_on_warning_days_percent']
        coverage_text = 'N/A (no warnings)' if alert_coverage is None else f'{alert_coverage:.2f}%'
        lines.append(f"| {name} | {100*r['false_warning_frequency']:.2f}% | {100*result['insufficient_data_frequency']:.2f}% | {result['mean_input_coverage_percent']:.2f}% | {coverage_text} | {r['abstained_high_risk_days']} |")
    lines += ['', 'The 2024 reduction from nine false-warning days to zero also reduced correct any-risk warnings from two to one and missed all five strong days. Zero false warnings therefore does not establish a reliable early-warning detector. Sparse and geographically sampled ground truth differs from dense forecast support, and the labels are not verified toxic blooms. The warning-validation gate was disabled before 2025 testing; no holdout result changed that decision.', '',
              '## Final configuration and reproducibility', '', f"- Checkpoint: `outputs/improvement/final/model.keras`; SHA-256 `{training['model_sha256']}`.",
              f"- Original baseline: `models/corrected_convlstm.keras`; SHA-256 `{read_json(BASELINE_DIR / 'freeze.json')['model_sha256']}`. Additional complete snapshot in `outputs/improvement/baseline`.",
              f"- Architecture: ConvLSTM 8/4 with spatial pooling by two and bilinear restoration, {training['parameters']:,} parameters, five 120×84 input frames, five three-class output maps. No filter/capacity expansion; the two calendar inputs add only their required input weights.",
              '- Features: seven environmental fields, seven binary observation flags, then month sine/cosine. Training-only log1p chlorophyll/rainfall, median/IQR normalization, ±8 clipping. Missing input values are zero standardized values with flag zero; missing targets remain 255 and are excluded.',
              f"- Masked loss retained. Normal/elevated/strong class weights: `{training['class_weights']}` (power 0.75, matched expected baseline loss mass).",
              f"- Frozen decision: `argmax(p × {calibration['decision_rule']['scales']})`. Positive score ratios adjust decisions, not raw model probabilities or calibrated confidence.",
              f"- Regional elevated/strong thresholds: `{calibration['regional_elevated_percent']:.8f}% / {calibration['regional_strong_percent']:.8f}%` predicted strong-label fraction; training P75/P90.",
              f"- Fit 2021–2023, development 2024, seed 42, Adam 0.001/clipnorm 1, batch two, CPU threads 4/2, {training['epochs_completed']} epochs, best epoch {training['best_epoch']}. Callbacks, histories, sources and hashes are saved.", '',
              '## Satellite coverage and unavailable predictions', '', f"2024 MODIS marine coverage is {satellite['modis']['marine_coverage_percent']:.2f}%, with {satellite['modis']['days_without_marine_observations']} empty days and a longest empty run of {satellite['modis']['longest_empty_run_days']} days. Sentinel-3 availability is {satellite['sentinel3']['2024_unverified_availability_percent']:.2f}%, but no retrieval quality flags were retained. **Fusion rejected**; no new observations or labels were fabricated.", '',
              'Inference reports cell-day counts, required minima, per-variable coverage/observation dates, freshness and availability. It requires 5% mean marine chlorophyll coverage, at least two observed input days and at least one observed day per environmental variable. Missing required inputs stop model execution; low-coverage historical maps cannot issue reliable warning labels. These are prototype safeguards, not confidence estimates.', '',
              '## Testing, integration and files', '', f"Complete suite: **{count} tests passed**, including the original regression tests, class weighting, cyclical features, input/preprocessing consistency, class boundaries, model reload, coverage, missing-input non-execution, regional aggregation/unknown truth/abstentions, API responses and real five-map integration. The dependency check passed. Real HTTP/browser checks are recorded separately.", '',
              'The existing API/dashboard use the separate final deployment pointer, matching checkpoint/preprocessing/calendar features/frozen decision/coverage policy. Current observations and five forecast dates, class scores, elevated/strong pixel counts, data coverage/freshness and unavailable states are dynamic. No SMS/push or live data ingestion is configured.', '',
              'Main changes: `experiments/`, shared `preprocessing/pipeline.py` and `features.py`, `prediction/decision.py`, `coverage.py`, `regional.py`, `service.py`, model loader, `config.py`, API/dashboard, tests and these documentation snapshots. Working acquisition code, labels, masked loss, two-layer architecture, dashboard layout and original checkpoints are retained.', '',
              'Run/train/test instructions: [commands](improvement_commands.md). Detailed diagnoses/controls: [methodology](improvement_methodology.md). Real dashboard/HTTP verification: [integration checks](improvement_integration_checks.md). Machine-readable results and confusion matrix/history plots: `docs/results/improvement/`.', '',
              '## Remaining limitations', '', 'Weak elevated precision/recall, lower strong recall than baseline, inadequate regional strong-warning skill, sparse sampled satellite truth, unverified Sentinel retrieval quality, proxy rather than field HAB labels, unknown product delivery latency, no live refresh and no external notifications remain. Scores are not calibrated confidence. Independent future-period or external verified observations are required for operational validation.']
    text = '\n'.join(lines) + '\n'
    (config.ROOT / 'docs/improvement_results.md').write_text(text)
    (FINAL_DIR / 'evaluation.md').write_text(text)
    public = config.ROOT / 'docs/results/improvement'; public.mkdir(parents=True, exist_ok=True)
    paths = [EXPERIMENT_DIR / name for name in ['selection.json', 'weight_comparison.json', 'seasonal_comparison.json', 'satellite_quality_2024.json', 'regional_comparison_2024.json', 'regional_common_cohort_2024.json', 'false_alarm_analysis_2024.json', 'combination_fixed_comparison.json']]
    paths += [FINAL_DIR / name for name in ['training_config.json', 'calibration.json', 'evaluation.json', 'freeze.json', 'training_history.csv']]
    paths += [EXPERIMENT_DIR / 'diagnostics_2024/diagnostics.json']
    for path in paths:
        shutil.copy2(path, public / path.name)
    shutil.copy2(EXPERIMENT_DIR / 'final_http_smoke.log', public / 'http_smoke.json')
    from prediction.plots import plot_results
    plot_results()
    for name in ['training_loss.png', 'training_masked_accuracy.png', 'confusion_matrix.png']:
        shutil.copy2(FINAL_DIR / name, public / name)
    print('Published actual final results, snapshots and plots: docs/improvement_results.md', flush=True)

if __name__ == '__main__':
    main()
