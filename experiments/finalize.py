"""Retrain, freeze, evaluate once, then activate the selected configuration."""
import argparse
import shutil
import os
import json
import numpy as np
from pathlib import Path
import config
from preprocessing.data_utils import file_sha256, environment_fingerprint
from preprocessing.pipeline import EnvironmentalPipeline
from prediction.metrics import classification_metrics, confusion_matrix
from experiments.common import EXPERIMENT_DIR, BASELINE_DIR, read_json, write_json, utc_now, verify_baseline, baseline_pipeline, load_model, collect_predictions, cached_samples
from experiments.development import metrics, valid_scores
from experiments.regional import evaluate_cache, OLD_POLICY, summarize_rows

FINAL_DIR = EXPERIMENT_DIR / 'final'
FINAL_PREPROCESSING = EXPERIMENT_DIR / 'final_preprocessing'


def prepare_pipeline(selection):
    if not FINAL_PREPROCESSING.exists():
        shutil.copytree(BASELINE_DIR / 'sequences', FINAL_PREPROCESSING)
        stats = read_json(FINAL_PREPROCESSING / 'preprocessing.json')
        stats['seasonal_features'] = selection['seasonal_features']
        stats['feature_encoding'] = 'month sin/cos(2*pi*month/12), appended after seven availability masks' if selection['seasonal_features'] else 'seven standardized fields followed by seven availability masks'
        write_json(FINAL_PREPROCESSING / 'preprocessing.json', stats)
    return EnvironmentalPipeline(label_dir=BASELINE_DIR / 'labels', sequence_dir=FINAL_PREPROCESSING)


def train_final():
    import tensorflow as tf
    from experiments.train import run
    selected = read_json(EXPERIMENT_DIR / 'selection.json')['selected_configuration']
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    pipeline = prepare_pipeline(selected)
    run('final', selected['weight_power'], selected['seasonal_features'], selected['epochs_requested'], pipeline=pipeline)
    # Boundary settings were frozen before retraining; never retune them here.
    training = read_json(FINAL_DIR / 'training_config.json')
    calibration = {**selected['regional_parameters'], 'decision_rule': selected['decision_rule'],
                   'strong_probability_threshold': selected['decision_rule'].get('strong_threshold'),
                   'threshold_fit_split': '2024 development only', 'threshold_selection': read_json(EXPERIMENT_DIR / 'selection.json')['criterion']['description'],
                   'coverage_policy': selected['coverage_policy'], 'model_sha256': training['model_sha256'],
                   'preprocessing_sha256': training['preprocessing_sha256'], 'created_at_utc': utc_now()}
    write_json(FINAL_DIR / 'calibration.json', calibration)
    y, p = valid_scores('final', pipeline)
    validation = metrics(y, p, selected['decision_rule'])
    write_json(FINAL_DIR / 'validation_metrics.json', validation)
    regional = evaluate_cache(FINAL_DIR / 'validation', pipeline, selected['decision_rule'], calibration, selected['coverage_policy'], FINAL_DIR / 'regional_2024')
    print(f"Final retraining complete: 2024 macro F1={validation['macro_f1']:.4f}; elevated F1={validation['classes'][1]['f1']:.4f}. Rule unchanged.", flush=True)


def freeze_final():
    if (FINAL_DIR / 'freeze.json').exists():
        raise ValueError('Final freeze exists and cannot be overwritten.')
    selected_path = EXPERIMENT_DIR / 'selection.json'
    selected = read_json(selected_path)['selected_configuration']
    record = read_json(FINAL_DIR / 'training_config.json')
    calibration = read_json(FINAL_DIR / 'calibration.json')
    regional = read_json(FINAL_DIR / 'regional_2024/regional_metrics.json')['daily_horizon_1']
    calibration['operational_warning_validated'] = False
    calibration['regional_validation_status'] = 'Research prototype: regional skill is insufficient and labels are chlorophyll proxies. Independent verified HAB validation is required before operational warning delivery.'
    calibration['regional_validation_2024'] = {'any_warning': regional['any_warning'], 'strong_warning': regional['strong_warning']}
    write_json(FINAL_DIR / 'calibration.json', calibration)
    if record.get('smoke') or not record.get('epochs_completed') or record['weight_power'] != selected['weight_power'] or record['seasonal_features'] != selected['seasonal_features'] or calibration['decision_rule'] != selected['decision_rule']:
        raise ValueError('Final configuration does not match the frozen selection.')
    for key in ['filters', 'spatial_downsample', 'seed', 'epochs_requested', 'batch_size', 'learning_rate', 'clipnorm', 'channels']:
        if record.get(key) != selected[key]:
            raise ValueError(f'Final training setting differs from frozen selection: {key}')
    np.testing.assert_allclose(record['class_weights'], selected['class_weights'], rtol=1e-7)
    path = FINAL_DIR / 'freeze.json'
    if path.exists():
        raise ValueError('Final freeze exists and cannot be overwritten.')
    sources = ['models/convlstm.py', 'training/losses.py', 'training/data_generator.py', 'preprocessing/pipeline.py', 'preprocessing/features.py', 'prediction/decision.py', 'prediction/coverage.py', 'prediction/regional.py', 'experiments/train.py', 'experiments/select.py', 'experiments/finalize.py', 'experiments/common.py', 'experiments/development.py', 'experiments/regional.py', 'prediction/metrics.py', 'preprocessing/data_utils.py', 'config.py']
    files = [FINAL_DIR / name for name in ['model.keras', 'training_config.json', 'calibration.json', 'training_history.csv', 'validation_metrics.json']]
    files += [FINAL_PREPROCESSING / name for name in ['preprocessing.json', 'sequence_index.csv', 'manifest.json']]
    files += [BASELINE_DIR / 'freeze.json']
    files += [BASELINE_DIR / 'labels' / name for name in ['hab_labels_2021_2025.npy', 'stable_ocean_mask.npy', 'monthly_chl_climatology_2021_2025.npy']]
    files += [selected_path, BASELINE_DIR / 'labels/label_metadata.json', BASELINE_DIR / 'labels/climatology_metadata.json']
    files += [config.ROOT / name for name in sources]
    freeze = {'frozen_at_utc': utc_now(), 'configuration': selected,
              'files': {str(p.relative_to(config.ROOT)): file_sha256(p) for p in files},
              'environment_fingerprint': environment_fingerprint(config.PROCESSED_DIR),
              'test_disclosure': read_json(selected_path)['test_disclosure']}
    write_json(path, freeze)
    verify_final()
    print('Final checkpoint, preprocessing, decision rule, coverage policy and fitting sources frozen before 2025 inference.', flush=True)


def verify_final():
    freeze = read_json(FINAL_DIR / 'freeze.json')
    for name, digest in freeze['files'].items():
        if file_sha256(config.ROOT / name) != digest:
            raise ValueError(f'Frozen final artifact changed: {name}')
    if freeze['environment_fingerprint'] != environment_fingerprint(config.PROCESSED_DIR):
        raise ValueError('Environmental data changed after the final freeze.')
    return freeze


def evaluate_pixels(directory, pipeline, rule):
    from prediction.decision import apply_rule
    cm = np.zeros((3, 3), np.int64); horizons = np.zeros((5, 3, 3), np.int64)
    brier, count = 0., 0
    for row, p, target in cached_samples(directory, pipeline, development=False):
        valid = target != 255; predicted = apply_rule(p, rule)
        cm += confusion_matrix(target[valid], predicted[valid])
        brier += float(np.square(p[..., 2][valid] - (target[valid] == 2)).sum()); count += int(valid.sum())
        for h in range(5):
            v = valid[h]
            horizons[h] += confusion_matrix(target[h][v], predicted[h][v])
    result = classification_metrics(cm)
    result.update({'decision_rule': rule, 'horizons': [classification_metrics(m) for m in horizons], 'strong_probability_brier_score': brier / count,
                   'false_warning_pixel_rate': float(cm[0, 1:].sum() / cm[0].sum()), 'missed_risk_pixel_rate': float(cm[1:, 0].sum() / cm[1:].sum())})
    return result


def claim_test_evaluation(directory, freeze_sha256):
    directory = Path(directory)
    if (directory / 'evaluation.json').exists():
        raise ValueError('Final test evaluation already completed; do not repeat or retune.')
    try:
        with (directory / 'test_started.json').open('x') as stream:
            json.dump({'started_at_utc': utc_now(), 'freeze_sha256': freeze_sha256}, stream, indent=2)
    except FileExistsError as error:
        raise ValueError('Final test evaluation already started; do not repeat or retune.') from error


def test_once():
    import tensorflow as tf
    import pandas as pd
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    freeze = verify_final(); verify_baseline()
    claim_test_evaluation(FINAL_DIR, file_sha256(FINAL_DIR / 'freeze.json'))
    selected = freeze['configuration']; pipeline = prepare_pipeline(selected)
    calibration = read_json(FINAL_DIR / 'calibration.json')
    model = load_model(FINAL_DIR / 'model.keras')
    collect_predictions(model, pipeline, FINAL_DIR / 'test', split='test', seasonal=selected['seasonal_features'])
    pixel = evaluate_pixels(FINAL_DIR / 'test', pipeline, selected['decision_rule'])
    regional = evaluate_cache(FINAL_DIR / 'test', pipeline, selected['decision_rule'], calibration, selected['coverage_policy'], FINAL_DIR / 'regional_2025', development=False)
    # Frozen baseline inference at this final stage supplies the previously missing regional comparison.
    original_pipeline = baseline_pipeline(); baseline_cal = read_json(BASELINE_DIR / 'calibration.json')
    baseline_rule = read_json(BASELINE_DIR / 'freeze.json')['decision_rule']
    baseline_model = load_model(BASELINE_DIR / 'model.keras')
    collect_predictions(baseline_model, original_pipeline, FINAL_DIR / 'baseline_test', split='test')
    baseline_region = evaluate_cache(FINAL_DIR / 'baseline_test', original_pipeline, baseline_rule, baseline_cal, OLD_POLICY, FINAL_DIR / 'baseline_regional_2025', development=False)
    baseline_replay = evaluate_pixels(FINAL_DIR / 'baseline_test', original_pipeline, baseline_rule)
    baseline_saved = read_json(BASELINE_DIR / 'evaluation.json')['test']['model']
    if baseline_replay['confusion_matrix'] != baseline_saved['confusion_matrix']:
        raise ValueError('Frozen baseline replay differs from the recorded baseline: investigate reproducibility before reporting.')
    old = pd.read_csv(FINAL_DIR / 'baseline_regional_2025/daily_cases.csv'); new = pd.read_csv(FINAL_DIR / 'regional_2025/daily_cases.csv')
    if not old[['input_end_date', 'target_date', 'horizon']].equals(new[['input_end_date', 'target_date', 'horizon']]):
        raise ValueError('Final/baseline regional dates differ.')
    common = new.available & old.available & new.truth_qualified & old.truth_qualified
    comparison = {'baseline': summarize_rows(old[common & (old.horizon == 1)]), 'final': summarize_rows(new[common & (new.horizon == 1)])}
    validation = read_json(FINAL_DIR / 'validation_metrics.json')
    results = {'created_at_utc': utc_now(), 'model_sha256': read_json(FINAL_DIR / 'training_config.json')['model_sha256'],
               'final_freeze_sha256': file_sha256(FINAL_DIR / 'freeze.json'), 'calibration': calibration,
               'validation': {'model': validation, 'regional': read_json(FINAL_DIR / 'regional_2024/regional_metrics.json')},
               'test': {'model': pixel, 'regional': regional},
               'baseline_test': {'model': baseline_saved, 'regional': baseline_region},
               'common_available_input_cohort_2025': comparison, 'test_disclosure': freeze['test_disclosure'],
               'scoring_note': 'Pixel metrics use exactly the same valid corrected target labels for baseline and final, including low-input cases. Rolling horizons repeat dates at distinct origins. Regional metrics separately count coverage abstentions as misses on observed high-risk days. No parameters changed after the final freeze.'}
    write_json(FINAL_DIR / 'evaluation.json', results)
    print(f"Frozen 2025 result: macro F1={pixel['macro_f1']:.4f}, elevated F1={pixel['classes'][1]['f1']:.4f}, strong F1={pixel['classes'][2]['f1']:.4f}.", flush=True)
    print(freeze['test_disclosure'], flush=True)


def activate():
    verify_final(); verify_baseline()
    if not (FINAL_DIR / 'evaluation.json').exists():
        raise ValueError('Complete the frozen final evaluation before activation.')
    deployment = {'schema_version': 1, 'activated_at_utc': utc_now(),
                  'model_path': str((FINAL_DIR / 'model.keras').relative_to(config.ROOT)),
                  'run_dir': str(FINAL_DIR.relative_to(config.ROOT)),
                  'sequence_dir': str(FINAL_PREPROCESSING.relative_to(config.ROOT)),
                  'label_dir': str((BASELINE_DIR / 'labels').relative_to(config.ROOT)),
                  'freeze_sha256': file_sha256(FINAL_DIR / 'freeze.json')}
    tmp = FINAL_DIR / 'deployment.tmp'
    write_json(tmp, deployment); os.replace(tmp, FINAL_DIR / 'deployment.json')
    print('Final model activated through a separate deployment pointer. Restart the API; all baseline artifacts remain intact.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['train', 'freeze', 'test', 'activate', 'verify'])
    args = parser.parse_args()
    {'train': train_final, 'freeze': freeze_final, 'test': test_once, 'activate': activate, 'verify': verify_final}[args.phase]()

if __name__ == '__main__':
    main()
