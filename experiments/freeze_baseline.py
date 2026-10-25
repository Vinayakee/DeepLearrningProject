"""Freeze the actual current checkpoint, fitted artifacts and existing results."""
from pathlib import Path
import shutil
import config
from preprocessing.data_utils import file_sha256, environment_fingerprint
from prediction.evaluation import load_trained_model
from experiments.common import BASELINE_DIR, read_json, write_json, utc_now, verify_baseline


def main():
    if (BASELINE_DIR / 'freeze.json').exists():
        verify_baseline()
        print('Existing immutable baseline verified; not overwritten.')
        return
    if BASELINE_DIR.exists():
        raise ValueError('An incomplete baseline directory exists; inspect it before retrying.')
    model, training = load_trained_model()
    results = read_json(config.RUN_DIR / 'evaluation.json')
    if results['model_sha256'] != training['model_sha256']:
        raise ValueError('Current model and evaluation differ.')
    BASELINE_DIR.mkdir(parents=True)
    shutil.copy2(config.MODEL_PATH, BASELINE_DIR / 'model.keras')
    shutil.copytree(config.LABEL_DIR, BASELINE_DIR / 'labels')
    shutil.copytree(config.SEQUENCE_DIR, BASELINE_DIR / 'sequences')
    for name in ('training_config.json', 'training_history.csv', 'calibration.json', 'evaluation.json', 'evaluation.md'):
        shutil.copy2(config.RUN_DIR / name, BASELINE_DIR / name)
    source_names = ['config.py', 'models/convlstm.py', 'training/train.py', 'training/losses.py',
                    'training/data_generator.py', 'preprocessing/pipeline.py', 'preprocessing/data_utils.py',
                    'preprocessing/create_chl_climatology.py', 'preprocessing/create_final_hab_labels.py',
                    'prediction/metrics.py', 'prediction/evaluation.py', 'prediction/service.py']
    for name in source_names:
        target = BASELINE_DIR / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.ROOT / name, target)
    write_json(BASELINE_DIR / 'architecture.json', {'input_shape': model.input_shape, 'output_shape': model.output_shape,
               'parameters': model.count_params(), 'keras_config': model.get_config()})
    freeze = {'created_at_utc': utc_now(), 'checkpoint_original': str(config.MODEL_PATH),
              'model_sha256': training['model_sha256'], 'decision_rule': {'type': 'strong_override',
              'strong_threshold': results['calibration']['strong_probability_threshold']},
              'environment_fingerprint': environment_fingerprint(config.PROCESSED_DIR),
              'test_disclosure': '2025 metrics were evaluated and inspected previously. New development will use 2024 only; final 2025 evaluation cannot be called untouched.',
              'files': {str(p.relative_to(BASELINE_DIR)): file_sha256(p) for p in sorted(BASELINE_DIR.rglob('*')) if p.is_file()}}
    write_json(BASELINE_DIR / 'freeze.json', freeze)
    verify_baseline()
    report = ['# Frozen HAB baseline', '', f"Checkpoint SHA-256: `{training['model_sha256']}`", '',
              '8,211 parameters; 14 inputs; ConvLSTM8/4; masked weighted cross-entropy; square-root class weights.', '',
              f"Decision rule: strong score >= {freeze['decision_rule']['strong_threshold']}; otherwise argmax normal/elevated.", '',
              '| Period | Macro F1 | Elevated F1 | Strong precision | Strong recall | Strong F1 |', '|---|---:|---:|---:|---:|---:|']
    for period in ('validation', 'test'):
        r = results[period]['model']; c = r['classes'][2]
        report.append(f"| {period} | {r['macro_f1']:.4f} | {r['classes'][1]['f1']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} |")
    report += ['', freeze['test_disclosure'], '', 'This folder includes a checkpoint, fitted label arrays, preprocessing, original source and checksum manifest. Large aligned environmental fields remain shared and must retain their recorded identity. Baseline source/checkpoint are never overwritten by experiment commands.']
    (BASELINE_DIR / 'report.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
