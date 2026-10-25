"""Read-only scientific baseline verification; never run test-set inference."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import config
from experiments.common import verify_baseline
from experiments.finalize import verify_final
from preprocessing.data_utils import file_sha256
from prediction.metrics import classification_metrics


def audit():
    verify_baseline()
    verify_final()
    upgrade = json.loads((config.ROOT / 'outputs/upgrade/freeze.json').read_text())
    for name, digest in upgrade['files'].items():
        if file_sha256(config.ROOT / name) != digest:
            raise ValueError(f'Post-freeze evidence changed: {name}')
    evaluation_path = config.RUN_DIR / 'evaluation.json'
    result = json.loads(evaluation_path.read_text())
    if file_sha256(evaluation_path) != upgrade['preserved_2025_evaluation_sha256']:
        raise ValueError('Saved 2025 evidence changed.')
    if result['model_sha256'] != file_sha256(config.MODEL_PATH):
        raise ValueError('Saved evaluation and active model differ.')
    reconciled = {}
    for split in ('validation', 'test'):
        saved = result[split]['model']
        calculated = classification_metrics(saved['confusion_matrix'])
        for name in ('accuracy', 'macro_f1', 'valid_pixel_forecasts'):
            if not np.isclose(saved[name], calculated[name], rtol=1e-10):
                raise ValueError(f'{split} {name} disagrees with confusion counts.')
        for recorded, computed in zip(saved['classes'], calculated['classes'], strict=True):
            for name in ('precision', 'recall', 'f1', 'support'):
                if not np.isclose(recorded[name], computed[name], rtol=1e-10):
                    raise ValueError(f'{split} {recorded["name"]} {name} disagrees with counts.')
        cm = np.asarray(saved['confusion_matrix'])
        for name, computed in [('false_warning_pixel_rate', cm[0, 1:].sum()/max(cm[0].sum(), 1)),
                               ('missed_risk_pixel_rate', cm[1:, 0].sum()/max(cm[1:].sum(), 1))]:
            if name in saved and not np.isclose(saved[name], computed, rtol=1e-10):
                raise ValueError(f'{split} {name} disagrees with its denominator.')
            calculated[name] = float(computed)
        reconciled[split] = calculated
    index = pd.read_csv(config.SEQUENCE_DIR / 'sequence_index.csv')
    dates = np.loadtxt(config.PROCESSED_DIR / 'dates_2021_2025.txt', dtype='datetime64[D]')
    for row in index.itertuples():
        first, last = config.SPLITS[row.split]
        indices = (row.input_start_idx, row.input_end_idx, row.target_start_idx, row.target_end_idx)
        if not (0 <= indices[0] <= indices[1] < indices[2] <= indices[3] < len(dates)):
            raise ValueError('Sequence indices are invalid or input overlaps its target.')
        if (indices[1] - indices[0] != 4 or indices[2] != indices[1] + 1
                or indices[3] - indices[2] != 4
                or dates[indices[0]] < np.datetime64(first) or dates[indices[3]] > np.datetime64(last)):
            raise ValueError('Sequence crosses split boundaries or has the wrong horizon.')
    return {'status': 'verified', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'model_sha256': result['model_sha256'], 'evaluation_sha256': file_sha256(evaluation_path),
            'preprocessing_sha256': file_sha256(config.SEQUENCE_DIR / 'preprocessing.json'),
            'checks': {'baseline_freeze': True, 'final_freeze': True, 'post_freeze_evidence': True,
                       'confusion_metrics_reconciled': True, 'split_safe_sequences': True},
            'sequence_counts': {str(k): int(v) for k, v in index.split.value_counts().items()},
            'saved_metrics': reconciled, 'new_test_inference': False,
            'label_semantics': 'Training-defined chlorophyll anomaly proxies, not confirmed HAB events.',
            'test_disclosure': result['test_disclosure']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new report path.')
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
