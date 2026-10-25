"""Train a separate HAB model using real verified targets and fresh output paths."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import shutil
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from preprocessing.hab_sequences import load_dataset, write_json
from preprocessing.data_utils import file_sha256
from models.hab_convlstm import build_hab_convlstm, MaskedBinaryLoss


def baseline_features(X, zones):
    return np.stack([np.concatenate([X[:, mask, :].mean((0, 1)), np.eye(len(zones))[i]])
                     for i, mask in enumerate(zones)])


class HABGenerator(tf.keras.utils.Sequence):
    def __init__(self, pipeline, labels, events, rows, batch_size=2, balanced=False, **kwargs):
        super().__init__(**kwargs)
        if not rows or batch_size < 1:
            raise ValueError('Nonempty windows and positive batch size required.')
        self.pipeline, self.labels, self.rows = pipeline, labels, rows
        self.batch_size, self.balanced = batch_size, balanced
        self.rng = np.random.default_rng(42)
        groups = []
        for row in rows:
            start = row['target_start']
            groups.append(set(events[start:start+5].ravel())-{''})
        counts = {event: sum(event in group for group in groups) for event in set().union(*groups)}
        # Each event has similar sampling mass despite long/overlapping reports.
        weights = np.array([sum(1/counts[e] for e in g)/len(g) if g else 1/max(1, sum(not x for x in groups))
                            for g in groups])
        self.probabilities = weights/weights.sum()
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.rows)/self.batch_size))

    def on_epoch_end(self):
        self.order = (self.rng.choice(len(self.rows), len(self.rows), replace=True, p=self.probabilities)
                      if self.balanced else np.arange(len(self.rows)))

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        samples = []
        for position in self.order[index*self.batch_size:(index+1)*self.batch_size]:
            row = self.rows[position]
            first, target = row['input_start'], row['target_start']
            X = self.pipeline.transform(np.arange(first, target))
            samples.append((X, self.labels[target:target+5].astype(np.float32)))
        return tuple(np.stack(part) for part in zip(*samples))


def fit_baselines(pipeline, labels, rows):
    features, targets = [], []
    for row in rows:
        features.append(baseline_features(pipeline.transform(np.arange(row['input_start'], row['target_start'])), pipeline.zones))
        targets.append(labels[row['target_start']:row['target_start']+5])
    X, y = np.asarray(features), np.asarray(targets)
    models = []
    for horizon in range(5):
        truth = y[:, horizon].ravel()
        valid = truth != 255
        target, x = truth[valid], X.reshape(-1, X.shape[-1])[valid]
        if not len(target):
            raise ValueError('Each training horizon needs verified targets.')
        if len(np.unique(target)) < 2:
            models.append({'constant': float(target.mean())})
        else:
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42).fit(x, target)
            models.append({'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0])})
    return {'horizons': models, 'training_prevalence': float(y[y != 255].mean())}


def train(dataset, output, epochs=30, batch_size=2, balanced=False):
    if epochs < 1 or batch_size < 1:
        raise ValueError('Positive epochs and batch size required.')
    output, dataset = Path(output), Path(dataset).resolve()
    if output.exists():
        raise ValueError('Use a fresh HAB run directory; existing runs cannot be overwritten.')
    pipeline, labels, events, rows, manifest = load_dataset(dataset)
    for split in ('train', 'validation', 'calibration', 'test'):
        selected = [r for r in rows if r['split'] == split]
        targets = np.concatenate([labels[r['target_start']:r['target_start']+5].ravel() for r in selected]) if selected else []
        if not all(np.any(np.asarray(targets) == c) for c in (0, 1)):
            raise ValueError(f'{split} needs usable windows with verified positives AND negatives. See the label audit.')
    tf.keras.utils.set_random_seed(42)
    train_rows = [r for r in rows if r['split'] == 'train']
    train_targets = np.concatenate([labels[r['target_start']:r['target_start']+5] for r in train_rows])
    for j, zone in enumerate(pipeline.spec['zones']):
        if not all(np.any(train_targets[:, j] == c) for c in (0, 1)):
            raise ValueError(f'Zone {zone["id"]} needs both verified classes in usable training targets.')
    train_generator = HABGenerator(pipeline, labels, events, train_rows, batch_size, balanced)
    validation = HABGenerator(pipeline, labels, events, [r for r in rows if r['split'] == 'validation'], batch_size)
    truth = np.concatenate([labels[r['target_start']:r['target_start']+5].ravel() for r in train_rows])
    # Weighting and event-balanced sampling are alternatives, avoiding double correction.
    weight = 1. if balanced else float(np.sqrt((truth == 0).sum()/(truth == 1).sum()))
    model = build_hab_convlstm((pipeline.spec['input_days'], *pipeline.shape[1:], 16), pipeline.zones)
    model.compile(optimizer=tf.keras.optimizers.Adam(.001, clipnorm=1.), loss=MaskedBinaryLoss(weight))
    baseline = fit_baselines(pipeline, labels, train_rows)
    output.mkdir(parents=True)
    write_json(output/'baselines.json', baseline)
    record = {'schema': 'hab-training-v1', 'state': 'training', 'dataset_dir': str(dataset), 'seed': 42,
              'dataset_manifest_sha256': file_sha256(dataset/'manifest.json'), 'positive_weight': weight,
              'event_balanced_sampling': balanced, 'target': pipeline.spec['event_definition'],
              'input_shape': list(model.input_shape[1:]), 'output_shape': list(model.output_shape[1:]),
              'tensorflow_version': tf.__version__, 'parameter_count': model.count_params(),
              'dependencies': {package: version(package) for package in
                               ('tensorflow', 'keras', 'numpy', 'scipy', 'scikit-learn', 'pandas')},
              'operational_warning_validated': False, 'input_timing': manifest['input_timing'],
              'label_audit': manifest['audit']}
    source_root = Path(__file__).resolve().parents[1]
    source_files = ('config.py', 'models/hab_convlstm.py', 'training/train_hab.py',
                    'preprocessing/hab_labels.py', 'preprocessing/hab_sequences.py',
                    'preprocessing/data_utils.py', 'preprocessing/pipeline.py', 'preprocessing/features.py',
                    'prediction/coverage.py', 'prediction/hab_evaluation.py', 'prediction/hab_service.py',
                    'requirements-lock.txt')
    record['source_sha256'] = {}
    for name in source_files:
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root/name, destination)
        record['source_sha256'][name] = file_sha256(destination)
    write_json(output/'training.json', record)
    history = model.fit(train_generator, validation_data=validation, epochs=epochs, verbose=2, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(output/'model.keras'), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=.5),
        tf.keras.callbacks.CSVLogger(str(output/'history.csv')), tf.keras.callbacks.TerminateOnNaN()])
    if not all(np.isfinite(history.history[key]).all() for key in ('loss', 'val_loss')):
        raise ValueError('Nonfinite training history; this run cannot serve forecasts.')
    restored = tf.keras.models.load_model(output/'model.keras', compile=False)
    X, _ = validation[0]
    p = restored(X, training=False).numpy()
    if p.shape != (len(X), 5, len(pipeline.zones)) or not np.isfinite(p).all():
        raise ValueError('Saved HAB checkpoint failed reload/inference verification.')
    record.update(state='trained', epochs_completed=len(history.history['loss']),
                  best_epoch=int(np.argmin(history.history['val_loss']))+1,
                  model_sha256=file_sha256(output/'model.keras'), baselines_sha256=file_sha256(output/'baselines.json'))
    write_json(output/'training.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--event-balanced', action='store_true')
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    print(json.dumps(train(args.dataset_dir, args.output_dir, args.epochs, args.batch_size, args.event_balanced), indent=2))


if __name__ == '__main__':
    main()
