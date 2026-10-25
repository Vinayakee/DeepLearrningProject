"""Controlled full-training experiments without touching active artifacts."""
import argparse
import time
import numpy as np
import config
import tensorflow as tf
from models.convlstm import build_convlstm
from preprocessing.features import add_seasonal_features
from training.data_generator import HABDataGenerator
from training.losses import MaskedWeightedLoss, MaskedAccuracy
from preprocessing.data_utils import file_sha256
from experiments.common import BASELINE_DIR, EXPERIMENT_DIR, read_json, write_json, utc_now, verify_baseline, baseline_pipeline, load_model, collect_predictions


def class_weights(counts, power=.5):
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (3,) or np.any(counts <= 0) or not np.isfinite(counts).all():
        raise ValueError('Three positive training class counts are required.')
    if power not in (.5, .75, 1.):
        raise ValueError('Only the predeclared weighting powers are supported.')
    frequency = counts / counts.sum()
    baseline = np.sqrt(counts.max() / counts)
    weights = (counts.max() / counts) ** power
    # Match expected weighted loss mass to baseline, avoiding a global LR-scale confound.
    return (weights * (frequency @ baseline) / (frequency @ weights)).tolist()


class FeatureGenerator(HABDataGenerator):
    def __init__(self, *args, seasonal=False, **kwargs):
        self.seasonal = seasonal
        super().__init__(*args, **kwargs)

    def _load_sample(self, row):
        X, y = super()._load_sample(row)
        if self.seasonal and not self.pipeline.stats.get('seasonal_features', False):
            first, last = int(row['input_start_idx']), int(row['input_end_idx'])
            X = add_seasonal_features(X, self.pipeline.dates[first:last + 1])
        return X, y


def run(name, power=.5, seasonal=False, epochs=30, smoke=False, pipeline=None):
    verify_baseline()
    output = EXPERIMENT_DIR / ('smoke_' + name if smoke else name)
    if output.exists():
        raise ValueError(f'Experiment exists and will not be overwritten: {output}')
    pipeline = baseline_pipeline() if pipeline is None else pipeline
    train = FeatureGenerator('train', batch_size=2, shuffle=True, pipeline=pipeline, seasonal=seasonal, limit=12 if smoke else None)
    validation = FeatureGenerator('validation', batch_size=2, shuffle=False, pipeline=pipeline, seasonal=seasonal, limit=8 if smoke else None)
    counts = read_json(BASELINE_DIR / 'sequences/manifest.json')['splits']['train']['class_counts']
    weights = class_weights(counts, power)
    channels = 16 if seasonal else 14
    tf.keras.utils.set_random_seed(42)
    model = build_convlstm(channels=channels)
    model.compile(optimizer=tf.keras.optimizers.Adam(.001, clipnorm=1.), loss=MaskedWeightedLoss(weights), metrics=[MaskedAccuracy()])
    record = {'name': name, 'created_at_utc': utc_now(), 'smoke': smoke, 'weight_power': power,
              'class_weights': weights, 'normalization': 'expected training-pixel weight equals frozen square-root baseline',
              'seasonal_features': seasonal, 'channels': channels, 'filters': [8, 4], 'spatial_downsample': 2,
              'seed': 42, 'learning_rate': .001, 'clipnorm': 1., 'batch_size': 2, 'epochs_requested': epochs,
              'train_sequences': train.num_samples, 'validation_sequences': validation.num_samples,
              'fitting_period': ['2021-01-01', '2023-12-31'], 'development_period': ['2024-01-01', '2024-12-31'],
              'baseline_model_sha256': read_json(BASELINE_DIR / 'freeze.json')['model_sha256'],
              'label_metadata_sha256': file_sha256(pipeline.label_dir / 'label_metadata.json'),
              'preprocessing_sha256': file_sha256(pipeline.sequence_dir / 'preprocessing.json'),
              'input_shape': [5, 120, 84, channels], 'output_shape': [5, 120, 84, 3],
              'tensorflow_version': tf.__version__, 'parameters': model.count_params(),
              'callbacks': {'early_stopping_patience': 4, 'reduce_lr_patience': 2, 'reduce_lr_factor': .5, 'minimum_lr': 1e-5}}
    write_json(output / 'training_config.json', record)
    started = time.monotonic()
    print(f'Training {name}: weights={weights}; seasonal={seasonal}; full={not smoke}', flush=True)
    history = model.fit(train, validation_data=validation, epochs=epochs, verbose=2, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(output / 'model.keras'), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=.5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(output / 'training_history.csv')), tf.keras.callbacks.TerminateOnNaN()])
    record.update({'epochs_completed': len(history.history['loss']), 'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
                   'duration_seconds': time.monotonic() - started, 'model_sha256': file_sha256(output / 'model.keras')})
    write_json(output / 'training_config.json', record)
    restored = load_model(output / 'model.keras')
    X, y = validation[0]
    p = restored.predict_on_batch(X)
    assert p.shape == y.shape and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(-1), 1., atol=1e-6)
    if not smoke:
        collect_predictions(restored, pipeline, output / 'validation', seasonal=seasonal)
    print(f'{name} completed; checkpoint reload and real inference passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--weight-power', type=float, choices=[.5, .75, 1.], default=.5)
    parser.add_argument('--seasonal', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.name.replace('_', '').isalnum() or args.epochs < 1:
        parser.error('Use an alphanumeric experiment name and positive epochs.')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    run(args.name, args.weight_power, args.seasonal, args.epochs, args.smoke)


if __name__ == '__main__':
    main()
