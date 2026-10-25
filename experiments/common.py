"""Small experiment utilities; development functions only accept 2024."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import config
from preprocessing.data_utils import file_sha256

EXPERIMENT_DIR = config.ROOT / 'outputs' / 'improvement'
BASELINE_DIR = EXPERIMENT_DIR / 'baseline'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def baseline_pipeline():
    from preprocessing.pipeline import EnvironmentalPipeline
    return EnvironmentalPipeline(label_dir=BASELINE_DIR / 'labels', sequence_dir=BASELINE_DIR / 'sequences')


def load_model(path):
    import tensorflow as tf
    from training import losses
    return tf.keras.models.load_model(path, compile=False)


def verify_baseline():
    manifest = read_json(BASELINE_DIR / 'freeze.json')
    original = Path(manifest['checkpoint_original'])
    if original.exists() and file_sha256(original) != manifest['model_sha256']:
        raise ValueError('The original baseline checkpoint changed; the frozen copy remains available.')
    for name, digest in manifest['files'].items():
        if file_sha256(BASELINE_DIR / name) != digest:
            raise ValueError(f'Frozen baseline changed: {name}')
    return manifest


def collect_predictions(model, pipeline, directory, split='validation', seasonal=False):
    """Cache exact float32 predictions once; no tuning code accepts test data."""
    from training.data_generator import HABDataGenerator
    from preprocessing.features import add_seasonal_features
    if split not in ('validation', 'test'):
        raise ValueError('Only validation/test prediction collection is supported.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generator = HABDataGenerator(split, shuffle=False, pipeline=pipeline)
    shape = (generator.num_samples, 5, config.HEIGHT, config.WIDTH, 3)
    array = np.lib.format.open_memmap(directory / 'probabilities.npy', mode='w+', dtype='float32', shape=shape)
    for i in range(len(generator)):
        X, _ = generator[i]
        row = generator.indices.iloc[i]
        if seasonal and not pipeline.stats.get('seasonal_features', False):
            dates = pipeline.dates[int(row.input_start_idx):int(row.input_end_idx) + 1]
            X = add_seasonal_features(X, dates)
        p = model.predict_on_batch(X)
        if not np.isfinite(p).all() or not np.allclose(p.sum(-1), 1., atol=1e-5):
            raise ValueError('Invalid model scores.')
        array[i] = p[0]
    array.flush()
    generator.indices.to_csv(directory / 'index.csv', index=False)
    write_json(directory / 'cache.json', {'split': split, 'period': config.SPLITS[split],
               'seasonal': seasonal, 'created_at_utc': utc_now(), 'shape': list(shape),
               'label_metadata_sha256': file_sha256(pipeline.label_dir / 'label_metadata.json'),
               'probabilities_sha256': file_sha256(directory / 'probabilities.npy')})
    return array, generator.indices


def cached_samples(directory, pipeline, development=True):
    import pandas as pd
    directory = Path(directory)
    metadata = read_json(directory / 'cache.json')
    if development and metadata['split'] != 'validation':
        raise ValueError('Development diagnostics/tuning prohibit 2025 data.')
    if metadata['label_metadata_sha256'] != file_sha256(pipeline.label_dir / 'label_metadata.json'):
        raise ValueError('Prediction cache and labels differ.')
    if metadata["probabilities_sha256"] != file_sha256(directory / "probabilities.npy"):
        raise ValueError("Prediction cache changed.")
    p = np.load(directory / 'probabilities.npy', mmap_mode='r')
    rows = pd.read_csv(directory / 'index.csv')
    for i, row in enumerate(rows.itertuples()):
        target = np.asarray(pipeline.labels[row.target_start_idx:row.target_end_idx + 1])
        yield row, p[i], target
