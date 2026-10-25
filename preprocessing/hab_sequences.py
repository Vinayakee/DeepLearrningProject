"""Independent environmental fitting and manifests for verified HAB targets."""
import argparse
import json
from pathlib import Path
import numpy as np
import config
from preprocessing.data_utils import (load_dates, load_grid, load_pair, derive_ocean_support,
                                      environment_fingerprint, file_sha256)
from preprocessing.pipeline import value_transform
from preprocessing.features import add_seasonal_features
from preprocessing.hab_labels import validate_spec, build_labels, sequence_rows, SPLITS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class HABInputs:
    """No dependency on chlorophyll-derived label files or their fitted statistics."""
    def __init__(self, processed_dir, spec, stats=None, ocean=None):
        self.processed_dir, self.spec = Path(processed_dir), validate_spec(spec)
        self.dates = load_dates(self.processed_dir)
        self.latitude, self.longitude, self.shape = load_grid(self.processed_dir, self.dates)
        self.values, self.masks = {}, {}
        for name in config.CHANNEL_NAMES:
            self.values[name], self.masks[name] = load_pair(self.processed_dir, name, self.shape)
        first, last = map(np.datetime64, spec['splits']['train'])
        selected = (self.dates >= first) & (self.dates <= last)
        if not selected.any():
            raise ValueError('No environmental inputs in the training period.')
        self.ocean = (derive_ocean_support(self.processed_dir, self.shape, selected)
                      if ocean is None else np.asarray(ocean, bool))
        if self.ocean.shape != self.shape[1:] or not self.ocean.any():
            raise ValueError('Invalid training marine support.')
        self.zones = []
        occupied = np.zeros_like(self.ocean)
        for zone in spec['zones']:
            west, south, east, north = zone['bounds']
            mask = (self.ocean & (self.latitude[:, None] >= south) & (self.latitude[:, None] <= north)
                    & (self.longitude[None] >= west) & (self.longitude[None] <= east))
            if not mask.any() or np.any(mask & occupied):
                raise ValueError('Zones must contain supported marine cells and must not overlap on the grid.')
            occupied |= mask
            self.zones.append(mask)
        self.support = occupied
        self.stats = stats

    def valid(self, name, indices):
        values = np.asarray(self.values[name][indices])
        valid = (self.masks[name][indices] > 0) & np.isfinite(values) & self.support
        if name == 'chlorophyll':
            valid &= values > 0
        if name == 'rainfall':
            valid &= values >= 0
        return values, valid

    def fit(self):
        first, last = map(np.datetime64, self.spec['splits']['train'])
        selected = np.flatnonzero((self.dates >= first) & (self.dates <= last))
        stats = {'channel_order': list(config.CHANNEL_NAMES), 'seasonal_features': True, 'channels': {}}
        for name in config.CHANNEL_NAMES:
            values, valid = self.valid(name, selected)
            observed = value_transform(values[valid], name)
            if not len(observed):
                raise ValueError(f'No training observations for {name}.')
            q25, median, q75 = np.percentile(observed, [25, 50, 75])
            stats['channels'][name] = {'median': float(median), 'scale': float(max(q75-q25, 1e-6))}
        self.stats = stats
        return stats

    def transform(self, indices):
        indices = np.asarray(indices, int)
        if (self.stats is None or len(indices) != self.spec['input_days'] or indices.min() < 0
                or indices.max() >= len(self.dates) or np.any(np.diff(indices) != 1)):
            raise ValueError('Fitted statistics and a complete consecutive input window required.')
        values_out, masks_out = [], []
        for name in config.CHANNEL_NAMES:
            values, valid = self.valid(name, indices)
            stats = self.stats['channels'][name]
            transformed = value_transform(np.where(valid, values, 0), name)
            scaled = np.clip((transformed-stats['median'])/stats['scale'], -8, 8)
            values_out.append(np.where(valid, scaled, 0).astype(np.float32))
            masks_out.append(valid.astype(np.float32))
        return add_seasonal_features(np.stack(values_out+masks_out, -1), self.dates[indices])


def prepare(spec_path, observations, output, processed_dir=config.PROCESSED_DIR):
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh HAB dataset directory; existing artifacts cannot be overwritten.')
    pipeline = HABInputs(processed_dir, spec)
    labels, events, records = build_labels(observations, pipeline.dates, spec)
    rows = sequence_rows(pipeline.dates, labels, spec)
    audit = {}
    for split in SPLITS:
        first, last = map(np.datetime64, spec['splits'][split])
        selected = (pipeline.dates >= first) & (pipeline.dates <= last)
        part = labels[selected]
        audit[split] = {'positive_zone_days': int((part == 1).sum()), 'negative_zone_days': int((part == 0).sum()),
                        'unknown_zone_days': int((part == 255).sum()),
                        'events': len(set(events[selected].ravel())-{''}),
                        'windows': sum(r['split'] == split for r in rows),
                        'by_zone': {zone['id']: {'positive_days': int((part[:, j] == 1).sum()),
                                                'negative_days': int((part[:, j] == 0).sum()),
                                                'unknown_days': int((part[:, j] == 255).sum())}
                                    for j, zone in enumerate(spec['zones'])}}
    stats = pipeline.fit()
    output.mkdir(parents=True)
    write_json(output/'spec.json', spec)
    write_json(output/'records.json', records)
    write_json(output/'preprocessing.json', stats)
    write_json(output/'index.json', rows)
    np.savez_compressed(output/'targets.npz', labels=labels, events=events, ocean=pipeline.ocean)
    files = ('spec.json', 'records.json', 'preprocessing.json', 'index.json', 'targets.npz')
    manifest = {'schema': 'hab-dataset-v1', 'processed_dir': str(Path(processed_dir).resolve()),
                'environment_fingerprint': environment_fingerprint(processed_dir),
                'source_coordinates_sha256': {name: file_sha256(Path(processed_dir)/name) for name in
                    ('dates_2021_2025.txt', 'latitude_kerala_2021_2025.npy', 'longitude_kerala_2021_2025.npy')},
                'observations_sha256': file_sha256(observations), 'audit': audit,
                'files': {name: file_sha256(output/name) for name in files},
                'input_timing': 'Nominal archive dates; historical publication latency not established.'}
    write_json(output/'manifest.json', manifest)
    return manifest


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema') != 'hab-dataset-v1':
        raise ValueError('Unsupported HAB dataset manifest.')
    for name in ('spec.json', 'records.json', 'preprocessing.json', 'index.json', 'targets.npz'):
        if file_sha256(directory/name) != manifest['files'][name]:
            raise ValueError(f'HAB fitting artifact changed: {name}')
    processed = Path(manifest['processed_dir'])
    if environment_fingerprint(processed) != manifest['environment_fingerprint']:
        raise ValueError('Environmental archive changed; prepare a separate dataset/run.')
    for name, digest in manifest['source_coordinates_sha256'].items():
        if file_sha256(processed/name) != digest:
            raise ValueError('Environmental dates or coordinates changed.')
    spec = json.loads((directory/'spec.json').read_text())
    with np.load(directory/'targets.npz', allow_pickle=False) as arrays:
        labels, events, ocean = (arrays[k] for k in ('labels', 'events', 'ocean'))
    pipeline = HABInputs(processed, spec, json.loads((directory/'preprocessing.json').read_text()), ocean)
    if labels.shape != (len(pipeline.dates), len(spec['zones'])) or not np.isin(labels, [0, 1, 255]).all():
        raise ValueError('Invalid HAB targets.')
    index = json.loads((directory/'index.json').read_text())
    if index != sequence_rows(pipeline.dates, labels, spec):
        raise ValueError('HAB sequence index differs from the declared split/target contract.')
    return pipeline, labels, events, index, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--observations', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--processed-dir', default=str(config.PROCESSED_DIR))
    args = parser.parse_args()
    print(json.dumps(prepare(args.spec, args.observations, args.output_dir, args.processed_dir), indent=2))


if __name__ == '__main__':
    main()
