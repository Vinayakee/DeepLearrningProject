"""Process real native OLCI pilot files into a checked, separate research bundle."""
import argparse
import json
from pathlib import Path
import numpy as np
import config
from operations.sentinel_native import read_native, aggregate_granules, GUIDE
from preprocessing.data_utils import file_sha256

SCHEMA = 'hab-native-olci-daily-v1'


def process(raw, output, expected_catalogue=None, start='2024-02-11', end='2024-02-14'):
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new versioned pilot directory.')
    folders = [p for p in sorted(raw.glob('*.SEN3'))
               if start <= json.loads((p / 'source_provenance.json').read_text())['catalogue']['ContentDate']['Start'][:10] <= end]
    if not folders:
        raise ValueError('No actual native SEN3 products found.')
    latitude = np.load(config.PROCESSED_DIR / 'latitude_kerala_2021_2025.npy')
    longitude = np.load(config.PROCESSED_DIR / 'longitude_kerala_2021_2025.npy')
    ocean = np.load(config.LABEL_DIR / 'stable_ocean_mask.npy').astype(bool)
    reports, grouped = [], {}
    for folder in folders:
        sums, good, total, report = read_native(folder, latitude, longitude)
        day = report['observation_start_utc'][:10]
        if not day.startswith('2024-'):
            raise ValueError('This development pilot accepts 2024 observations only.')
        reports.append(report)
        grouped.setdefault(day, []).append((sums, good, total))
    dates = sorted(grouped)
    values, masks, good_counts, total_counts, daily = [], [], [], [], []
    for day in dates:
        v, m = aggregate_granules(grouped[day], ocean)
        selected = [r for r in reports if r['observation_start_utc'][:10] == day]
        values.append(v); masks.append(m)
        good_counts.append(np.sum([g[1] for g in grouped[day]], axis=0))
        total_counts.append(np.sum([g[2] for g in grouped[day]], axis=0))
        daily.append({'date': day, 'valid_cells': int(m.sum()), 'support_coverage_percent': float(m.sum()/ocean.sum()*100),
                      'observation_end_utc': max(r['observation_end_utc'] for r in selected),
                      'available_at_utc': max(r['available_at_utc'] for r in selected)})
    actual_ids = [r['source_provenance']['catalogue']['Name'] for r in reports]
    missing = []
    if expected_catalogue:
        expected = json.loads(Path(expected_catalogue).read_text())['catalogue']['value']
        missing = sorted({p['Name'] for p in expected if start <= p['ContentDate']['Start'][:10] <= end} - set(actual_ids))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'observations.npz', chlorophyll=np.stack(values), valid=np.stack(masks),
                        valid_native_samples=np.stack(good_counts), total_native_samples=np.stack(total_counts),
                        dates=dates, latitude=latitude, longitude=longitude)
    metadata = {'schema': SCHEMA, 'crs': 'EPSG:4326', 'units': 'mg/m3', 'native_qc_before_resampling': True,
                'product_id': ';'.join(actual_ids), 'product_ids': actual_ids, 'processing_baseline': 'Collection 003',
                'quality_reference': GUIDE, 'daily': daily,
                'observation_end_utc': max(r['observation_end_utc'] for r in reports),
                'available_at_utc': max(r['available_at_utc'] for r in reports),
                'observations_sha256': file_sha256(output / 'observations.npz'),
                'aggregation': '>=3 valid native samples and >=50% QC-valid covered samples; training support applied.',
                'qc_reports': reports, 'pilot_period': [start, end], 'missing_products': missing, 'pilot_complete': not missing if expected_catalogue else None,
                'operationally_approved': False, 'training_changed': False,
                'limitations': 'Partial granules do not establish complete daily area coverage. Publication latency is retained. No performance or coverage gain is implied.'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return metadata


def load_daily(directory, latitude, longitude, forecast_start=None):
    """Validate exported support and native provenance before using any daily cell."""
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text())
    if (meta.get('schema') != SCHEMA or meta.get('crs') != 'EPSG:4326' or meta.get('units') != 'mg/m3'
            or meta.get('native_qc_before_resampling') is not True
            or meta.get('observations_sha256') != file_sha256(directory / 'observations.npz')):
        raise ValueError('Native daily bundle schema, units or hash is invalid.')
    with np.load(directory / 'observations.npz', allow_pickle=False) as a:
        for key, expected in [('latitude', latitude), ('longitude', longitude)]:
            if a[key].shape != np.asarray(expected).shape or not np.allclose(a[key], expected, atol=1e-7, rtol=0):
                raise ValueError('Native daily grid differs from the model grid.')
        dates = a['dates'].astype('datetime64[D]')
        values, valid, good, total = [a[k].copy() for k in ('chlorophyll', 'valid', 'valid_native_samples', 'total_native_samples')]
    shape = (len(dates), len(latitude), len(longitude))
    if (not len(dates) or np.isnat(dates).any() or np.any(np.diff(dates).astype(int) <= 0)
            or any(x.shape != shape for x in (values, valid, good, total)) or valid.dtype != bool
            or good.dtype.kind not in 'ui' or total.dtype.kind not in 'ui'
            or np.any(good < 0) or np.any(total < good)
            or np.any(valid & ((good < 3) | (good < .5 * total) | ~np.isfinite(values) | (values <= 0)))
            or np.any(~valid & np.isfinite(values))):
        raise ValueError('Native daily sample support or values are invalid.')
    if len(meta['daily']) != len(dates):
        raise ValueError('Daily timing records differ from the observations.')
    eligible = np.ones(len(dates), bool)
    for i, (day, record) in enumerate(zip(dates, meta['daily'], strict=True)):
        if str(day) != record['date']:
            raise ValueError('Daily timing date differs.')
        stamps = []
        for key in ('observation_end_utc', 'available_at_utc'):
            stamp = record[key]
            if not isinstance(stamp, str) or not stamp.endswith('Z'):
                raise ValueError('Explicit native UTC acquisition/publication is required.')
            stamps.append(np.datetime64(stamp[:-1], 'us'))
        if np.isnat(stamps).any() or stamps[0].astype('datetime64[D]') != day or stamps[1] < stamps[0]:
            raise ValueError('Native daily acquisition/publication ordering is invalid.')
        if forecast_start is not None:
            eligible[i] = stamps[1] < np.datetime64(forecast_start, 'us')
    # Late publications stay unavailable; never rewrite acquisition dates to claim lead time.
    valid[~eligible] = False
    values[~valid] = np.nan
    return dates, values, valid, meta, {'late_publication_days_excluded': int((~eligible).sum()),
                                     'valid_cell_days': int(valid.sum()), 'native_qc_verified_export': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=config.ROOT / 'data/raw/sentinel3_qc_pilot')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-catalogue', type=Path)
    parser.add_argument('--start', default='2024-02-11')
    parser.add_argument('--end', default='2024-02-14')
    args = parser.parse_args()
    report = process(args.raw_dir, args.output_dir, args.expected_catalogue, args.start, args.end)
    print(json.dumps({k: report[k] for k in ('product_ids', 'daily', 'pilot_complete', 'missing_products')}, indent=2))


if __name__ == '__main__':
    main()
