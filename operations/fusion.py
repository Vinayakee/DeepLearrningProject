"""Explicit research inference fusion, preserving MODIS and source provenance."""
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from upgrade.satellite import load_bundle, compare_and_fill
from upgrade.live import prepare_inputs, DataTimingError
from preprocessing.data_utils import file_sha256
from operations.provenance import validate_manifest
import config


def stage_fusion(directory, sentinel_dir, destination, pipeline):
    """Missing/rejected Sentinel falls back to untouched, validated MODIS inputs."""
    directory = Path(directory)
    _, dates, metadata = prepare_inputs(directory, pipeline)
    report = {'state': 'sentinel3_unavailable', 'used': False, 'source': 'MODIS',
              'experimental': True, 'operationally_approved': False}
    if not sentinel_dir:
        return directory, report
    sentinel_dir = Path(sentinel_dir)
    if not all((sentinel_dir / p).is_file() for p in ('observations.npz', 'metadata.json')):
        report['reason'] = 'Sentinel bundle is missing; MODIS fallback.'
        return directory, report
    try:
        source_header = json.loads((sentinel_dir / 'metadata.json').read_text())
        if source_header.get('schema') == 'hab-native-olci-daily-v1':
            from operations.sentinel_pilot import load_daily
            sd, sv, sm, source, qc = load_daily(sentinel_dir, pipeline.latitude, pipeline.longitude,
                                              forecast_start=dates[-1] + np.timedelta64(1, 'D'))
        else:
            sd, sv, sm, source, qc = load_bundle(sentinel_dir, pipeline.latitude, pipeline.longitude)
        if not len(sd):
            raise ValueError('Empty Sentinel observations.')
        now = np.datetime64(datetime.now(timezone.utc).replace(tzinfo=None), 's')
        end_times = []
        for key in ('observation_end_utc', 'available_at_utc'):
            value = source.get(key, '')
            if not isinstance(value, str) or not value.endswith('Z'):
                raise DataTimingError('Sentinel source timing provenance is required.')
            stamp = np.datetime64(value[:-1], 's')
            if np.isnat(stamp) or stamp > now or stamp >= dates[-1] + np.timedelta64(1, 'D'):
                raise DataTimingError('Sentinel source was unavailable before the target.')
            end_times.append(stamp)
        if sd[-1] > dates[-1] or sd[0] < dates[0] or (dates[-1] - sd[-1]).astype(int) > config.MAX_DATA_AGE_DAYS:
            report.update(state='sentinel3_stale_or_unaligned', reason='Sentinel dates outside the input window or too old; MODIS fallback.')
            return directory, report
        if end_times[0].astype('datetime64[D]') < sd[-1]:
            raise DataTimingError('Source end precedes a claimed Sentinel observation.')
        if end_times[1] < end_times[0]:
            raise DataTimingError('Sentinel availability precedes acquisition end.')
        with np.load(directory / 'inputs.npz', allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in ('values', 'masks', 'dates', 'latitude', 'longitude')}
        aligned = np.full(arrays['values'].shape[:3], np.nan, np.float32)
        aligned_mask = np.zeros_like(aligned, bool)
        for i, day in enumerate(sd):
            index = int(np.flatnonzero(dates == day)[0])
            aligned[index], aligned_mask[index] = sv[i], sm[i]
        fused, mask, provenance, agreement = compare_and_fill(arrays['values'][..., 0], arrays['masks'][..., 0], aligned, aligned_mask, pipeline.ocean)
        report.update(state='qc_passed' if sm.any() else 'sentinel3_insufficient', qc=qc, agreement=agreement,
                      used=bool(agreement['additional_cell_days']), source='MODIS + Sentinel-3' if agreement['additional_cell_days'] else 'MODIS',
                      source_metadata=source, source_sha256=file_sha256(sentinel_dir / 'observations.npz'),
                      metadata_sha256=file_sha256(sentinel_dir / 'metadata.json'))
        if not report['used']:
            return directory, report
        arrays['values'][..., 0], arrays['masks'][..., 0] = fused, mask
        # The combined channel carries the latest actual observation/publication
        # timestamp of either source, not a rewritten calendar date.
        original = metadata['sources']['chlorophyll']
        metadata['sources']['chlorophyll'] = {**original,
            'product_ids': original['product_ids'] + [source['product_id']],
            'quality_reference': original['quality_reference'] + '; ' + source['quality_reference'],
            **{key: max(original[key], source[key]) for key in ('observation_end_utc', 'available_at_utc')}}
        metadata['sentinel_fusion'] = report
        destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(destination / 'inputs.npz', **arrays)
        np.save(destination / 'chlorophyll_provenance.npy', provenance)
        metadata['provenance_sha256'] = file_sha256(destination / 'chlorophyll_provenance.npy')
        (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        prepare_inputs(destination, pipeline)
        validate_manifest(destination)
        return destination, report
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError):
        report.update(state='sentinel3_rejected', reason='Sentinel QC, alignment, units or timing validation failed; MODIS fallback.')
        return directory, report
