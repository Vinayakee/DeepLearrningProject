"""Fail-closed ingestion of real, georeferenced OLCI observations and flags."""
from pathlib import Path
import json
import numpy as np

COMMON_REJECT = ('CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'INVALID', 'COSMETIC',
                 'SATURATED', 'SUSPECT', 'HISOLZEN', 'SNOW_ICE', 'AC_FAIL')


def quality_control(values, flags, metadata):
    values, flags = np.asarray(values), np.asarray(flags)
    if values.shape != flags.shape or flags.dtype.kind != 'u':
        raise ValueError('Raw unsigned quality flags must match chlorophyll observations.')
    if not metadata.get('product_id') or not metadata.get('processing_baseline') or not metadata.get('quality_reference'):
        raise ValueError('Product, processing baseline and quality-policy reference are required.')
    mapping = metadata.get('flag_masks', {})
    reject = metadata.get('reject_flags', [])
    failure = 'CHLOR_A_FAIL' if 'CHLOR_A_FAIL' in mapping else 'OC4ME_FAIL'
    glint = 'UNCORR_GLINT' if 'UNCORR_GLINT' in mapping else 'MEGLINT'
    required = set(COMMON_REJECT) | {failure, glint}
    if not required.issubset(reject) or not set(reject).issubset(mapping) or 'WATER' not in mapping:
        raise ValueError('Incomplete version-specific retrieval/cloud/glint/atmospheric quality policy.')
    masks = [int(v) for v in mapping.values()]
    if any(v <= 0 or v & (v-1) for v in masks) or len(masks) != len(set(masks)):
        raise ValueError('Flag masks must be distinct positive single-bit masks from the source attributes.')
    f = flags.astype(np.uint64, copy=False)
    water = (f & np.uint64(mapping['WATER'])) != 0
    rejected = np.zeros(values.shape, dtype=bool)
    counts = {}
    for name in reject:
        marked = (f & np.uint64(mapping[name])) != 0
        rejected |= marked
        counts[name] = int(marked.sum())
    units = metadata.get('units')
    if units == 'log10(mg/m3)':
        with np.errstate(over='ignore', invalid='ignore'):
            linear = np.power(10., values.astype(np.float64))
    elif units == 'mg/m3':
        linear = values.astype(np.float64)
    else:
        raise ValueError('Decoded units must explicitly be mg/m3 or log10(mg/m3).')
    valid = water & ~rejected & np.isfinite(values) & np.isfinite(linear) & (linear > 0)
    # No clipping of large bloom values; source retrieval failure flags decide validity.
    clean = np.where(valid, linear, np.nan).astype(np.float32)
    valid &= np.isfinite(clean)
    clean[~valid] = np.nan
    return clean, valid, {'valid_observations': int(valid.sum()), 'rejections_by_flag': counts,
                          'nonwater': int((~water).sum()), 'flag_counts_overlap': True}


def load_bundle(directory, target_latitude, target_longitude):
    directory = Path(directory)
    with np.load(directory / 'observations.npz', allow_pickle=False) as source:
        arrays = {name: source[name] for name in ('chlorophyll', 'quality_flags', 'latitude', 'longitude', 'timestamps')}
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('crs') != 'EPSG:4326' or metadata.get('native_qc_before_resampling') is not True:
        raise ValueError('Require WGS84 coordinates and documented native-pixel QC before resampling.')
    for name, expected in [('latitude', target_latitude), ('longitude', target_longitude)]:
        actual = arrays[name]
        if actual.shape != np.asarray(expected).shape or not np.allclose(actual, expected, atol=1e-7, rtol=0):
            raise ValueError(f'{name} does not match the actual project grid; bbox/shape alone is insufficient.')
    timestamps = arrays['timestamps'].astype(str)
    if any(not t.endswith('Z') for t in timestamps):
        raise ValueError('Acquisition timestamps must be explicitly UTC, ending Z.')
    times = np.array([np.datetime64(t[:-1], 's') for t in timestamps])
    dates = times.astype('datetime64[D]')
    if np.isnat(times).any() or np.any(np.diff(dates).astype(int) <= 0):
        raise ValueError('One documented acquisition/composite per unique increasing UTC day is required.')
    shape = (len(dates), len(target_latitude), len(target_longitude))
    if arrays['chlorophyll'].shape != shape:
        raise ValueError('Observation dates and grid shape differ.')
    values, valid, report = quality_control(arrays['chlorophyll'], arrays['quality_flags'], metadata)
    return dates, values, valid, metadata, report


def compare_and_fill(modis, modis_valid, sentinel, sentinel_valid, marine):
    if any(np.asarray(a).shape != np.asarray(modis).shape for a in (modis_valid, sentinel, sentinel_valid)):
        raise ValueError('Sources must have identical dates and coordinates before comparison.')
    marine = np.broadcast_to(np.asarray(marine, bool), modis.shape)
    m = np.asarray(modis_valid, bool) & np.isfinite(modis) & (modis > 0) & marine
    s = np.asarray(sentinel_valid, bool) & np.isfinite(sentinel) & (sentinel > 0) & marine
    overlap = m & s
    agreement = {'overlap_cell_days': int(overlap.sum())}
    if overlap.any():
        residual = np.log10(sentinel[overlap]) - np.log10(modis[overlap])
        agreement.update(log10_bias=float(residual.mean()), log10_rmse=float(np.sqrt(np.mean(residual**2))),
                         median_ratio=float(np.median(sentinel[overlap] / modis[overlap])))
    # Candidate only: preserve MODIS wherever valid, never replace proxy ground truth.
    fill = ~m & s
    fused = np.where(m, modis, np.where(fill, sentinel, np.nan)).astype(np.float32)
    provenance = np.where(m, 1, np.where(fill, 2, 0)).astype(np.uint8)
    return fused, m | s, provenance, {**agreement, 'modis_valid_cell_days': int(m.sum()),
                                    'additional_cell_days': int(fill.sum()), 'retained_for_training': False}


def availability(directory):
    directory = Path(directory)
    ready = (directory / 'observations.npz').exists() and (directory / 'metadata.json').exists()
    return {'state': 'pending_quality_validation' if ready else 'sentinel3_unavailable',
            'quality_controlled': False, 'fusion_active': False,
            'reason': 'A complete bundle still requires flag, coordinates, units and alignment validation.' if ready else
                      'No georeferenced observations with retained retrieval quality flags are available.'}
