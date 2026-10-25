"""Verified zone/day HAB labels, independent of chlorophyll proxy labels."""
import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np

SCHEMA = 'hab-occurrence-v1'
SPLITS = ('train', 'validation', 'calibration', 'test')
RECORD_COLUMNS = ('record_id', 'event_id', 'zone_id', 'observed_at_utc', 'available_at_utc',
                  'latitude', 'longitude', 'confirmed_hab', 'observation_scope',
                  'definition_id', 'species', 'verification_method', 'source_reference', 'verified_by')


def utc(value):
    time = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if time.utcoffset() is None or time.utcoffset().total_seconds() != 0:
        raise ValueError('Observation and publication timestamps must explicitly use UTC.')
    return time


def validate_spec(spec):
    if spec.get('schema') != SCHEMA:
        raise ValueError('Unsupported HAB dataset schema.')
    for key in ('definition_id', 'event_definition', 'negative_definition'):
        if not isinstance(spec.get(key), str) or not spec[key].strip() or 'REPLACE' in spec[key]:
            raise ValueError(f'A reviewed {key} is required.')
    if type(spec.get('input_days')) is not int or not 2 <= spec['input_days'] <= 60:
        raise ValueError('input_days must be an integer from 2 to 60 (coverage requires two observed days).')
    if spec.get('output_days') != 5:
        raise ValueError('This workflow predicts five future daily targets.')
    previous = None
    for name in SPLITS:
        bounds = spec.get('splits', {}).get(name)
        if not isinstance(bounds, list) or len(bounds) != 2 or not all(isinstance(x, str) for x in bounds):
            raise ValueError(f'Explicit start/end dates required for {name}.')
        if any(date.fromisoformat(x).isoformat() != x for x in bounds):
            raise ValueError('Split dates must use YYYY-MM-DD.')
        first, last = map(lambda x: np.datetime64(x, 'D'), bounds)
        if np.isnat(first) or np.isnat(last) or first > last or (previous is not None and first <= previous):
            raise ValueError('Splits must be ordered, disjoint calendar periods.')
        previous = last
    zones = spec.get('zones', [])
    if not zones or len({z['id'] for z in zones}) != len(zones):
        raise ValueError('Unique coastal zones are required.')
    for zone in zones:
        if not zone.get('id') or not zone.get('name'):
            raise ValueError('Zone identifiers and names are required.')
        bounds = np.asarray(zone.get('bounds', []), float)
        if (bounds.shape != (4,) or any(type(x) not in (int, float) for x in zone['bounds']) or not np.isfinite(bounds).all()
                or not -180 <= bounds[0] < bounds[2] <= 180 or not -90 <= bounds[1] < bounds[3] <= 90):
            raise ValueError('Zone bounds must be [west, south, east, north].')
    return spec


def split_for(day, spec):
    return next((name for name, (first, last) in spec['splits'].items() if first <= day <= last), None)


def build_labels(path, dates, spec):
    """Only complete-zone monitoring can establish a zone-level negative."""
    validate_spec(spec)
    dates = [str(d) for d in dates]
    lookup = {day: i for i, day in enumerate(dates)}
    zones = {z['id']: (i, z) for i, z in enumerate(spec['zones'])}
    labels = np.full((len(dates), len(zones)), 255, np.uint8)
    events = np.full(labels.shape, '', dtype='<U128')
    records, seen, event_splits = [], set(), {}
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not set(RECORD_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError('HAB observation columns are incomplete.')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed HAB observation row.')
            row = {k: v.strip() for k, v in row.items()}
            for key in RECORD_COLUMNS:
                if key not in ('event_id', 'species') and not row[key]:
                    raise ValueError(f'Missing observation field: {key}')
            if row['record_id'] in seen:
                raise ValueError('Duplicate HAB record identifier.')
            seen.add(row['record_id'])
            observed, available = utc(row['observed_at_utc']), utc(row['available_at_utc'])
            if available < observed or available > datetime.now(timezone.utc):
                raise ValueError('Publication must follow observation and cannot be in the future.')
            day = str(observed.date())
            if day not in lookup or split_for(day, spec) is None:
                raise ValueError('Observation lies outside the configured dataset/splits.')
            if row['definition_id'] != spec['definition_id'] or row['zone_id'] not in zones:
                raise ValueError('Observation definition or zone differs from the dataset.')
            j, zone = zones[row['zone_id']]
            lat, lon = float(row['latitude']), float(row['longitude'])
            west, south, east, north = zone['bounds']
            if not np.isfinite([lat, lon]).all() or not west <= lon <= east or not south <= lat <= north:
                raise ValueError('Observation coordinates lie outside their declared zone.')
            if row['confirmed_hab'] not in ('true', 'false') or row['observation_scope'] not in ('point', 'complete_zone'):
                raise ValueError('Explicit HAB status and point/complete_zone scope required.')
            positive = row['confirmed_hab'] == 'true'
            if not positive and row['observation_scope'] != 'complete_zone':
                raise ValueError('Point absence cannot establish zone-wide HAB absence.')
            if positive:
                if not row['event_id'] or len(row['event_id']) > 128:
                    raise ValueError('Positive observations require an event_id of at most 128 characters.')
                split = split_for(day, spec)
                if row['event_id'] in event_splits and event_splits[row['event_id']] != split:
                    raise ValueError('One HAB event crosses dataset splits; revise split boundaries.')
                event_splits[row['event_id']] = split
            i = lookup[day]
            if labels[i, j] != 255 and labels[i, j] != int(positive):
                raise ValueError('Conflicting zone/day observations require adjudication.')
            if positive and events[i, j] and events[i, j] != row['event_id']:
                raise ValueError('Multiple event IDs in one zone/day require adjudication.')
            labels[i, j] = int(positive)
            events[i, j] = row['event_id'] if positive else ''
            records.append({**row, 'date': day})
    if not records:
        raise ValueError('No verified HAB observations supplied; no labels were fabricated.')
    return labels, events, records


def sequence_rows(dates, labels, spec):
    rows = []
    n, h = spec['input_days'], spec['output_days']
    for start in range(len(dates) - n - h + 1):
        end, target = start + n + h - 1, start + n
        split = split_for(str(dates[start]), spec)
        if split and split == split_for(str(dates[end]), spec) and np.any(labels[target:end+1] != 255):
            rows.append({'split': split, 'input_start': start, 'target_start': target})
    return rows
