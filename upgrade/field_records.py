"""Verified field-record interface; never invents labels or silently trains on them."""
import csv
from datetime import datetime
from pathlib import Path
import math

COLUMNS=['record_id','observed_at_utc','latitude','longitude','confirmed_hab','species',
         'verification_method','source_reference','verified_by']


def load_verified_records(path, purpose='independent'):
    if purpose not in ('independent','development'):
        raise ValueError('Explicit independent or development purpose required.')
    with Path(path).open(newline='') as handle:
        reader=csv.DictReader(handle)
        if not set(COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError('Verified field observation schema is incomplete.')
        records=[]; seen=set()
        for row in reader:
            if any(not row[key].strip() for key in ['record_id','observed_at_utc','verification_method','source_reference','verified_by']):
                raise ValueError('Record identity, timestamp and verifiable provenance are required.')
            if row['record_id'] in seen:
                raise ValueError('Duplicate field record ID.')
            seen.add(row['record_id'])
            time=datetime.fromisoformat(row['observed_at_utc'].replace('Z','+00:00'))
            if time.utcoffset() is None or time.utcoffset().total_seconds()!=0:
                raise ValueError('Field observations must have UTC timestamps.')
            lat,lon=float(row['latitude']),float(row['longitude'])
            if not math.isfinite(lat) or not math.isfinite(lon) or not (-90<=lat<=90 and -180<=lon<=180):
                raise ValueError('Invalid field coordinates.')
            if row['confirmed_hab'] not in ('true','false'):
                raise ValueError('Confirmed status must be explicitly adjudicated true/false, never missing-as-negative.')
            date=str(time.date())
            if purpose=='independent' and date<='2025-12-31':
                raise ValueError('Known 2021–2025 period is not an independent new-period assessment.')
            if purpose=='development' and not '2024-01-01'<=date<='2024-12-31':
                raise ValueError('Development field assessment only permits 2024.')
            records.append({**row,'latitude':lat,'longitude':lon,'date':date,'confirmed_hab':row['confirmed_hab']=='true'})
    return records


def field_status(path):
    path=Path(path)
    if not path.exists():
        return {'state':'verified_field_records_unavailable','record_count':0,
                'target':'chlorophyll anomaly proxy; no verified toxicity/species labels'}
    records=load_verified_records(path)
    return {'state':'available' if records else 'verified_field_records_unavailable','record_count':len(records),
            'target':'field-confirmed HAB; independent of proxy labels; do not automatically refit targets'}
