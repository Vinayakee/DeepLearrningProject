"""Audit downloaded public HAB evidence without manufacturing training labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def spreadsheet_rows(path, sheet_name):
    """Read source values only; no formula execution, workbook edits or dependencies."""
    with ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            shared = [''.join(item.itertext()) for item in root.findall('s:si', NS)]
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        sheet = next(s for s in workbook.findall('s:sheets/s:sheet', NS) if s.attrib['name'] == sheet_name)
        relation = sheet.attrib['{'+NS['r']+'}id']
        links = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        target = next(item.attrib['Target'] for item in links if item.attrib['Id'] == relation)
        target = target.lstrip('/') if target.startswith('/') else 'xl/'+target
        root = ET.fromstring(archive.read(target))
        rows = []
        for row in root.findall('s:sheetData/s:row', NS):
            values = {}
            for cell in row.findall('s:c', NS):
                column = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                value = cell.find('s:v', NS)
                text = value.text if value is not None else ''
                if cell.attrib.get('t') == 's':
                    text = shared[int(text)]
                elif cell.attrib.get('t') == 'inlineStr':
                    inline = cell.find('s:is', NS)
                    text = ''.join(inline.itertext()) if inline is not None else ''
                values[column] = text
            rows.append((int(row.attrib['r']), values))
    return rows


def audit(source_dir):
    root = Path(source_dir)
    metadata = json.loads((root/'zenodo_record.json').read_text())
    entry = next(x for x in metadata['files'] if x['key'].endswith('.xlsx'))
    workbook = root/entry['key']
    if 'md5:'+hashlib.md5(workbook.read_bytes()).hexdigest() != entry['checksum']:
        raise ValueError('Public spreadsheet differs from its published checksum.')
    raw = spreadsheet_rows(workbook, 'Indian HAB records (1849-2022)')
    records = [(number, row) for number, row in raw if row.get('A', '').isdigit()]
    recent = [(number, row) for number, row in records if row.get('C', '').isdigit() and 2021 <= int(row['C']) <= 2025]
    candidates = []
    for number, row in recent:
        if 'kerala' not in row.get('G', '').lower() and row.get('E') != 'Southwest Coast':
            continue
        candidates.append({'source': 'zenodo:14690930', 'sheet_row': number, 'catalogue_id': row['A'],
                           'year': int(row['C']), 'month': row.get('B'), 'location_text': row.get('E'),
                           'region_text': row.get('G'), 'reported_taxon': row.get('H'), 'reference': row.get('J'),
                           'training_eligible': False,
                           'blockers': ['No exact daily observation date or coordinates in catalogue',
                                        'Literature report requires original-evidence adjudication',
                                        'No monitored zone-wide negative observations']})
    with (root/'haedat_india.csv').open(encoding='utf-8-sig', newline='') as handle:
        haedat = list(csv.DictReader(handle))
    if any(row['Country'] != 'INDIA' for row in haedat):
        raise ValueError('HAEDAT country filter did not return exclusively India records.')
    dated = [row['Event Date'] for row in haedat if row['Event Date']]
    in_period = [row for row in haedat if '2021-01-01' <= row['Event Date'] <= '2025-12-31']
    geographic = json.loads((root/'obis_kerala_all_years.json').read_text())
    period = json.loads((root/'obis_kerala_2021_2025.json').read_text())
    for row in geographic['results']:
        if not 74 <= row['decimalLongitude'] <= 78 or not 8 <= row['decimalLatitude'] <= 12:
            raise ValueError('OBIS geometry filter returned out-of-bounds records.')
    if geographic['total'] != len(geographic['results']) or period['total'] != len(period['results']):
        raise ValueError('Source query is paginated; inspect all results before reporting coverage.')
    return {'schema': 'hab-label-source-audit-v1', 'environment_period': ['2021-01-01', '2025-12-31'],
            'study_bbox': [74, 8, 78, 12],
            'zenodo': {'doi': metadata['metadata']['doi'], 'license': metadata['metadata']['license']['id'],
                       'published_md5_verified': True, 'catalogue_rows': len(records),
                       'all_india_2021_2025_rows': len(recent), 'kerala_named_recent_rows': sum('kerala' in row.get('G','').lower() for _, row in recent),
                       'daily_dates_present': False, 'coordinates_present': False, 'monitored_negatives_present': False},
            'haedat': {'india_rows': len(haedat), 'actual_event_date_range': [min(dated), max(dated)] if dated else None,
                       'india_2021_2025_event_rows': len(in_period), 'source': 'https://haedat.iode.org/events/export/filtered/csv/?country=57'},
            'obis': {'bbox_all_years_records': geographic['total'], 'bbox_period_records': period['total'],
                     'all_years_record_summary': [{key: row.get(key) for key in ('eventID','eventDate','decimalLatitude','decimalLongitude','coordinateUncertaintyInMeters','occurrenceStatus')} for row in geographic['results']]},
            'candidates_for_evidence_followup_only': candidates, 'accepted_training_records': 0,
            'accepted_monitored_negative_zone_days': 0, 'training_decision': 'blocked_by_insufficient_verified_daily_labels_and_monitored_absences',
            'source_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()},
            'limitations': 'A finite source search, not proof that no suitable observations exist. No automatic species, date, coordinate, absence or event-duration inference. Public occurrence records are not daily monitoring negatives.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.source_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(output), 'decision': result['training_decision'], 'accepted_training_records': 0}))


if __name__ == '__main__':
    main()
