"""Download only native pilot components from the documented anonymous mirror."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

FILES = ('xfdumanifest.xml', 'geo_coordinates.nc', 'chl_oc4me.nc', 'wqsf.nc', 'time_coordinates.nc')


def md5_from_manifest(path):
    result = {}
    for obj in ET.parse(path).getroot().iter():
        if obj.tag.rsplit('}', 1)[-1] != 'dataObject':
            continue
        locations = [e for e in obj.iter() if e.tag.rsplit('}', 1)[-1] == 'fileLocation']
        checksums = [e for e in obj.iter() if e.tag.rsplit('}', 1)[-1] == 'checksum' and e.attrib.get('checksumName') == 'MD5']
        if locations and checksums:
            result[Path(locations[0].attrib['href']).name] = checksums[0].text.strip()
    return result


def download(products, destination):
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    records = []
    for product in products:
        date = product['ContentDate']['Start'][:10]
        if not date.startswith('2024-') or '_OL_2_WFR_' not in product['Name']:
            raise ValueError('This pilot accepts real 2024 marine WFR products only.')
        name = product['Name']; platform = name[:3]
        prefix = f"NTC/{platform}/OL_2_WFR___/{date.replace('-', '/')}/{name}/"
        url = 'https://meeo-s3.s3.amazonaws.com/?' + urlencode({'list-type': '2', 'prefix': prefix, 'max-keys': '1000'})
        with urlopen(url, timeout=40) as response:
            listing = ET.fromstring(response.read())
        sizes = {Path(e.findtext('s:Key', namespaces=ns)).name: int(e.findtext('s:Size', namespaces=ns)) for e in listing.findall('s:Contents', ns)}
        if not all(f in sizes for f in FILES):
            raise ValueError('Required native components missing from mirror; stop rather than substitute data.')
        folder = destination / name; folder.mkdir(exist_ok=True)
        record = {'catalogue': product, 'anonymous_mirror_prefix': prefix, 'files': {}, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat()}
        for filename in FILES:
            path = folder / filename
            source = 'https://meeo-s3.s3.amazonaws.com/' + prefix + filename
            if not path.exists():
                temporary = path.with_suffix(path.suffix + '.partial')
                with urlopen(source, timeout=40) as response, temporary.open('wb') as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                if temporary.stat().st_size != sizes[filename]:
                    raise ValueError('Downloaded size differs from public source metadata.')
                temporary.replace(path)
            raw = path.read_bytes()
            if len(raw) != sizes[filename]:
                raise ValueError('Existing native file size differs.')
            if filename != 'xfdumanifest.xml':
                expected = md5_from_manifest(folder / 'xfdumanifest.xml').get(filename)
                if not expected or hashlib.md5(raw).hexdigest() != expected:
                    raise ValueError('Native file MD5 differs from the preserved source manifest.')
            record['files'][filename] = {'url': source, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'native_manifest_md5_verified': filename != 'xfdumanifest.xml'}
            print(f"Verified {name[:42]} {filename} {len(raw)} bytes", flush=True)
        (folder / 'source_provenance.json').write_text(json.dumps(record, indent=2))
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogue', type=Path, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--max-products', type=int, default=10)
    parser.add_argument('--destination', type=Path, default=Path('data/raw/sentinel3_qc_pilot'))
    args = parser.parse_args()
    products = json.loads(args.catalogue.read_text())['catalogue']['value']
    products = [p for p in products if args.start <= p['ContentDate']['Start'][:10] <= args.end]
    if not products or len(products) > args.max_products:
        parser.error('Empty or oversized pilot; specify bounded real product dates.')
    try:
        records = download(products, args.destination)
        result = {'status': 'success', 'products_downloaded': len(records), 'credentials_used': False, 'products': records}
    except Exception as error:
        result = {'status': 'download_stopped', 'error_type': type(error).__name__, 'reason': str(error), 'evaluation_performed': False}
    target = Path('outputs/sentinel_access/download_result.json')
    target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'products'}, indent=2))


if __name__ == '__main__':
    main()
