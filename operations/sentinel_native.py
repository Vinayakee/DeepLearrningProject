"""Native Collection-3 OLCI QC and conservative aggregation onto the existing grid."""
import json
import re
from pathlib import Path
import numpy as np
import netCDF4
from operations.sentinel_download import md5_from_manifest, FILES
from preprocessing.data_utils import file_sha256

GUIDE = 'https://user.eumetsat.int/s3/ope-eup-strapi-media/S3_OC_L2_guide_V1_4_Dec2025_fe594734bc.pdf'


def flag_policy(meanings, masks):
    names = meanings.split()
    mapping = dict(zip(names, map(int, masks), strict=True))
    bits = list(mapping.values())
    if len(names) != len(mapping) or len(set(bits)) != len(bits) or any(b <= 0 or b & (b - 1) for b in bits):
        raise ValueError('Native flag names and single-bit masks must be unique.')
    def find(name):
        candidates = [n for n in names if re.sub('[^A-Z0-9]', '', n.upper()) == re.sub('[^A-Z0-9]', '', name.upper())]
        if len(candidates) != 1:
            raise ValueError(f'Native recommended QC flag missing/ambiguous: {name}')
        return candidates[0]
    common = ['CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'INVALID', 'COSMETIC', 'SATURATED', 'SUSPECT', 'HISOLZEN', 'SNOW_ICE', 'HIGHGLINT']
    chain = ['AC_FAIL', 'WHITECAPS', 'ADJAC', *[f'RWNEG_O{i}' for i in range(2, 9)], 'OC4ME_FAIL']
    # Table 6 Collection-3 chlorophyll policy; MEGLINT is not indiscriminately
    # rejected (it denotes corrected moderate glint, unlike HIGHGLINT).
    reject = [find(n) for n in common + chain]
    return mapping, find('WATER'), reject


def grid_bins(latitude, longitude, target_latitude, target_longitude):
    lat, lon = np.asarray(target_latitude, dtype=np.float64), np.asarray(target_longitude, dtype=np.float64)
    if (lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or len(lon) < 2
            or not np.isfinite(lat).all() or not np.isfinite(lon).all()
            or not np.all(np.diff(lat) < 0) or not np.all(np.diff(lon) > 0)):
        raise ValueError('Existing descending-latitude/ascending-longitude grid required.')
    dy, dx = float(np.mean(-np.diff(lat))), float(np.mean(np.diff(lon)))
    if not np.allclose(-np.diff(lat), dy, atol=1e-5) or not np.allclose(np.diff(lon), dx, atol=1e-5):
        raise ValueError('Native aggregation requires a regular target grid.')
    finite = np.isfinite(latitude) & np.isfinite(longitude)
    lat_edges = np.r_[lat[0]+(lat[0]-lat[1])/2, (lat[:-1]+lat[1:])/2, lat[-1]-(lat[-2]-lat[-1])/2]
    lon_edges = np.r_[lon[0]-(lon[1]-lon[0])/2, (lon[:-1]+lon[1:])/2, lon[-1]+(lon[-1]-lon[-2])/2]
    rows = np.searchsorted(-lat_edges, -np.where(finite, latitude, -999), side='right')-1
    cols = np.searchsorted(lon_edges, np.where(finite, longitude, -999), side='right')-1
    inside = finite & (rows >= 0) & (rows < len(lat)) & (cols >= 0) & (cols < len(lon))
    return rows * len(lon) + cols, inside


def read_native(directory, target_latitude, target_longitude):
    import hashlib
    directory = Path(directory)
    provenance = json.loads((directory / 'source_provenance.json').read_text())
    name = provenance['catalogue']['Name']
    if name != directory.name or '_OL_2_WFR_' not in name or not name.endswith('_003.SEN3'):
        raise ValueError('Native product identity must match the Collection-3 marine WFR directory.')
    checksums = md5_from_manifest(directory / 'xfdumanifest.xml')
    for name in FILES:
        path = directory / name
        if file_sha256(path) != provenance['files'][name]['sha256']:
            raise ValueError('Native source changed after download.')
        if name != 'xfdumanifest.xml' and hashlib.md5(path.read_bytes()).hexdigest() != checksums[name]:
            raise ValueError('Original product manifest MD5 mismatch.')
    with netCDF4.Dataset(directory / 'geo_coordinates.nc') as ds:
        latitude = np.ma.filled(ds['latitude'][:], np.nan)
        longitude = np.ma.filled(ds['longitude'][:], np.nan)
    with netCDF4.Dataset(directory / 'time_coordinates.nc') as ds:
        variable = ds['time_stamp']
        stamps = variable[:]
        if (stamps.shape != (latitude.shape[0],) or np.ma.getmaskarray(stamps).any()
                or not np.isfinite(stamps).all() or np.any(np.diff(stamps) < 0)):
            raise ValueError('Native row times are missing, unordered or inconsistent with geolocation.')
        times = netCDF4.num2date([stamps.min(), stamps.max()], variable.units,
                                getattr(variable, 'calendar', 'standard'))
        start, end = [t.isoformat() + 'Z' for t in times]
        publication = provenance['catalogue']['PublicationDate']
        if np.datetime64(publication.rstrip('Z'), 'us') < np.datetime64(end.rstrip('Z'), 'us'):
            raise ValueError('Native acquisition ends after recorded source publication.')
        if start[:10] != end[:10]:
            raise ValueError('Cross-midnight granules require row-level daily partitioning; not supported by this pilot.')
    bins, inside = grid_bins(latitude, longitude, target_latitude, target_longitude)
    del latitude, longitude
    with netCDF4.Dataset(directory / 'wqsf.nc') as ds:
        variable = ds['WQSF']; variable.set_auto_maskandscale(False)
        mapping, water_name, reject = flag_policy(variable.flag_meanings, variable.flag_masks)
        flags = variable[:]
        if flags.shape != inside.shape or flags.dtype.kind != 'u':
            raise ValueError('Native flag/geolocation dimensions or unsigned encoding differ.')
    flag_missing = np.ma.getmaskarray(flags)
    flags = np.ma.filled(flags, 0)
    valid = inside & ~flag_missing & ((flags & np.uint64(mapping[water_name])) != 0)
    counts = {}
    for name in reject:
        marked = (flags & np.uint64(mapping[name])) != 0
        counts[name] = int((inside & marked).sum()); valid &= ~marked
    with netCDF4.Dataset(directory / 'chl_oc4me.nc') as ds:
        variable = ds['CHL_OC4ME']; units = str(variable.units)
        values = np.ma.filled(variable[:], np.nan).astype(np.float64)
    if values.shape != inside.shape:
        raise ValueError('Native chlorophyll/geolocation dimensions differ.')
    # CF scale/offset is decoded by netCDF4; CHL_OC4ME remains log10 by units.
    if not ('lg' in units.lower() or 'log10' in units.lower()):
        raise ValueError(f'Unreviewed native chlorophyll units: {units}')
    if 'mg' not in units.lower():
        raise ValueError('Native concentration reference units must be mg/m3.')
    with np.errstate(over='ignore', invalid='ignore'):
        values = 10. ** values
    valid &= np.isfinite(values) & (values > 0)
    size = len(target_latitude) * len(target_longitude)
    total = np.bincount(bins[inside], minlength=size)
    good = np.bincount(bins[valid], minlength=size)
    sums = np.bincount(bins[valid], weights=values[valid], minlength=size)
    # Equal-area approximation within this small grid; do not interpolate flags
    # or fill clouds via distant nearest-valid neighbors.
    shape = (len(target_latitude), len(target_longitude))
    report = {'native_inside_pixels': int(inside.sum()), 'native_valid_pixels': int(valid.sum()),
              'rejections_by_flag': counts, 'flag_mapping': mapping, 'reject_flags': reject,
              'source_units': units, 'decoded_units': 'mg/m3', 'qc_reference': GUIDE,
              'aggregation': 'QC first, native-pixel mean within actual project cell boundaries; >=50% native samples valid and >=3 valid samples.',
              'aggregation_limit': 'Native sample fraction measures QC among covered samples, not fraction of complete cell area.',
              'observation_start_utc': start, 'observation_end_utc': end, 'available_at_utc': publication,
              'catalogue_time_offset_seconds': float((np.datetime64(start[:-1], 'us') - np.datetime64(provenance['catalogue']['ContentDate']['Start'][:-1], 'us')) / np.timedelta64(1, 's')),
              'source_provenance': provenance}
    return sums.reshape(shape), good.reshape(shape), total.reshape(shape), report


def aggregate_granules(granules, ocean):
    if not granules:
        return np.full(ocean.shape, np.nan, np.float32), np.zeros(ocean.shape, bool)
    sums = np.sum([g[0] for g in granules], axis=0)
    good = np.sum([g[1] for g in granules], axis=0)
    total = np.sum([g[2] for g in granules], axis=0)
    valid = (good >= 3) & (good >= .5 * total) & ocean
    values = np.divide(sums, good, out=np.full(ocean.shape, np.nan), where=valid).astype(np.float32)
    return values, valid
