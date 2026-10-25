"""Compare geographic land polygons with model support without changing labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from preprocessing.data_utils import file_sha256
import config


def land_fraction(geojson, latitude, longitude, samples=8):
    """Subcell land fraction in WGS84; rectangular angular samples, not exact area."""
    lat, lon = np.asarray(latitude), np.asarray(longitude)
    if (lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or len(lon) < 2
            or not np.isfinite(lat).all() or not np.isfinite(lon).all()
            or not np.all(np.diff(lat) < 0) or not np.all(np.diff(lon) > 0)
            or lat.min() < -90 or lat.max() > 90 or lon.min() < -180 or lon.max() > 180):
        raise ValueError('Require descending WGS84 latitude and ascending longitude centers.')
    dy, dx = -np.diff(lat).mean(), np.diff(lon).mean()
    if not np.allclose(-np.diff(lat), dy, atol=1e-5) or not np.allclose(np.diff(lon), dx, atol=1e-5):
        raise ValueError('Require a regular geographic grid.')
    if not isinstance(samples, int) or not 1 <= samples <= 32:
        raise ValueError('Subcell samples must be an integer in 1..32.')
    if geojson.get('type') != 'FeatureCollection' or geojson.get('crs') not in (
            None, {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}):
        raise ValueError('Supply a WGS84 GeoJSON FeatureCollection of physical land polygons.')
    geometries = [f['geometry'] for f in geojson['features']]
    if not geometries or any(g.get('type') not in ('Polygon', 'MultiPolygon') for g in geometries):
        raise ValueError('Physical land Polygon/MultiPolygon features are required.')
    high = rasterize(((g, 1) for g in geometries), out_shape=(len(lat)*samples, len(lon)*samples),
                     transform=from_origin(lon[0]-dx/2, lat[0]+dy/2, dx/samples, dy/samples),
                     fill=0, dtype='uint8')
    return high.reshape(len(lat), samples, len(lon), samples).mean((1, 3))


def compare(support, fraction):
    support, fraction = np.asarray(support), np.asarray(fraction)
    if support.dtype != bool or support.shape != fraction.shape or not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1)):
        raise ValueError('Boolean support and finite land fractions must share a grid.')
    # Keep mixed coastal cells explicit rather than silently classifying them.
    land, water = fraction >= .95, fraction <= .05
    mixed = ~(land | water)
    return {'support_cells': int(support.sum()), 'reference_land_cells': int(land.sum()),
            'reference_water_cells': int(water.sum()), 'mixed_coastal_cells': int(mixed.sum()),
            'supported_land_cells': int((support & land).sum()),
            'unsupported_water_cells': int((~support & water).sum()),
            'supported_mixed_cells': int((support & mixed).sum()),
            'interpretation': 'Unsupported water may reflect sparse training data. Land disagreements require inspection; reference scale limits coastal precision.',
            'candidate_policy': 'Research mask excludes cells with >=95% reference land; mixed cells remain explicit. Not deployed.'}


def run(reference, source, output, samples=8):
    reference, output = Path(reference), Path(output)
    if output.exists() or not source.strip():
        raise ValueError('Choose a new output directory and provide a source citation.')
    lat = np.load(config.PROCESSED_DIR / 'latitude_kerala_2021_2025.npy')
    lon = np.load(config.PROCESSED_DIR / 'longitude_kerala_2021_2025.npy')
    support = np.load(config.LABEL_DIR / 'stable_ocean_mask.npy').astype(bool)
    fraction = land_fraction(json.loads(reference.read_text()), lat, lon, samples)
    report = {**compare(support, fraction), 'source': source, 'reference_sha256': file_sha256(reference),
              'crs': 'EPSG:4326', 'samples_per_axis': samples, 'active_mask_changed': False,
              'fraction_method': 'Fraction of uniformly spaced angular subcell centers on physical land.'}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'comparison.npz', latitude=lat, longitude=lon, land_fraction=fraction,
                        current_support=support, candidate_support=support & (fraction < .95))
    rows, cols = np.where(support & (fraction >= .95))
    report['supported_land_examples'] = [{'latitude': float(lat[r]), 'longitude': float(lon[c])} for r, c in zip(rows[:20], cols[:20])]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    for ax, values, title in zip(axes, [support, fraction, support & (fraction >= .95)],
                                 ['Current training support', 'Reference land fraction', 'Support on >=95% land'], strict=True):
        image = ax.pcolormesh(lon, lat, values, shading='nearest', vmin=0, vmax=1)
        ax.set(title=title, xlabel='Longitude (°E)', ylabel='Latitude (°N)')
        fig.colorbar(image, ax=ax, shrink=.7)
    fig.savefig(output / 'comparison.png', dpi=140)
    plt.close(fig)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--land-geojson', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.land_geojson, args.source, args.output_dir, args.samples), indent=2))


if __name__ == '__main__':
    main()
