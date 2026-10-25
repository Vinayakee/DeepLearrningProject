"""Inspect 2024 satellite coverage without synthesizing observations or labels."""
import numpy as np
import config
from preprocessing.data_utils import month_numbers, load_dates, validate_pair
from preprocessing.report_data_quality import longest_missing_run
from experiments.common import EXPERIMENT_DIR, baseline_pipeline, write_json

DOC = 'https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3OLCIL2.html'


def main():
    pipeline = baseline_pipeline()
    selected = np.flatnonzero((pipeline.dates >= np.datetime64('2024-01-01')) & (pipeline.dates <= np.datetime64('2024-12-31')))
    _, valid = pipeline.valid('chlorophyll', selected)
    dates = pipeline.dates[selected]; denominator = int(pipeline.ocean.sum())
    daily = valid.sum((1, 2)); cell = valid.sum(0)
    modis = {'period': '2024 only', 'marine_support_pixels': denominator,
             'marine_coverage_percent': float(valid.sum() / (len(selected) * denominator) * 100),
             'days_without_marine_observations': int((daily == 0).sum()), 'longest_empty_run_days': longest_missing_run(daily),
             'marine_cells_never_observed_in_2024': int(((cell == 0) & pipeline.ocean).sum()),
             'marine_cell_observed_days_quantiles': np.percentile(cell[pipeline.ocean], [0, 25, 50, 75, 100]).tolist(),
             'months': {str(m): {'mean_coverage_percent': float(daily[month_numbers(dates) == m].mean() / denominator * 100),
                               'empty_days': int((daily[month_numbers(dates) == m] == 0).sum())} for m in range(1, 13)},
             'latitude_bands': {f'{low}..{low+1}N': {'marine_cells': int((pipeline.ocean & ((pipeline.latitude[:, None] >= low) & (pipeline.latitude[:, None] < low+1))).sum()),
                  'valid_cell_days': int(valid[:, (pipeline.latitude >= low) & (pipeline.latitude < low+1)].sum())} for low in range(8, 13)},
             'missing_cause': 'Processed masks cannot distinguish each cloud, glint, retrieval failure or missing acquisition. These are plausible mechanisms, not proven causes for individual gaps.',
             'handling': 'Missing values remain masked. Model inputs use zero standardized value plus observation flag 0. Missing targets remain class 255 and excluded from loss/scoring.'}
    np.save(EXPERIMENT_DIR / 'modis_observed_days_2024.npy', cell)
    sentinel = {'fusion_retained': False, 'reason': 'Only CHL_OC4ME and dataMask were requested/retained. No per-pixel science/quality flags or independent source georeferencing are available in the saved bundle. dataMask and tile-level cloud filtering do not establish valid chlorophyll retrievals.',
                'documentation': DOC, 'future_requirement': 'Acquire original OLCI water products and documented per-pixel quality flags; apply appropriate retrieval/cloud/land/glint checks before resampling, validate georeferencing and MODIS agreement on training/development data, preserve source provenance. No blind fusion or boundary-value clipping.'}
    source = config.PROCESSED_DIR / 'sentinel3_chl_2021_2025.npy'
    if source.exists():
        sentinel_dates = load_dates(config.PROCESSED_DIR, 'sentinel3_dates_2021_2025.txt')
        if not np.array_equal(sentinel_dates, pipeline.dates):
            raise ValueError('Sentinel dates do not match the shared calendar.')
        values = np.load(source, mmap_mode='r'); masks = np.load(config.PROCESSED_DIR / 'sentinel3_mask_2021_2025.npy', mmap_mode='r')
        validate_pair(values, masks, pipeline.shape)
        present = (masks[selected] == 1) & np.isfinite(values[selected]) & (values[selected] > 0) & pipeline.ocean
        sentinel['available'] = True
        sentinel['2024_unverified_availability_percent'] = float(present.sum() / (len(selected) * denominator) * 100)
        sentinel['2024_empty_days'] = int((present.sum((1, 2)) == 0).sum())
    else:
        sentinel['available'] = False
    report = {'modis': modis, 'sentinel3': sentinel}
    write_json(EXPERIMENT_DIR / 'satellite_quality_2024.json', report)
    lines = ['# Satellite quality and coverage investigation', '', f"2024 MODIS marine coverage: {modis['marine_coverage_percent']:.2f}%; {modis['days_without_marine_observations']} empty days; longest empty run {modis['longest_empty_run_days']} days; {modis['marine_cells_never_observed_in_2024']} marine cells never observed.", '', modis['missing_cause'], '', modis['handling'], '', '**Sentinel-3 fusion rejected with the current saved data.** ' + sentinel['reason'], '', f"Unverified 2024 Sentinel availability: {sentinel.get('2024_unverified_availability_percent')}%. This is availability, not quality-controlled usable coverage.", '', sentinel['future_requirement'], '', f'[Official Sentinel Hub documentation]({DOC}) documents log10 units, dataMask availability and tile-based cloud filtering.']
    (EXPERIMENT_DIR / 'satellite_quality_2024.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)

if __name__ == '__main__':
    main()
