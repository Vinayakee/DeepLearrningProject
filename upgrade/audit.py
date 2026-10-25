"""Additional provenance/coordinate checks; retain completed coverage experiments."""
import json
import numpy as np
import config
from experiments.common import baseline_pipeline, read_json, write_json
from upgrade.satellite import availability
from upgrade.field_records import field_status


def main():
    output=config.ROOT/'outputs/upgrade/data_audit.json'
    if output.exists():
        report=read_json(output)
        if 'modis_native_product_example' not in report:
            from upgrade.modis import read_daily
            pipeline=baseline_pipeline()
            _,_,metadata=read_daily(config.ROOT/'data/raw/Chlorophyll-a/AQUA_MODIS.20240201.L3m.DAY.CHL.chlor_a.4km.nc',pipeline.latitude,pipeline.longitude)
            report['modis_native_product_example']=metadata
            report['timing_limitation']='Nominal 2024-02-01 MODIS composite has global acquisition bounds 2024-01-31 through 2024-02-02; the inspected reprocessed file was created in July 2026. Nominal calendar windows alone do not establish historical source availability or operational forecast lead time. Region-specific acquisition/publication timing must be validated separately.'
            write_json(output,report)
        print('Existing audit retained; native MODIS acquisition/publication timing recorded.');return
    pipeline=baseline_pipeline()
    existing=read_json(config.ROOT/'outputs/improvement/satellite_quality_2024.json')
    expected_lat=13-(np.arange(120)+.5)*5/120
    expected_lon=74+(np.arange(84)+.5)*3.5/84
    source=config.PROCESSED_DIR/'sentinel3_chl_2021_2025.npy'
    invalid=None
    if source.exists():
        select=np.flatnonzero((pipeline.dates>=np.datetime64('2024-01-01'))&(pipeline.dates<=np.datetime64('2024-12-31')))
        values=np.load(source,mmap_mode='r')[select]
        mask=np.load(config.PROCESSED_DIR/'sentinel3_mask_2021_2025.npy',mmap_mode='r')[select]
        invalid={'flagged_present_but_nonfinite':int(((mask==1)&~np.isfinite(values)).sum()),
                 'flagged_present_but_nonpositive':int(((mask==1)&np.isfinite(values)&(values<=0)).sum())}
    report={'period':'2024 audit; no 2025 predictions or tuning','retained_coverage_results':existing,
            'saved_sentinel_product':'Sentinel Hub sentinel-3-olci-l2, CHL_OC4ME + dataMask export',
            'saved_unit_conversion':'LOG10_MILLIGRAMS_M3 decoded via 10**x after bilinear resampling of log10 values.',
            'timestamp_limitation':'Daily request windows saved, but individual acquisition IDs/times and processing baselines were not retained.',
            'coordinates':{'declared_crs':'EPSG:4326 from request/reprojection source code',
                           'project_latitude_direction':'descending','project_longitude_direction':'ascending',
                           'latitude_center_max_deviation_degrees':float(np.abs(expected_lat-pipeline.latitude).max()),
                           'longitude_center_max_deviation_degrees':float(np.abs(expected_lon-pipeline.longitude).max()),
                           'conclusion':'Bbox-derived centers agree numerically with project grid, but independent georeferencing was not saved; this does not validate each acquisition.'},
            'invalid_saved_2024_sentinel_observations':invalid,
            'quality_controlled_sentinel':availability(config.ROOT/'data/incoming_qc/sentinel3'),
            'field_records':field_status(config.ROOT/'data/verified_hab_records.csv'),
            'quality_reference':'https://user.eumetsat.int/news-events/news/new-sentinel-3-olci-level-2-water-processing',
            'flag_mapping_reference':'https://user.eumetsat.int/s3/eup-strapi-media/Sentinel_3_OLCI_L2_report_for_baseline_collection_OL_L2_M_003_2_B_c8bbc6d986.pdf',
            'fusion_experiment':'Not run: source retrieval flags/acquisition provenance unavailable. QC adapter exists, but coverage and classification improvements cannot be claimed.'}
    write_json(output,report);print(json.dumps(report['quality_controlled_sentinel'],indent=2))

if __name__=='__main__':main()
