# Data included in this submission

The archived processed 2021–2025 arrays needed by the existing loaders are included, so all three satellite bloom-event forecast demos and the frozen chlorophyll dashboard run offline. Raw environmental downloads are not required to run the demo. Model inputs, targets, frozen statistics and scientific numbers have not been resampled, refitted or changed.

- `processed/`: seven environmental fields and original observation masks, dates and coordinates. Existing optional Sentinel processed placeholders are retained for source-state compatibility; no Sentinel merging was added.
- `processed/bloom_events/dataset_01/`: strict run_01 labels and fitting/index metadata.
- `processed/bloom_events_v2/dataset_02/`: unchanged P75 one-observed-day patch≥2 labels for run_02/run_03.
- `processed/bloom_events_v3/`: frozen causal feature array (about 1 GiB), statistics, coast distance and channel manifest.
- `processed/training_only_labels/`, `sequences/corrected/` and `../outputs/improvement/baseline/labels/`: earlier fitting/integrity dependencies. Apparently duplicate files are kept only where existing paths or fingerprint checks require them; safe internal aliases remove duplicate storage.
- `raw/hab_source_audit_20261024/`: approximately 1.3 MiB of public source-audit evidence. These records are **not** accepted daily positive/monitored-negative training observations; the verified-observation workflow remains untrained/not configured.

## Large raw data intentionally omitted

The retained source download scripts describe the original acquisition requests for **2021-01-01 through 2025-12-31**, Kerala/surrounding Arabian Sea, approximately 74–77.5°E and 8–13°N. Use your own provider account and credentials; no `.env`, token, `.netrc`, CDS or Copernicus credentials are bundled. No downloads were performed during submission packaging.

| Source | Product/variables | Exact destination | Retained acquisition/preprocessing code |
|---|---|---|---|
| [NASA Earthdata](https://search.earthdata.nasa.gov/) / [Ocean Color](https://oceancolor.gsfc.nasa.gov/) | Aqua MODIS `MODISA_L3m_CHL`, daily `chlor_a`, 4 km; `AQUA_MODIS.YYYYMMDD.L3m.DAY.CHL.chlor_a.4km.nc` | `data/raw/Chlorophyll-a/` | `preprocessing/download_data.py`, `preprocessing/preprocess_5years_chlorophyll.py` |
| [Copernicus Marine](https://data.marine.copernicus.eu/) | `METOFFICE-GLO-SST-L4-REP-OBS-SST` sea-surface temperature | `data/raw/sst/` | `preprocessing/download_sst_5years.py`, `preprocessing/preprocess_5years_sst.py` |
| [Copernicus Marine](https://data.marine.copernicus.eu/) | `cmems_mod_glo_phy_my_0.083deg_P1D-m`, surface current `uo` / `vo` | `data/raw/currents/` | `preprocessing/download_currents_5years.py`, `preprocessing/preprocess_5years_currents.py` |
| [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) | ERA5 single levels, hourly 10 m U/V wind | `data/raw/wind/` | `preprocessing/download_era5_wind.py`, `preprocessing/preprocess_5years_wind.py` |
| Same CDS product | ERA5 total precipitation | `data/raw/rainfall/` | `preprocessing/download_era5_rainfall.py`, `preprocessing/preprocess_5years_rainfall.py` |

Optional acquisition dependencies are listed in `requirements-download.txt`; the submission demo needs only `requirements.txt`. Provider product versions and availability may change. Do not overwrite included frozen arrays when redownloading or rerunning preprocessing; use a separate directory and verify provenance first. Existing field-source URLs and checksums are in `raw/hab_source_audit_20261024/retrieval.json` and `../docs/results/hab_submission/source_audit.json`.

## One optional raw-product test

The original suite runs 130 tests: 129 pass and one raw-source test is skipped. Three additional submission-portability tests also pass because raw MODIS downloads are excluded. To enable that specific test, retrieve the exact February 1, 2024 NASA granule above into `data/raw/Chlorophyll-a/AQUA_MODIS.20240201.L3m.DAY.CHL.chlor_a.4km.nc`, then verify SHA256 `374256c250fc419ab6dba43fab391f2073ed58adda461e6f724c35ae22d3de5c`. The frozen-model integration tests and all demo HTTP checks run without that raw file.

## Why the processed files remain relatively large

Existing inference and integrity checks load full daily archives and the full frozen run_03 feature cache. Keeping them avoids changing scientific logic or regenerating labels/features. The complete folder exceeds 500 MB. A future compact edition could regenerate the ~1 GiB feature cache from unchanged included inputs and frozen statistics, and optionally losslessly pack environmental arrays with a validated cache loader. Neither change was applied silently to this submission.
