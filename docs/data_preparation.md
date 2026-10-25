# Existing data preparation and acquisition

The reproducible modelling path starts from the existing aligned seven-field dataset. Raw data and trained weights are intentionally not committed. Sharing code alone cannot reproduce scientific results without these real data. Copy the aligned input files listed in README, then run the corrected climatology → labels → sequences → training → evaluation → forecast commands.

The existing acquisition scripts remain available. They are manual batch scripts, use repository-relative paths and require real service accounts. Run from the repository root with the virtual environment activated; install `requirements-download.txt` if acquisition is needed. Existing raw files are skipped by downloaders. Do not invoke five-year acquisition merely to run tests.

| Field | Retained acquisition | Retained alignment |
|---|---|---|
| Aqua MODIS daily L3 4 km chlorophyll `chlor_a` | `preprocessing/download_data.py` (NASA Earthdata `.netrc`) | `preprocess_5years_chlorophyll.py` creates reference grid/dates first |
| SST `analysed_sst` | `download_sst_5years.py` (Copernicus Marine CLI/login) | `preprocess_5years_sst.py` |
| Surface currents `uo`, `vo` | `download_currents_5years.py` (Copernicus Marine login) | `preprocess_5years_currents.py` |
| Wind `u10`, `v10` | `download_era5_wind.py` (CDS API credentials/licence) | `preprocess_5years_wind.py` |
| Daily precipitation `tp` | `download_era5_rainfall.py` (CDS API credentials/licence) | `preprocess_5years_rainfall.py` |
| Sentinel-3 OLCI chlorophyll | `download_sentinel3_5years.py` (CDSE OAuth environment variables) | Export already maps to the reference grid; coverage analysis only |

After real downloads, run the alignment scripts in table order (chlorophyll first). Then run `python preprocessing/verify_all_data.py` and `python preprocessing/report_data_quality.py`. Inspect reports before creating corrected labels. Raw rainfall files in this dataset end in `.zip` but are actually HDF/NetCDF; the retained rainfall reader correctly opens them as datasets. Wind archives are genuine ZIPs. File extensions alone do not establish format.

These acquisition scripts were retained rather than rewriting working source-specific code. Their remote downloads were not repeated during completion: authenticated service availability, product reprocessing and account licences cannot be established by local unit tests. Before acquiring a new dataset, confirm source variable units and ordering, compare date/grid coverage, and re-fit/retrain the complete corrected pipeline. Current code is explicitly configured for the existing 2021–2025 data, not a continuous operational ingestion service.

`preprocess.py`, `preprocess_month.py`, `align_data.py`, `create_hab_labels.py` and visualization scripts are earlier single-month/label exploration retained as references. They are not commands in the corrected training path. `training/train_backup.py` retains its original source behind an explicit reference function and refuses direct execution, avoiding accidental overwrite of original checkpoints. Sentinel files beginning `test_` are manual download experiments; automated tests live only in `tests/`.

MODIS masks identify available finite positive chlorophyll, not verified HAB events. SST converts Kelvin to Celsius, rainfall metres to mm/day, and bilinear environmental alignment retains finite-value masks. Sentinel `CHL_OC4ME` is exported after conversion from log10 to mg/m³, but `dataMask` is an availability mask rather than a scientific retrieval-quality filter. Strong source disagreement/outliers remain visible in the quality report; the project neither silently clips raw scientific observations nor labels Sentinel quality as validated.

Original source assumptions and official Sentinel documentation are linked in `label_preprocessing.md`. This is a data-dependent research demonstration, not a replacement for field sampling or official coastal advisories.
