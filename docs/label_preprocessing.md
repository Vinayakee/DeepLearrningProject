# Training-only HAB proxy labels

This step corrects the existing monthly median / relative anomaly label
pipeline. It does not replace the seven environmental variables, change the
grid, impute label gaps, or retrain the model.

## Fitting and application

1. Validate consecutive daily dates, monotonic coordinates, array shapes, and
   binary observation masks. A chlorophyll observation must be masked as
   available, finite, and positive.
2. Build a fixed ocean support mask from the union of available, finite SST,
   current U, and current V cells during 2021–2023. This is a marine-product
   availability proxy, not an independently surveyed coastline. The union
   reduces exclusions caused by a single product's interpolation gaps;
   unsupported coastal cells can still be excluded.
3. Retain cells with at least 20 training chlorophyll observations, as in the
   original approach, and intersect them with ocean support. For each calendar
   month, fit a per-cell median from training observations only. Require at
   least three observed days for that month's median. Three is a configurable
   safeguard against fitting a median from one observation, not a calibrated
   operational threshold.
4. Calculate `(chlorophyll - monthly_median) / monthly_median` on observed,
   supported cells. Fit each month's pooled P75 and P90 anomaly thresholds
   from training observations only; apply those fixed thresholds to all dates.

| Label | Definition |
|---|---|
| 0 | Anomaly below training monthly P75 |
| 1 | Anomaly at/above P75 and below P90 |
| 2 | Anomaly at/above P90 |
| 255 | Missing, outside support, or insufficient baseline observations |

An entirely missing month retains NaN thresholds and labels 255. A dataset
with no usable training climatology or thresholds fails explicitly. Positive
chlorophyll anomalies are proxies; these labels do not establish toxicity,
harmful species, or confirmed HAB events. Thresholds may describe ordinary
variability rather than a harmful bloom.

## Commands and artifacts

Use the existing `.venv` and the commands in the README. Both label commands
accept `--processed-dir` and `--output-dir`; their default output is a separate
`data/processed/training_only_labels/` directory. The climatology command also
accepts `--train-start`, `--train-end`, `--min-observations`, and
`--min-monthly-observations`. The label command reads its fitted date range
from climatology metadata to avoid fitting thresholds for a different period.

| Artifact | Contents |
|---|---|
| `monthly_chl_climatology_2021_2025.npy` | Twelve per-cell training monthly medians; filename retained for compatibility |
| `monthly_chl_observation_count.npy` | Training observation counts per calendar month and cell |
| `pixel_observation_count.npy` | Total training chlorophyll observation counts |
| `ocean_support_mask.npy` | Fixed training marine-product support |
| `stable_ocean_mask.npy` | Ocean support intersected with minimum training chlorophyll coverage |
| `chlorophyll_anomaly_2021_2025.npy` | Relative anomalies across the full dataset |
| `hab_labels_2021_2025.npy` | Full-period proxy labels using training-fitted statistics |
| `hab_p75_monthly.npy`, `hab_p90_monthly.npy` | Twelve training-fitted thresholds each |
| `climatology_metadata.json`, `label_metadata.json` | Date ranges, assumptions, artifact checksums, and class counts |

The label command rejects changed climatology arrays or source files whose
size/modification time changed since fitting. Regenerate climatology first
when inputs change. New outputs are reproducible from existing processed
inputs and are ignored by Git like other processed data. The commands reject
the source directory itself as the output directory to preserve old labels.

The corrected generator now loads these labels and `data/sequences/corrected/`.
Run `python -m preprocessing.pipeline` after label fitting to regenerate the
sequence index and training normalization, since supported targets and valid-pixel
counts differ from the original labels.
Do not mix the old checkpoint/evaluation results with the corrected dataset.

## Coverage and Sentinel-3 assessment

`report_data_quality.py` writes `report.md`, `report.json`, and
`daily_coverage.csv` under `outputs/data_quality/`. It reports whole-grid and
fixed-ocean coverage separately, empty days, longest gaps, yearly/monthly
coverage, recent input coverage, mask/data inconsistencies, and jointly
observed MODIS/Sentinel chlorophyll differences. Integrity problems yield a
nonzero exit status after the report is written. Boundary counts are
diagnostics; no values are clipped or automatically substituted.

The current Sentinel-3 downloader converts CHL_OC4ME from log10(mg/m³) to
linear concentration and retains `dataMask`. Sentinel Hub documents this mask
as data availability; OLCI products have separate quality/science information.
The exported arrays lack those flags and independent per-source georeferencing.
The report checks inferred downloader cell centres against the MODIS grid,
but this cannot independently verify the original raster transforms.

Keep Sentinel-3 as a candidate until retrieval quality, boundary-value piles,
and source agreement are examined using representative original products.
Broader coverage is useful, but is not sufficient evidence to replace MODIS.
No new download or provider credentials are needed for this local assessment.

Sources: [Sentinel Hub OLCI L2 documentation](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3OLCIL2.html),
[Copernicus OLCI product documentation](https://sentiwiki.copernicus.eu/web/olci-products).
