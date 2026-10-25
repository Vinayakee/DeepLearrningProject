# Real Sentinel-3 access investigation

Status: partial downloads; bounded pilot blocked by missing public mirror products.
No fusion performance claim, pixel-QC processing, model evaluation, or production
integration was performed. Existing model checkpoints and 2025 results are unchanged.

## Product and availability

Use Sentinel-3A/B OLCI Level-2 Water Full Resolution (`OL_2_WFR`, 300 m),
non-time-critical marine Collection 003 for this 2024 pilot. Native `CHL_OC4ME`
and its water-quality flags are required, rather than land products, radiances,
or rendered chlorophyll images. The anonymous CDSE catalogue returned 908
2024 products intersecting longitude 74–77.5 E, latitude 8–13 N. Intersection
does not guarantee cloud-free marine observations.

The declared initial pilot is February 11–14, 2024: ten intersecting granules.
The AWS MEEO mirror permits anonymous downloads, but its inspected 2024 marine
NTC listing contains only January/February and does not provide all required
pilot granules. Two February 11 Sentinel-3A granules downloaded completely.
All ten downloaded components passed preserved SHA-256 checks; the eight
NetCDF components also passed the original manifest MD5 checks. Source URLs,
catalogue identifiers, acquisition/publication times and manifests are preserved
under `data/raw/sentinel3_qc_pilot/<original-product-name>/`.

The next granule was absent from the mirror:
`S3B_OL_2_WFR____20240212T051009_20240212T051309_20240213T115221_0179_089_290_2700_MAR_O_NT_003.SEN3`.
The download attempt stopped immediately. The official CDSE native download
probe returned HTTP 401, `Token not found`; authenticated file downloads were
not retried. Network access itself was possible with approved access.
EUMETSAT Data Store is another official source with authenticated API access.

## Exact manual download requirements

Sign in to Copernicus Data Space Browser or EUMETSAT Data Store and download
the **eight products** listed by exact name and catalogue UUID in
[manual_remaining_products.csv](results/sentinel_access/manual_remaining_products.csv).
The CSV includes authenticated CDSE download URLs and acquisition/publication
timestamps. These are the remaining February 12–14 products for the declared
pilot, not a request to download the entire year. Full products total approximately
3.70 GB (decimal); downloading native components individually can reduce this.

Extract each product to
`data/raw/sentinel3_qc_pilot/<exact-original-name>.SEN3/`, without renaming it.
Retain at least:

- `xfdumanifest.xml`
- `geo_coordinates.nc`
- `chl_oc4me.nc`
- `wqsf.nc`
- `time_coordinates.nc`

Preserve the downloaded archive and source/download metadata if available.
Do not download the two already verified February 11 products again. Do not
substitute screenshots, flag-free GeoTIFFs, Sentinel-2 data, or simulated data.
Manual files require checksum/provenance validation before processing; simply
placing files in the directory does not make them trusted observations.

## Required QC and subsequent evaluation

Read native WQSF `flag_meanings`/`flag_masks`, not guessed bit positions.
For marine chlorophyll require WATER and reject CLOUD, CLOUD_AMBIGUOUS,
CLOUD_MARGIN, INVALID, COSMETIC, SATURATED, SUSPECT, HISOLZEN, SNOW_ICE,
HIGHGLINT, AC_FAIL, WHITECAPS, ADJAC, RWNEG_O2 through RWNEG_O8 and OC4ME_FAIL.
The Collection-3 recommendation distinguishes corrected moderate glint
(MEGLINT) from HIGHGLINT. The existing project adapter additionally rejects
MEGLINT; that conservative extra restriction must be made explicit when
integrating, rather than silently changing an existing QC policy.

Actual downloaded native chlorophyll units are `lg(re mg.m-3)`, with CF
scale factor 0.015748031 and offset -2.0. Decode scale/offset once, then
convert log10 chlorophyll to positive finite linear mg/m³. Exclude fill values.
Apply native-pixel QC before aggregation onto the existing geographic grid;
retain rejected/missing cells as missing and record valid sample support.
Use actual row times (`microseconds since 2000-01-01`) and preserve publication
latency so observations unavailable before a forecast target are excluded.

`operations/sentinel_native.py` is preparatory processing code. Its native
flag-mapping contract was checked against both real downloaded granules;
the complete pixel-QC, fusion and evaluation workflow has **not** been verified.

Once the remaining real files are available, validate provenance/checksums,
process QC, integrate through the existing preprocessing/model, and compare
paired MODIS-only versus MODIS+Sentinel predictions on 2024 only. Preserve
MODIS observations, frozen model/decision settings and baseline checkpoint.
Report coverage, per-class metrics, false alarms and regional warnings,
with the four-day pilot limitation. Do not tune on or rerun 2025.

## Evidence and sources

- [Outcome](results/sentinel_access/outcome.json)
- [Download verification](results/sentinel_access/download_verification.json)
- [Official access probe](results/sentinel_access/access_probe.json)
- [Original pilot catalogue](results/sentinel_access/pilot_catalogue.json)
- [OLCI marine product documentation](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3OLCIL2.html)
- [CDSE catalogue and authenticated downloads](https://documentation.dataspace.copernicus.eu/APIs/OData.html)
- [Anonymous AWS Sentinel-3 registry](https://registry.opendata.aws/sentinel-3/)
- [EUMETSAT Data Store access](https://user.eumetsat.int/resources/user-guides/data-store-detailed-guide)
- [Collection-3 QC guide, Table 6](https://user.eumetsat.int/s3/ope-eup-strapi-media/S3_OC_L2_guide_V1_4_Dec2025_fe594734bc.pdf)
