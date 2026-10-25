# HAB data quality report

Period: 2021-01-01 to 2025-12-31

Ocean support: 5691 / 10080 grid pixels, derived from training-year marine availability.

| Source | Grid coverage | Ocean coverage | Empty ocean days | Longest gap (days) |
|---|---:|---:|---:|---:|
| chlorophyll | 4.88% | 8.58% | 962 | 22 |
| sst | 54.02% | 95.68% | 0 | 0 |
| wind_u | 100.00% | 100.00% | 0 | 0 |
| wind_v | 100.00% | 100.00% | 0 | 0 |
| rainfall | 100.00% | 100.00% | 0 | 0 |
| current_u | 55.79% | 98.82% | 0 | 0 |
| current_v | 55.79% | 98.82% | 0 | 0 |
| sentinel3 | 21.24% | 37.18% | 107 | 3 |

Coverage is measured before imputation. Ocean support is an availability proxy, not a coastline map.

## chlorophyll: latest five input days

| Date | Ocean coverage |
|---|---:|
| 2025-12-27 | 0.00% |
| 2025-12-28 | 7.17% |
| 2025-12-29 | 0.00% |
| 2025-12-30 | 0.00% |
| 2025-12-31 | 0.00% |

Integrity issue counts: {'masked_nonfinite_pixels': 0, 'finite_unmasked_pixels': 0, 'masked_nonpositive_chlorophyll_pixels': 0}

## sst: latest five input days

| Date | Ocean coverage |
|---|---:|
| 2025-12-27 | 95.68% |
| 2025-12-28 | 95.68% |
| 2025-12-29 | 95.68% |
| 2025-12-30 | 95.68% |
| 2025-12-31 | 95.68% |

Integrity issue counts: {'masked_nonfinite_pixels': 0, 'finite_unmasked_pixels': 0, 'masked_nonpositive_chlorophyll_pixels': 0}

## wind_u: latest five input days

| Date | Ocean coverage |
|---|---:|
| 2025-12-27 | 100.00% |
| 2025-12-28 | 100.00% |
| 2025-12-29 | 100.00% |
| 2025-12-30 | 100.00% |
| 2025-12-31 | 100.00% |

Integrity issue counts: {'masked_nonfinite_pixels': 0, 'finite_unmasked_pixels': 0, 'masked_nonpositive_chlorophyll_pixels': 0}

## wind_v: latest five input days

| Date | Ocean coverage |
|---|---:|
| 2025-12-27 | 100.00% |
| 2025-12-28 | 100.00% |
| 2025-12-29 | 100.00% |
| 2025-12-30 | 100.00% |
| 2025-12-31 | 100.00% |

Integrity issue counts: {'masked_nonfinite_pixels': 0, 'finite_unmasked_pixels': 0, 'masked_nonpositive_chlorophyll_pixels': 0}

## rainfall: latest five input days

| Date | Ocean coverage |
|---|---:|
| 2025-12-27 | 100.00% |
| 2025-12-28 | 100.00% |
| 2025-12-29 | 100.00% |
| 2025-12-30 | 100.00% |
| 2025-12-31 | 100.00% |

Integrity issue counts: {'masked_nonfinite_pixels': 0, 'finite_unmasked_pixels': 0, 'masked_nonpositive_chlorophyll_pixels': 0}

## current_u: latest five input days

| Date | Ocean coverage |
|---|---:|
| 2025-12-27 | 98.82% |
| 2025-12-28 | 98.82% |
| 2025-12-29 | 98.82% |
| 2025-12-30 | 98.82% |
| 2025-12-31 | 98.82% |

Integrity issue counts: {'masked_nonfinite_pixels': 0, 'finite_unmasked_pixels': 0, 'masked_nonpositive_chlorophyll_pixels': 0}

## current_v: latest five input days

| Date | Ocean coverage |
|---|---:|
| 2025-12-27 | 98.82% |
| 2025-12-28 | 98.82% |
| 2025-12-29 | 98.82% |
| 2025-12-30 | 98.82% |
| 2025-12-31 | 98.82% |

Integrity issue counts: {'masked_nonfinite_pixels': 0, 'finite_unmasked_pixels': 0, 'masked_nonpositive_chlorophyll_pixels': 0}

## sentinel3: latest five input days

| Date | Ocean coverage |
|---|---:|
| 2025-12-27 | 55.81% |
| 2025-12-28 | 89.18% |
| 2025-12-29 | 2.79% |
| 2025-12-30 | 69.07% |
| 2025-12-31 | 8.94% |

Integrity issue counts: {'masked_nonfinite_pixels': 0, 'finite_unmasked_pixels': 0, 'masked_nonpositive_chlorophyll_pixels': 0}

Retrieval quality unverified; saved chlorophyll/dataMask arrays contain no quality/science flags.

Boundary diagnostics within ocean support: {'at_0_01_mg_m3_pixels': 20, 'at_100_mg_m3_pixels': 525, 'above_100_mg_m3_pixels': 0}

Inferred downloader grid centres match MODIS: True. Independent per-source georeferencing was not saved.

## Interpretation

- Cloud/observation gaps remain missing; coverage is measured before imputation.
- Chlorophyll anomaly labels do not establish species, toxicity, or confirmed HAB events.
- Sentinel-3 remains excluded from modelling until retrieval quality is checked; boundary values are diagnostic and have not been removed.

Paired source agreement (jointly observed ocean pixels; not HAB validation):

| Split | Paired pixels | Median log10 difference (Sentinel − MODIS) | Median absolute log10 difference |
|---|---:|---:|---:|
| train | 370991 | -0.03966149687767029 | 0.08418703079223633 |
| validation | 117189 | -0.03749585151672363 | 0.06106942892074585 |
| test | 89222 | -0.045554276555776596 | 0.07627379894256592 |

Sentinel Hub documents CHL_OC4ME in log10(mg/m³) and dataMask as data availability. OLCI product documentation describes separate quality/science information. The current downloader retains only chlorophyll and dataMask.

Sources: [Sentinel Hub OLCI L2](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3OLCIL2.html), [Copernicus OLCI products](https://sentiwiki.copernicus.eu/web/olci-products).
