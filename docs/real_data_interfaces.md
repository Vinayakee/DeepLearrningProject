# Real observations and update interfaces

These interfaces extend the existing project. They do not generate satellite measurements, field labels or fake normal observations. No new quality-controlled Sentinel bundle or verified field record was found locally.

This document describes the original ingestion contracts. The newer [runtime workflows](runtime_workflows.md) connect them to guarded live inference, field matching, independent evaluation, polling and notification transports. External delivery is still disabled; original statements about missing integrations are historical context superseded by that guide.

## Sentinel-3 quality-controlled bundle

Provide `data/incoming_qc/sentinel3/observations.npz` and `metadata.json`. The NPZ contains decoded `chlorophyll` and raw unsigned `quality_flags` arrays `(T,120,84)`, the actual project `latitude`/`longitude` vectors, and `timestamps` (one actual acquisition/composite per increasing unique UTC day, strings ending `Z`). Preserve original native products and their quality flags outside this aligned bundle.

Metadata must include `product_id`, `processing_baseline`, `quality_reference`, `crs: EPSG:4326`, decoded `units: mg/m3` or `log10(mg/m3)`, and `native_qc_before_resampling: true`. It must retain the source `flag_masks` mapping and an explicit documented `reject_flags` list. Read bit meanings/masks from each native product's attributes, rather than copying fixed bit positions. The loader requires WATER and rejects cloud/ambiguous/margin, invalid/cosmetic/saturated/suspect, high solar zenith, snow/ice, atmospheric correction failure, glint and the version-specific chlorophyll failure flag.

[EUMETSAT's product guidance](https://user.eumetsat.int/s3/eup-strapi-media/Sentinel_3_OLCI_L2_report_for_baseline_collection_OL_L2_M_003_2_B_c8bbc6d986.pdf) recommends source-attribute flag mappings. [The new water processor documentation](https://user.eumetsat.int/news-events/news/new-sentinel-3-olci-level-2-water-processing) documents changed parameter/flag names. Processing version therefore matters; the adapter supports OC4ME_FAIL or CHLOR_A_FAIL, and requires the applicable glint policy. The strict adapter may reject a valid product with a different schema until its documented policy is reviewed and supported; it will not silently omit QC.

Perform native retrieval QC before resampling values; retain raw flags and transfer exclusion masks conservatively. Never bilinearly interpolate flag integers. Decode scale/offset/fill values and units before ingestion. Match the real project coordinates, UTC dates and marine support; matching array shape or geographic bounding box alone is insufficient. Swath reprojection and composite provenance must be supplied by the source-specific acquisition adapter; the ingestion validator accepts only explicitly aligned bundles.

`upgrade.satellite.load_bundle` performs the remaining contract/QC checks. `compare_and_fill` measures overlap log10 bias/RMSE and median concentration ratio, preserves valid MODIS and fills only its missing cells with valid Sentinel candidates. It returns provenance codes 0=missing, 1=MODIS, 2=Sentinel. This is an experiment helper, never automatic deployment. Agreement, coverage, and class/regional performance must be assessed on training/development data before retention. Proxy ground truth must remain independent of fused inputs. No fusion performance result exists yet.

The old Sentinel exports only retain CHL_OC4ME/dataMask, daily request dates and bbox-based resampling. They lack retrieval flags, acquisition IDs and independent saved source georeferencing. They remain excluded. MODIS coverage experiments are retained unchanged.

`upgrade.modis.read_daily` reads actual NASA daily mapped MODIS products, validates concentration units and source coordinates, decodes CF fill/scale metadata, and uses the source valid_min/valid_max. Values outside that documented range remain missing rather than being clipped. The existing L3 file has no individual L2 retrieval/cloud flags; the adapter cannot diagnose each missing pixel's cause. This reader was tested on a real 2024 local product, and does not overwrite existing observations or labels.

## Verified field observations

Provide a real CSV at `data/verified_hab_records.csv` with columns:

```text
record_id,observed_at_utc,latitude,longitude,confirmed_hab,species,verification_method,source_reference,verified_by
```

`confirmed_hab` is an explicitly verified `true` or `false`, not an inference from a missing entry. Retain actual laboratory/field evidence and verification provenance. `upgrade.field_records.load_verified_records` rejects missing provenance, duplicate IDs, invalid coordinates and ambiguous statuses. Development records must be from 2024; new-period independent records must be after the already inspected 2021–2025 period.

No field records are currently available. The interface does not convert toxin concentrations or species counts into labels without a scientifically defined adjudication rule. It does not refit training labels automatically. Records should enter a separate independent evaluation by matching actual forecast target UTC date, spatial support and lead time. Out-of-grid observations cannot be snapped to a coastal edge and treated as valid forecasts. A new region requires its own supported grid, data and evaluation protocol.

## Live daily input bundle and acquisition adapter

The static fitted archive stays intact. Supply a separate staged directory containing `inputs.npz` and `metadata.json`. NPZ keys: `values` and binary `masks` `(5,120,84,7)`, five consecutive UTC `dates`, and matching coordinate vectors. Invalid/unobserved values retain mask zero. Metadata includes exact `channel_order`, `crs`, units `[mg/m3,degC,m/s,m/s,mm/day,m/s,m/s]`, and `sources` per channel containing actual `product_ids`, `quality_reference` and `native_qc_applied: true`. Source adapters must perform the claimed native QC; the manifest is provenance, not proof of scientific accuracy.

```bash
python -m upgrade.live --input-dir /absolute/path/to/staged_real_inputs
```

Alternatively set `HAB_LIVE_ADAPTER=your_source_module:acquire` and run `python -m upgrade.live`. The function acquires/validates actual products and returns a staged directory. Acquisition credentials remain in the environment: NASA, Copernicus Marine and CDS authentication depend on the selected original source scripts; Sentinel requires CDSE credentials. No generic adapter is configured here because the existing downloaders are fixed historical batch scripts and product delivery/credentials are source-specific.

Schedule that command using an external scheduler only after configuring an acquisition adapter. No scheduler was installed or enabled automatically. It validates grid/channels/units/quality provenance, applies the existing frozen training normalization/flags/calendar features, checks freshness and coverage before calling the frozen ConvLSTM, then applies its existing decision rule and regional aggregation. It writes separate `outputs/live` bundles/status. Failed acquisition retains the last successful update record. API reads recheck data age; stale, insufficient or failed updates cannot be presented as live predictions.

The current outcome is `not_configured`, with no successful live update. A real historical input contract was tested against the shared preprocessing, but no fresh-data result is claimed. `/api/system` exposes update/source/model/calibration state; `/api/live/prediction` returns unavailable until a valid fresh bundle exists. Historical `/api/predict` and five-map dashboard behavior remain unchanged.

Every staged source must also provide verified `observation_end_utc` and `available_at_utc` timestamps, ending `Z`, both preceding the first target day. Missing or overlapping timing returns `input_timing_invalid` without model execution. The inspected nominal February 1, 2024 MODIS product has global time bounds January 31 through February 2 and was reprocessed in July 2026. Daily filename dates therefore do not prove per-region acquisition or historical publication timing. Use verified region-specific bounds when available; do not invent them from the nominal filename. The existing hindcast remains a nominal-date proxy evaluation, not proof of operational forecast lead time.

Live sources also cannot claim acquisition/availability in the future relative to execution time. Source observation age is checked separately from the five-frame calendar ending date, using the existing three-day freshness limit. The first forecast target day must still be in the future. Calendar frames for genuinely unavailable recent observations may stay NaN/mask zero, subject to the frozen coverage gates; adding an empty calendar row cannot make stale source observations fresh.

## Notifications

`upgrade.notifications` supplies a gated, idempotent local outbox. Production eligibility requires a coverage-supported, fresh, non-demo forecast with operational regional validation explicitly true. The current model fails regional validation, so production notifications are blocked. Missing inputs, low coverage and NORMAL predictions do not generate warnings. Explicit local demo entries are marked demo and never sent externally.

For future provider adapters, use environment configuration such as `HAB_SMTP_HOST`, `HAB_SMTP_PORT`, `HAB_SMTP_USER`, `HAB_SMTP_PASSWORD`, `HAB_EMAIL_FROM`, `HAB_EMAIL_TO`, or provider-specific SMS credentials. No credentials or recipients belong in source control. External email/SMS transport is intentionally not activated or claimed implemented: reliable warning skill and explicit delivery authorization are prerequisites. The current usable notification channel is the local/demo outbox.

## Evaluation boundary

All new diagnostics, sampling, support and calibration experiments use 2024. Calibration uses January–June complete windows for fitting and July–December complete windows for assessment, excluding windows crossing that boundary. Both remain development evidence; the checkpoint's early stopping/features were already developed on 2024. Original 2025 outputs and hashes are preserved. No new genuinely independent period/region was available; no second 2025 evaluation or new operational-skill claim is made.
