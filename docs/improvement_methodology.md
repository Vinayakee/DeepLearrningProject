# Controlled improvement methodology

This extends the existing HAB student project. The original working corrected ConvLSTM checkpoint and fitted artifacts are preserved; no model has been rebuilt or replaced without evaluation.

## Baseline and elevated-class diagnosis

The frozen baseline checkpoint SHA-256 is `0dff8dd0ded97db428cb9694a03a1caf20aa2605322d32ff3dd90992fbea678d`. Its copied checkpoint, architecture, source, labels, preprocessing, training record, decision thresholds and previously measured results are under `outputs/improvement/baseline`. `python -m experiments.freeze_baseline` creates this snapshot once; subsequent calls verify it instead of overwriting it.

2024 contains 798,993 valid rolling pixel forecasts: 709,603 normal, 53,420 elevated and 35,970 strong. Direct argmax predicts only 1,308 elevated pixels; elevated recall is 0.0017 and F1 is 0.0033. The existing 0.225 strong override reduces elevated predictions to six, none correct. On actual elevated pixels the median normal/elevated/strong scores are 0.6371/0.2274/0.1345. Elevated scores beat normal on only 0.8611% of elevated targets.

The learned ranking is the main failure; postprocessing further suppresses elevated detections. Imbalance, overlapping intermediate proxy labels and scarce chlorophyll inputs are plausible contributors. These diagnostics do not establish a capacity limitation. The architecture's filters remain unchanged.

The 2024 diagnostic JSON retains confusion matrices, class distributions, score quantiles, precision-recall curves/average precision, actual error examples with dates/coordinates/environmental inputs, and coverage strata. Macro F1 is 0.3581 on targets whose input cell has no chlorophyll observations, versus 0.4163 where at least one exists. This association does not prove causality.

## Experimental controls

All fits use 2021–2023 corrected training-only climatology labels, 2024 validation, seed 42, five input days, five output days, batch size two, Adam learning rate 0.001 with gradient clipping, up to 30 epochs, the existing masked loss, early stopping patience four and learning-rate reduction patience two. The target class 255 always means unobserved and is excluded from loss/scoring.

Weighting compares square-root, power 0.75 and inverse-frequency strategies. Each is normalized to the same expected training-pixel loss weight as baseline, so an overall loss multiplier does not confound learning rate. Fixed direct argmax is the primary comparison, with the unchanged baseline strong threshold as a secondary reference. No decision boundary changes during this stage.

The subsequent decision experiment holds each checkpoint fixed. It tests direct argmax, 33 strong override thresholds (0.1–0.9), and 225 positive class-score ratio boundaries. The same finite grid is used for each checkpoint; macro F1 is primary, then minimum rare-class F1, summed rare-class precision and lower false-warning pixel rate. Ratio boundaries change decisions, not the raw displayed model probabilities. They are not probability calibration.

Calendar inputs are a separate baseline-versus-baseline-plus experiment: sine/cosine of `2π*month/12` on each input date, appended after the 14 existing channels. The baseline selected decision rule stays fixed for the controlled comparison. Calendar features are retained only for at least 0.01 absolute macro-F1 gain without more than 0.01 elevated-F1 loss. No future observation or target is used as an input feature.

Final selection uses macro F1 first. Within a 0.005 near-tie band, regional warning F1, rare-class F1, false warnings and missed strong days resolve the choice. Exact settings are saved before retraining. The final checkpoint, fitting artifacts, boundaries and coverage policy are hashed before any new 2025 evaluation.

## Test-period disclosure

2025 was already evaluated and inspected in earlier work on this project. It cannot honestly be called untouched. This improvement round excludes it from diagnostics and all model/threshold/feature selection. The final frozen configuration is evaluated once on this previously inspected holdout, and no parameters are adjusted from those results. Future independent confirmation requires a new period or an external labeled dataset.

## Regional scoring

Pixel classification and regional warning quality are reported separately. Existing regional thresholds remain the training-derived P75/P90 of daily observed strong-label fractions, with at least 5% marine coverage. The model's predicted strong-pixel fraction uses training-supported forecast cells. Actual strong fractions use only observed target labels; target days below the coverage minimum remain unknown.

Primary warning-day counts use horizon one, one target date per origin. Five-horizon totals count forecast cases, not independent days. Input-coverage abstentions on qualified high-risk days count as misses and are also reported separately. A common available-input cohort provides an additional baseline/final comparison. Historical input availability is evaluated at each origin; today's staleness is not used to erase all hindcast predictions.

These are chlorophyll anomaly proxies, not confirmed harmful/toxic blooms. Dense predictions and sparse observed target samples have different spatial coverage. Thresholds and input safeguards remain prototype choices, not operational coastal warning validation. Reanalysis/satellite delivery latency is not established.

## Satellite quality and explicit availability

On the fixed training marine-support mask, 2024 MODIS mean coverage is 8.30%, with 207 empty days and a longest empty run of 12 days. All marine cells are observed at least once in that year, but coverage is intermittent. The cell-day observation map and monthly/latitude summaries are retained in `outputs/improvement/satellite_quality_2024.json` and `modis_observed_days_2024.npy`.

Sentinel-3 has 37.55% unverified 2024 availability. The current downloader saves only CHL_OC4ME and dataMask, converts log10 concentration to mg/m³, and resamples onto an inferred grid. It does not retain per-pixel water retrieval quality/science flags or independent georeferencing. Therefore fusion is rejected. This preserves MODIS inputs and targets rather than claiming additional usable observations. [Official Sentinel Hub documentation](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S3OLCIL2.html) documents dataMask availability and tile-level cloud filtering; these cannot substitute for retrieval quality checks.

The prediction safeguard requires at least 5% mean chlorophyll marine coverage over five input days, observations on at least two input days, and at least one observed day for each required environmental variable. These thresholds are explicit prototype safeguards, not a confidence calibration. Valid observations are reported as cell-days, with required minimums and latest observed dates. Availability states distinguish valid observations, low coverage, insufficient data and missing satellite data. Missing required inputs stop model execution. Low-coverage historical exploration can return maps, but its regional warning labels are withheld.

## Regional limitations and false alarms

The initial 2024 comparison has 78 truth-qualified horizon-one days out of 325 forecast origins, including nine regionally high-risk proxy days. The original baseline has two correctly warned days, nine false-warning days and seven misses. Intermediate weighting plus its optimized decision rule has two correct warnings, ten false warnings and seven misses. Calendar features with square-root weights have one correct warning, one false warning and eight misses. This demonstrates why better pixel F1 alone cannot justify a claim of reliable warnings or of balanced false-alarm reduction.

Exact 2024 cases are retained in `false_alarm_analysis_2024.json` and each regional `daily_cases.csv`: dates, predicted/observed fractions, raw scores, input and target coverage, season and seven observed environmental means. Baseline false warnings cluster in August, September, October and December. August 25/27 have only about 5–6% input coverage and extreme dense predicted strong fractions, but several winter false warnings occur with over 20% input coverage. Scarcity alone cannot explain every error. No individual cloud/retrieval cause or causal environmental driver is claimed from this descriptive analysis. The final combination is evaluated separately before selection.
