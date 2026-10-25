# Post-freeze HAB improvement report

The existing repository and completed experiments were preserved. All new scientific development uses 2024. The original baseline checkpoint and saved 2025 evaluation match their archived hashes. 2025 was inspected previously and is not claimed untouched; it was not evaluated again in this round.

## Already completed and retained

Training-only corrected proxy labels, shared normalization/masks, small ConvLSTM, controlled weight/rule/seasonal experiments, coverage abstention, regional/false-warning evaluation, frozen model and 2025 results, five-map API/dashboard and the original 38 tests. Seasonal encoding is retained from its completed controlled experiment; no seasonal or class-weight experiment was repeated.

## New STRONG diagnostics

Unique observed training normal/elevated/strong counts: [423344, 84669, 56450]. Unique 2024 counts: [148271, 10783, 7221]. Rolling forecasts contain 35,970 strong targets; 17,496 have no chlorophyll observation at the cell in the input window. Repeated horizons/origins are not independent samples.

| Cell input coverage | Strong precision | Strong recall | Strong F1 | Strong support |
|---|---:|---:|---:|---:|
| No observed chlorophyll | 0.2814 | 0.0476 | 0.0814 | 17496 |
| Observed chlorophyll | 0.3891 | 0.4534 | 0.4188 | 18474 |

Adjacent proxy quantiles, class imbalance and missing observations are possible contributors. Strata are associations, not causal diagnoses. These data do not establish a model capacity limit. No architecture expansion is justified by this report alone.

Actual missed-pixel examples retain target/input date, coordinates, class confusion, scores, coverage, observed input days and environmental values. Monthly strata use the actual target date, with latitude and input-coverage strata also saved. These remain chlorophyll proxy events, not verified harmful species/toxicity. The diagnostics do not justify capacity expansion.

## Controlled sampling experiment

50% uniform + 50% class-balanced sequence mass; cap unnormalized probability at 3/N, renormalize; replacement N draws/epoch.

One full new run changed training sequence sampling only. Existing class weights, seasonal inputs, corrected labels, filters, optimizer, seed, batch size, early stopping and frozen pixel/regional rules were retained. Thresholds were not retuned.

17 epochs completed; best epoch 13. Uniform rolling training target counts: [2054779, 420605, 281517]. Expected sampled counts: [2981012.598305835, 719209.4242462125, 518357.52571882727].

| Model | Macro F1 | Normal P/R/F1 | Elevated P/R/F1 | Strong P/R/F1 | Pixel false-warning rate |
|---|---:|---|---|---|---:|
| Existing frozen final | 0.4652 | 0.9183 / 0.9244 / 0.9213 | 0.1602 / 0.1804 / 0.1697 | 0.3761 / 0.2560 / 0.3047 | 0.0756 |
| Balanced sequence sampling | 0.4137 | 0.9107 / 0.8807 / 0.8954 | 0.1260 / 0.1944 / 0.1529 | 0.2106 / 0.1778 / 0.1928 | 0.1193 |

Under SAME frozen pixel and regional rules: macro F1 gain >=0.005, neither rare-class F1 decreases, regional any/strong recall does not decrease, false warning days do not increase. No post-hoc threshold retuning.

Reject sampling change; keep existing frozen model.

| Model | Warning days | Correct warnings | False warnings | Missed risk days | Any-risk recall | Strong recall | Insufficient-input frequency | Mean input coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing frozen final | 1 | 1 | 0 | 8 | 0.1111 | 0.0000 | 0.4954 | 8.52% |
| Balanced sequence sampling | 2 | 0 | 2 | 9 | 0.0000 | 0.0000 | 0.4954 | 8.52% |

Full confusion matrices, argmax reference, per-class support, false-warning frequency, missed strong days, abstentions and warning-day coverage are in the machine-readable comparison. Existing baseline/weight/decision/seasonal/final tables are retained in [the prior report](improvement_results.md). Fusion has no comparable result because suitable input data does not exist.

## Regional support and false alarms

Dense extrapolation and sparsely sampled proxy truth use different support; compare an input-observed domain without looking at future target masks.

Using the unchanged thresholds on input-observed cells gave 0 correct warnings, 3 false warnings and 9 misses. Criterion passed: False. The support change was rejected and not deployed.

Candidate daily records retain scores, observed truth fractions, source-input coverage, month and environmental values, including the false-warning cases. Existing false-alarm diagnostics remain unchanged. No threshold was raised to make warnings disappear. Lack of warnings is not evidence of reliable detection.

## Probability calibration

Scalar temperature was fitted on complete January–June 2024 windows and assessed on complete July–December windows, excluding overlapping boundary windows. The checkpoint was already developed/early-stopped on 2024, so this is development assessment rather than an independent calibration test.

Temperature: 0.48397093. Declared criterion passed: True. Active calibration remains disabled; raw pixel decisions and historical 2025 results are unchanged.

| Metric | Raw | Temperature |
|---|---:|---:|
| nll | 0.596388 | 0.465567 |
| brier | 0.313434 | 0.242893 |
| ECE top_label | 0.214379 | 0.038235 |
| ECE 0 | 0.274787 | 0.109322 |
| ECE 1 | 0.156816 | 0.060005 |
| ECE 2 | 0.117971 | 0.049317 |

Per-class Brier and bin counts are saved with the [reliability figure](results/upgrade/calibration_2024.png). Improved aggregate calibration does not make strong-event scores reliable; high-score reliability remains poor. Scores therefore continue to be described as model scores, not operational event probabilities.

## Satellite, field labels and live updates

No new satellite observations or field labels were acquired or fabricated. Coverage did not improve in this round. Existing 2024 MODIS coverage remains 8.30%; saved Sentinel availability remains 37.55% unverified and excluded. New provenance audit found bbox-derived coordinates consistent with the actual project centers, but acquisition IDs, independent source georeferencing and retrieval quality flags are absent.

A real MODIS daily reader validates CF decoding, concentration units, actual coordinates and source valid-range metadata. The Sentinel adapter requires documented processing-version flag mappings, native QC, UTC timestamps, aligned coordinates and units; fusion remains unavailable. Candidate fusion helpers preserve MODIS, quantify overlap agreement and retain provenance, but are not enabled automatically.

Nominal 2024-02-01 MODIS composite has global acquisition bounds 2024-01-31 through 2024-02-02; the inspected reprocessed file was created in July 2026. Nominal calendar windows alone do not establish historical source availability or operational forecast lead time. Region-specific acquisition/publication timing must be validated separately.

Verified field-record validation requires actual verification provenance and explicit confirmed status. It neither invents labels nor refits proxy targets. No independent field period/region was found.

Separate staged live inputs use frozen normalization/calendar features and coverage/freshness gates. A configurable source acquisition adapter can supply real bundles. The current state is not configured; no live-data result is claimed. Failed updates retain the last successful record. Local/demo notification outbox is gated and idempotent; external delivery stays disabled because regional warning validation failed.

Data formats, source quality references, acquisition requirements, field evaluation entry point and notification configuration: [real-data interfaces](real_data_interfaces.md).

## Preserved 2025 evidence

| Metric | Baseline | Existing frozen final |
|---|---:|---:|
| Macro F1 | 0.3833 | 0.3981 |
| Elevated F1 | 0.0000 | 0.2409 |
| Strong F1 | 0.2574 | 0.0604 |

These are saved prior results, not a fresh evaluation of this upgrade. Elevated recovery generalized in that historical result, but strong detection regressed and regional strong recall was zero. No further parameter choice used 2025. New-period independent evaluation is blocked on actual new observations; this limitation cannot be fixed by code alone.

## Engineering checks and run commands

Full suite: **56 Python tests passed**, preserving all original tests. A separate JavaScript request-order regression also passed. Real MODIS contract checks, frozen preprocessing equivalence, calibration, quality flags, sampling, field schema, stale/missing/source-timing behavior, notification gates, new endpoints and real five-map integration are covered. Dependency, JavaScript syntax and whitespace checks passed.

Real HTTP five-map forecast and development calibration transformation passed; model/baseline and saved 2025 result hashes were verified. The browser displayed MODIS/Sentinel-unavailable/live-update/model status, separate STALE DATA and historical forecast states. The covered-date helper selects 2024 observations using input completeness alone. No missing-data day becomes NORMAL.

```bash
source .venv/bin/activate
python -m experiments.finalize verify
python -m unittest discover -s tests -v
python -m pip check
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/dashboard/`. Use “Find covered historical date”, or choose 2024-02-15 and enable Historical demonstration, then Generate forecast. In another activated terminal:

```bash
python tests/http_smoke.py
python -m upgrade.live  # reports not configured until a real acquisition adapter exists
```

Completed development commands (do not repeat existing output directories): `upgrade.audit`, `upgrade.diagnostics`, `upgrade.calibration`, `upgrade.regional_experiment`, `upgrade.sampling`, `upgrade.compare`, `upgrade.verify`, `upgrade.report`. Sampling refuses overwrite; existing reports are retained. No new training/evaluation command replaces the frozen baseline/final or reevaluates 2025.

New files: `upgrade/` modules, `tests/test_upgrade.py`, data-interface/report documentation and public evidence snapshots. Existing edits: `backend/app.py`, `prediction/service.py`, `frontend/index.html`, `frontend/script.js`, `README.md`. Frozen model/preprocessing/label/loss/decision/region code was preserved.

## Quality check and remaining limitations

- Verified: baseline/labels/frozen evaluation preserved; 2024-only diagnostics/calibration/sampling; decision/seasonal experiments retained; explicit missing/low/stale states; separate regional/false-warning evaluation; API/dashboard/tests/integration.
- Implemented but dependent on real data: Sentinel QC/fusion inputs, verified field schema, staged live acquisition/update interfaces and future independent evaluation.
- Local notifications only: regional warning reliability is inadequate; no external email/SMS was sent or enabled.
- Scientific limitations remain: sparse observations, proxy rather than confirmed HAB labels, poor rare-class/regional skill, correlated evaluation samples, single-region evidence and unavailable independent future-period assessment.
- No capacity increase, coverage gain, broad generalization, calibrated operational confidence or new independent test success is claimed.
