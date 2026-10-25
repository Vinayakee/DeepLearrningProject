# HAB improvement results

2025 was inspected in earlier work. It was excluded from this improvement development; the final frozen configuration is evaluated once. This is a previously inspected holdout, not an untouched test.

**Macro F1 and ELEVATED detection improved, but STRONG detection regressed on 2025. Operational warnings are disabled.** Historical maps, model scores and prototype alert entries remain available through explicit demonstration mode.

## 2024 controlled development results

| Experiment/rule | Macro F1 | Elevated precision | Elevated recall | Elevated F1 | Strong precision | Strong recall | Strong F1 | Normal F1 | False-warning pixel rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original deployed baseline | 0.3878 | 0.0000 | 0.0000 | 0.0000 | 0.1686 | 0.5015 | 0.2523 | 0.9110 | 0.1003 |
| Baseline/fixed argmax | 0.3230 | 0.0696 | 0.0017 | 0.0033 | 0.1292 | 0.0151 | 0.0270 | 0.9388 | 0.0058 |
| Inverse weights/fixed argmax | 0.4025 | 0.1144 | 0.1700 | 0.1368 | 0.1412 | 0.4530 | 0.2153 | 0.8555 | 0.2081 |
| Intermediate weights/fixed argmax | 0.4087 | 0.1366 | 0.0624 | 0.0857 | 0.1975 | 0.2511 | 0.2211 | 0.9193 | 0.0682 |
| Intermediate + selected boundary | 0.4359 | 0.1373 | 0.1893 | 0.1592 | 0.2065 | 0.3233 | 0.2520 | 0.8964 | 0.1292 |
| Baseline + calendar/held boundary | 0.4318 | 0.1569 | 0.1395 | 0.1477 | 0.2127 | 0.2556 | 0.2322 | 0.9154 | 0.0854 |
| Combined calendar + weights/held boundary | 0.4445 | 0.1725 | 0.0876 | 0.1162 | 0.2323 | 0.4151 | 0.2979 | 0.9194 | 0.0819 |
| Final retrained configuration | 0.4652 | 0.1602 | 0.1804 | 0.1697 | 0.3761 | 0.2560 | 0.3047 | 0.9213 | 0.0756 |

Intermediate weighting was retained over inverse frequency because its fixed-rule macro F1 was higher with fewer normal-pixel false positives. Decision boundaries were then tested separately on fixed checkpoints. Calendar inputs passed the predeclared controlled criterion: a 0.0242 macro-F1 gain with improved elevated F1. The combined candidate was trained only after measuring these individual effects. Final selection and thresholds were frozen before retraining; retraining used the same selected configuration.

## 2025 frozen holdout comparison

| Model | Accuracy | Macro F1 | Elevated precision | Elevated recall | Elevated F1 | Strong precision | Strong recall | Strong F1 | Pixel false-warning rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.7876 | 0.3833 | 0.0000 | 0.0000 | 0.0000 | 0.1714 | 0.5170 | 0.2574 | 0.1199 |
| Final | 0.7929 | 0.3981 | 0.1969 | 0.3101 | 0.2409 | 0.1126 | 0.0413 | 0.0604 | 0.1144 |

The selected configuration did not generalize its 2024 strong-class gains: 2025 strong recall fell from 51.70% to 4.13%. This is a material failure, despite the higher macro F1 and recovered elevated class. The frozen configuration was retained for honest reporting; no further tuning used these results.

Pixel metrics use exactly the same valid corrected target labels for baseline and final, including low-input cases. Rolling horizons repeat dates at distinct origins. Regional metrics separately count coverage abstentions as misses on observed high-risk days. No parameters changed after the final freeze.

Per-class details and confusion matrices:

### Baseline

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.9055 | 0.8801 | 0.8926 | 553413 |
| Elevated risk | 0.0000 | 0.0000 | 0.0000 | 53264 |
| Strong bloom risk | 0.1714 | 0.5170 | 0.2574 | 34106 |

Confusion matrix: rows actual, columns predicted normal/elevated/strong.
```text
[487041, 10, 66362]
[34374, 0, 18890]
[16473, 0, 17633]
```

### Final

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.9004 | 0.8856 | 0.8929 | 553413 |
| Elevated risk | 0.1969 | 0.3101 | 0.2409 | 53264 |
| Strong bloom risk | 0.1126 | 0.0413 | 0.0604 | 34106 |

Confusion matrix: rows actual, columns predicted normal/elevated/strong.
```text
[490127, 53613, 9673]
[35312, 16518, 1434]
[18930, 13767, 1409]
```

## Regional warning evaluation

Counts below are truth-qualified horizon-one days, not rolling pixel forecasts. Actual fractions use observed proxy labels only; days below 5% target coverage remain UNKNOWN. Predicted fractions use the dense training-supported domain. P75/P90 thresholds remain unchanged from training. Input abstentions on high-risk days remain misses. All-horizon counts and the common available-input cohort are saved separately.

| Period/model | Qualified days | Actual risk days | Warning days | Correct warnings | False warnings | Misses | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 baseline | 78 | 9 | 11 | 2 | 9 | 7 | 0.1818 | 0.2222 | 0.2000 |
| 2024 final | 78 | 9 | 1 | 1 | 0 | 8 | 1.0000 | 0.1111 | 0.2000 |
| 2025 baseline | 78 | 12 | 16 | 1 | 15 | 11 | 0.0625 | 0.0833 | 0.0714 |
| 2025 final | 78 | 12 | 0 | 0 | 0 | 12 | 0.0000 | 0.0000 | 0.0000 |

**Strong regional warnings must be assessed separately:**

| Period/model | Actual strong days | Correct strong warnings | Missed strong days | False strong warnings | Strong-warning F1 | Unknown target days | Unavailable input origins |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2024 baseline | 5 | 2 | 3 | 5 | 0.3333 | 247 | 152 |
| 2024 final | 5 | 0 | 5 | 0 | 0.0000 | 247 | 161 |
| 2025 baseline | 8 | 1 | 7 | 6 | 0.1333 | 250 | 151 |
| 2025 final | 8 | 0 | 8 | 0 | 0.0000 | 250 | 161 |

| Period/model | False any-warning frequency | Insufficient-input frequency | Mean input coverage | Mean coverage on warning days | Abstained high-risk days |
|---|---:|---:|---:|---:|---:|
| 2024 baseline | 11.54% | 46.77% | 8.52% | 15.69% | 4 |
| 2024 final | 0.00% | 49.54% | 8.52% | 25.64% | 4 |
| 2025 baseline | 19.23% | 46.04% | 7.26% | 11.41% | 8 |
| 2025 final | 0.00% | 49.09% | 7.26% | N/A (no warnings) | 9 |

The 2024 reduction from nine false-warning days to zero also reduced correct any-risk warnings from two to one and missed all five strong days. Zero false warnings therefore does not establish a reliable early-warning detector. Sparse and geographically sampled ground truth differs from dense forecast support, and the labels are not verified toxic blooms. The warning-validation gate was disabled before 2025 testing; no holdout result changed that decision.

## Final configuration and reproducibility

- Checkpoint: `outputs/improvement/final/model.keras`; SHA-256 `1504b6589f86e733f7235000669a1e6ac1ba7fdb8eecb521d542165817a491cc`.
- Original baseline: `models/corrected_convlstm.keras`; SHA-256 `0dff8dd0ded97db428cb9694a03a1caf20aa2605322d32ff3dd90992fbea678d`. Additional complete snapshot in `outputs/improvement/baseline`.
- Architecture: ConvLSTM 8/4 with spatial pooling by two and bilinear restoration, 8,787 parameters, five 120×84 input frames, five three-class output maps. No filter/capacity expansion; the two calendar inputs add only their required input weights.
- Features: seven environmental fields, seven binary observation flags, then month sine/cosine. Training-only log1p chlorophyll/rainfall, median/IQR normalization, ±8 clipping. Missing input values are zero standardized values with flag zero; missing targets remain 255 and are excluded.
- Masked loss retained. Normal/elevated/strong class weights: `[0.7973402827710556, 2.6660631961316086, 3.613398024973734]` (power 0.75, matched expected baseline loss mass).
- Frozen decision: `argmax(p × [1.0, 1.2840254166877414, 0.7788007830714049])`. Positive score ratios adjust decisions, not raw model probabilities or calibrated confidence.
- Regional elevated/strong thresholds: `13.72073536% / 24.17253849%` predicted strong-label fraction; training P75/P90.
- Fit 2021–2023, development 2024, seed 42, Adam 0.001/clipnorm 1, batch two, CPU threads 4/2, 14 epochs, best epoch 10. Callbacks, histories, sources and hashes are saved.

## Satellite coverage and unavailable predictions

2024 MODIS marine coverage is 8.30%, with 207 empty days and a longest empty run of 12 days. Sentinel-3 availability is 37.55%, but no retrieval quality flags were retained. **Fusion rejected**; no new observations or labels were fabricated.

Inference reports cell-day counts, required minima, per-variable coverage/observation dates, freshness and availability. It requires 5% mean marine chlorophyll coverage, at least two observed input days and at least one observed day per environmental variable. Missing required inputs stop model execution; low-coverage historical maps cannot issue reliable warning labels. These are prototype safeguards, not confidence estimates.

## Testing, integration and files

Complete suite: **38 tests passed**, including the original regression tests, class weighting, cyclical features, input/preprocessing consistency, class boundaries, model reload, coverage, missing-input non-execution, regional aggregation/unknown truth/abstentions, API responses and real five-map integration. The dependency check passed. Real HTTP/browser checks are recorded separately.

The existing API/dashboard use the separate final deployment pointer, matching checkpoint/preprocessing/calendar features/frozen decision/coverage policy. Current observations and five forecast dates, class scores, elevated/strong pixel counts, data coverage/freshness and unavailable states are dynamic. No SMS/push or live data ingestion is configured.

Main changes: `experiments/`, shared `preprocessing/pipeline.py` and `features.py`, `prediction/decision.py`, `coverage.py`, `regional.py`, `service.py`, model loader, `config.py`, API/dashboard, tests and these documentation snapshots. Working acquisition code, labels, masked loss, two-layer architecture, dashboard layout and original checkpoints are retained.

Run/train/test instructions: [commands](improvement_commands.md). Detailed diagnoses/controls: [methodology](improvement_methodology.md). Real dashboard/HTTP verification: [integration checks](improvement_integration_checks.md). Machine-readable results and confusion matrix/history plots: `docs/results/improvement/`.

## Remaining limitations

Weak elevated precision/recall, lower strong recall than baseline, inadequate regional strong-warning skill, sparse sampled satellite truth, unverified Sentinel retrieval quality, proxy rather than field HAB labels, unknown product delivery latency, no live refresh and no external notifications remain. Scores are not calibrated confidence. Independent future-period or external verified observations are required for operational validation.
