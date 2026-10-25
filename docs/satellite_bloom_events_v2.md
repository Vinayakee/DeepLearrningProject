# Satellite bloom-event forecast: relaxed run_02

Run_01 and all its reports are preserved byte-for-byte. Its strict P90, three-consecutive-observed-day, four-connected patch ≥3-pixel rule produced 23 training components but zero positive validation/test targets; recall is undefined. Run_02 is a separate satellite-label experiment, not verified harmful species, toxicity or operational warning validation. **2025 was previously inspected, including run_01.**

## Bounded definition selection


| ID | P | Observed days | Max missing gap | Patch M | Train events | Train positive % | Val events | Val positive % | Test positives? | Test components |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| A | 75 | 1 | 0 | 2 | 4364 | 24.5290 | 847 | 10.4750 | yes | 951 |
| B | 80 | 1 | 0 | 2 | 3811 | 19.5655 | 680 | 8.3032 | yes | 789 |
| C | 75 | 2 | 1 | 2 | 1136 | 9.7678 | 112 | 2.1859 | yes | 60 |
| D | 80 | 2 | 1 | 2 | 937 | 7.1905 | 86 | 1.7400 | yes | 36 |
| E | 75 | 2 | 2 | 2 | 1210 | 10.8127 | 128 | 2.4172 | yes | 122 |
| F | 80 | 2 | 2 | 1 | 1340 | 8.1529 | 138 | 1.9493 | yes | 113 |

Chosen: A, the first qualifying definition. Selection was saved before test label inspection. No model performance was used.

Missing days are never labelled positive and never count toward duration. Components can bridge bounded missing gaps between observed endpoints; they are not independent biological events.

Positive rates use valid labelled pixel-days, not the entire grid. Test columns contain only existence and component count, with no model results used for selection. The declared first-qualifying rule required ≥50 training components, ≥1,000 training positive pixel-days, ≥10 validation components, ≥100 validation positive pixel-days and ≥1% validation positive rate. All candidates were evaluated on training/validation; A was chosen as the first and simplest, and the selection record was saved before test-label inspection. No further candidates were explored after the test evaluation.

A means chlorophyll anomaly strictly above the monthly training P75 in a daily four-connected patch ≥2 pixels. It requires **one observed day**, so persistence is not established. The selected rule has no temporal gap; the two-day alternatives can bridge only the declared number of raw unobserved days. Unknown days neither receive positive labels nor count toward duration. Observed low days break a multi-day candidate. Unresolved high sub-patches remain unknown. Components count algorithmic connected structures, not independent biological blooms.

## Dataset and training

Fresh artifacts are in `data/processed/bloom_events_v2/dataset_02/`, `outputs/bloom_events/run_02/`, and `docs/results/bloom_events_v2/`. Monthly P75/P80 thresholds and class weights use 2021–2023 only; existing train-only climatology and normalization were verified and reused. Each complete five-input/five-target-day window stays in its chronological split: training 2021–2023, validation 2024, test 2025. Labels/components are constructed separately within splits. This is a pixel proxy experiment, not a biologically adjudicated event identity split.

The unchanged architecture has 8,737 parameters: 8/4-filter ConvLSTM trunk and five sigmoid pixel heads, 120×84 grids, seven environmental variables, seven observation masks and two seasonal channels. Unknown targets are masked; the positive loss weight is the square root of training negative/positive counts. Seed 42, Adam 0.001, clipnorm 1, batch 2, maximum 15 epochs, validation-loss checkpointing and early stopping are retained.

Training completed 14 epochs in 418.92 seconds, selecting epoch 10. Positive weight: 1.754081. Model SHA256: `3e1dd685afda40808a715a69179d29a7e1847223fb4fbaa7ba9edb757d4936a2`. Source snapshots, dependencies, seed, configuration and history are retained in the run directory.


## Validation decision and frozen test

Platt calibration and the maximum-F1 decision cutoff were fitted using observed positive **and negative** targets in complete January–June 2024 windows. Fitting probabilities using positives alone would be invalid. Support requirements were ≥100 forecasts of each class and ≥10 positive target days for fitting, and ≥100 each class / ≥5 positive days for July–December assessment. The checkpoint used full 2024 validation loss, so this assessment remains development evidence, not independent calibration validation.

There were 47,686 positive fit forecasts across 60 target days and 37,516 positive assessment forecasts across 61 days. Calibration was retained: second-half Brier improved from 0.112675 to 0.091788. The frozen calibrated cutoff is 0.149193138. Choices and model/source hashes were frozen before the single run_02 2025 evaluation.

## 2025 results

| Day | Model valid N | Precision | Recall | F1 | IoU / CSI | PR AUC | Persistence available N | Persistence F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 128680 | 0.2234 | 0.6054 | 0.3264 | 0.1950 | 0.2607 | 59622 | 0.3423 |
| +2 | 129087 | 0.2328 | 0.5740 | 0.3313 | 0.1985 | 0.2882 | 58440 | 0.3413 |
| +3 | 127081 | 0.2404 | 0.5855 | 0.3408 | 0.2054 | 0.2465 | 55518 | 0.3107 |
| +4 | 127081 | 0.2290 | 0.5511 | 0.3236 | 0.1930 | 0.2634 | 53143 | 0.3040 |
| +5 | 126794 | 0.2369 | 0.5428 | 0.3298 | 0.1975 | 0.2379 | 52996 | 0.3014 |


### Identical-target persistence comparison


| Day | Shared N | Model P | Model R | Model F1 | Model IoU / CSI | Persistence P | Persistence R | Persistence F1 | Persistence IoU / CSI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 59622 | 0.1797 | 0.5527 | 0.2712 | 0.1569 | 0.3297 | 0.3558 | 0.3423 | 0.2065 |
| +2 | 58440 | 0.2045 | 0.5074 | 0.2915 | 0.1707 | 0.3315 | 0.3516 | 0.3413 | 0.2057 |
| +3 | 55518 | 0.2044 | 0.4841 | 0.2875 | 0.1679 | 0.3002 | 0.3220 | 0.3107 | 0.1839 |
| +4 | 53143 | 0.2032 | 0.4846 | 0.2863 | 0.1671 | 0.3116 | 0.2967 | 0.3040 | 0.1792 |
| +5 | 52996 | 0.2213 | 0.4413 | 0.2948 | 0.1728 | 0.3093 | 0.2939 | 0.3014 | 0.1774 |


Persistence abstains where its last-input observed event state is unknown. Standalone denominators differ; the table above compares identical targets. Persistence has higher F1 and IoU/CSI at **all five horizons**. The neural model has higher matched recall but lower precision. The relaxed definition repairs the absence of evaluable positives; it does not establish a better forecasting model. Precision remains low (~0.22–0.24 on all observed targets), with many false alarms. IoU and CSI are the same binary statistic, not independent metrics. Full JSON includes per-horizon TP/FP/FN/TN, positive counts, PR AUC, Brier and persistence abstentions. Rolling forecasts count the same target on multiple input windows and are correlated.

## Launch and reproduction

```bash
cd /Users/vinayaks/Documents/HAB-Early-Warning-Submission
BLOOM_EVENT_RUN_DIR=outputs/bloom_events/run_02 MPLCONFIGDIR=/tmp/satellite-event-mpl \
  .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8015
```

Open `/dashboard/bloom.html`; choose relaxed or strict. Use the default 2024-02-15 historical demonstration; forecasts cover 16–20 February. Maps and CSVs retain run identity, input/forecast dates and score kind. `/dashboard/` preserves the original chlorophyll baseline; `/dashboard/hab.html` remains `not_configured`. Missing 2025-12-31 inputs are withheld. Unknown run IDs are rejected.

For reproduction, choose **fresh paths**; commands intentionally refuse existing fitted/evaluated paths:

```bash
MPLCONFIGDIR=/tmp/satellite-event-mpl .venv/bin/python -m preprocessing.bloom_events_v2 --output-dir data/processed/bloom_events_v2/reproduction_NEW --report-dir docs/results/bloom_events_v2/reproduction_NEW
MPLCONFIGDIR=/tmp/satellite-event-mpl .venv/bin/python -m training.train_bloom_events --dataset-dir data/processed/bloom_events_v2/reproduction_NEW --output-dir outputs/bloom_events/reproduction_NEW --epochs 15
MPLCONFIGDIR=/tmp/satellite-event-mpl .venv/bin/python -m prediction.bloom_events_v2 select --run-dir outputs/bloom_events/reproduction_NEW --report-dir docs/results/bloom_events_v2/reproduction_NEW
MPLCONFIGDIR=/tmp/satellite-event-mpl .venv/bin/python -m prediction.bloom_events_v2 evaluate --run-dir outputs/bloom_events/reproduction_NEW --report-dir docs/results/bloom_events_v2/reproduction_NEW
MPLCONFIGDIR=/tmp/satellite-event-mpl make check
```

## Limitations and failed objectives

- Strict run_01 cannot evaluate detection recall: zero positive validation/test targets.
- Relaxed run_02 fails to beat persistence F1/IoU on matched targets; no superior skill claim.
- Single-day satellite patches are weaker evidence than persistent bloom events and cannot establish harmful occurrence.
- Extremely sparse chlorophyll observations, non-random missingness and correlated pixel forecasts limit generalization. Model evaluation includes sparse input windows; dashboard inference additionally applies input-coverage gates.
- Calibration assessment shares development year 2024 with checkpoint selection; 2025 was previously inspected. No untouched independent assessment or operational calibration claim.
- Five-day rather than proposed fourteen-day lookback; existing 2021–2025 environmental period. No live acquisition or verified satellite publication latency.
- No independently verified species/toxicity/daily field-positive and monitored-negative labels. The verified-HAB workflow remains unavailable. No field-event detection-X-of-N result is justified.

See [evaluation](results/bloom_events_v2/evaluation.md), [candidate selection](results/bloom_events_v2/selection_before_test_labels.json), [decision](results/bloom_events_v2/decision.json), and [preserved strict methodology](satellite_bloom_events.md).
