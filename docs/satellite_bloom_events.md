# Satellite bloom-event forecast

This is a new, separately trained binary satellite-event experiment. It does **not** turn chlorophyll proxy labels into verified HAB observations. The original three-class chlorophyll model, deployment pointer, verified-HAB workflow and frozen scientific results remain separate.

## Event definition and training-only patch choice

The existing monthly median climatology and relative-anomaly P90 thresholds were fitted on 2021–2023 only. They are reused read-only, with hashes and fitting-period checks. A high pixel-day has relative chlorophyll anomaly **strictly greater** than its training-fitted calendar-month P90. Seasonal support retains the existing training observation minimums.

A conservative, precise implementation of the requested three-day patch rule is used: **the same common four-connected patch of at least M pixels must be observed high on each of three consecutive calendar days**. This requires an observed persistent spatial core rather than inventing an advected track through unobserved water. Every pixel in the qualifying common patch is retrospectively labelled as an event member on all three days. Longer runs can yield overlapping qualifying triples.

Only training data selected M:

| M, grid pixels | Training space/time components | Positive training pixel-days |
|---|---:|---:|
| 1 | 54 | 1,059 |
| 2 | 28 | 978 |
| **3** | **23** | **944** |
| 5 | 19 | 893 |
| 10 | 8 | 677 |
| 20 | 3 | 429 |

The documented training-support criterion chooses the largest non-singleton M retaining at least 20 training components and 80% of the M=1 positive pixel-days. **M=3** is a small spatial noise screen, not a biologically validated area threshold. No validation/test counts or model predictions selected M. Components use six-neighbour connectivity in time/space; observation gaps can split one physical bloom into multiple components, so these are not independent biological events.

## Unknowns and split boundaries

- Unobserved, unsupported or seasonally unsupported targets remain 255 and never enter loss or metrics.
- A measured high pixel interrupted by missing observations may have belonged to a qualifying event. An upper-bound calculation temporarily treats missing cells as potentially high solely to identify **uncertain negatives**; these remain 255. It never creates positive labels from missing values.
- Observed non-high pixels cannot meet this event definition and can be labelled 0. Fully resolved, nonqualifying observed high runs can also be 0.
- All triples are processed separately inside training 2021–2023, validation 2024 and test 2025. Boundary high runs without enough within-split evidence remain unknown. No input-plus-five-target sequence crosses a split boundary.
- Retrospective event membership can require two following observation days. The persistence comparator therefore uses separately stored **as-of confirmations**: only completed trailing three-day patches can become positive at issue time. It never looks up retrospective input labels that depend on later observations. Its latest known state is carried forward only within the five input days; otherwise it abstains.

## Label support actually found

| Split | Components | Positive pixel-days | Negative pixel-days | Valid pixel-days | Positive fraction | Usable windows |
|---|---:|---:|---:|---:|---:|---:|
| Training | 23 | 944 | 510,813 | 511,757 | 0.1845% | 1,012 |
| Validation | **0** | **0** | 159,329 | 159,329 | 0% | 325 |
| Test | **0** | **0** | 126,763 | 126,763 | 0% | 323 |

**There are no confirmed positive validation/test event targets under this strict observed-data definition.** This is a critical limitation, not evidence that coastal blooms were absent. MODIS chlorophyll observation coverage is only about 8.6% over marine availability support, and 6,946 validation / 6,785 test observed high candidate pixel-days remain uncertain. The event rule was not relaxed after inspecting held-out label counts.

Consequently, observed-negative validation loss can select a checkpoint but cannot establish detection generalization. The cutoff stays at the conventional **0.5** because validation detection-threshold fitting is unsupported. Scores are not probability-calibrated. Positive-event recall and PR AUC are not estimable on this test set; F1/IoU/CSI are undefined when TP+FP+FN is zero. Report `N/A`, not perfect accuracy or fabricated detection scores.

## Model and training

The original ConvLSTM trunk is reused without modifying its source or checkpoint: five 120×84×16 input frames, spatial pooling by two, ConvLSTM filters 8/4, layer normalization and bilinear restoration. A fresh five-channel 1×1 sigmoid head produces `(batch,5,120,84)`, one binary score per future day/pixel. All weights start from seed 42; this is a separate training run, not a relabelled output from the frozen model.

Environmental median/IQR statistics, log1p transforms, observation flags and calendar encoding are reused from verified 2021–2023 fitting, copied into the new dataset. Masked weighted binary cross-entropy excludes 255. Positive weight is sqrt(510813/944) = 23.2619, derived from training targets only. Adam starts at 0.001 with clipnorm 1 and batch size two. Checkpoints are selected by 2024 masked weighted validation loss.

The first three actual epochs took 28.90, 27.67 and 29.01 seconds. Their mean projected about 428 seconds for 15 epochs, below the two-hour runtime budget, so the same run/optimizer continued to at most 15 epochs with early stopping and learning-rate reduction. Package versions, source/lock-file snapshots, label/source hashes, training history, runtime pilot and checkpoint identity are recorded in `outputs/bloom_events/run_01`.

The completed run trained **15 epochs in 428.95 seconds**, selecting **epoch 14**. The new model has **8,737 parameters** and SHA-256 `ee8a3f5f2a56523fd987810eb52e1929e0f6fa96356282a169e4469a6943324c`.

## Frozen evaluation and denominator reporting

The new validation decision and checkpoint are frozen before collecting test scores. `test_started.json` permits one new evaluation only; interrupted runs retain the guard. **2025 was previously inspected for the original project and must not be described as untouched.** Its new satellite-event test scores are not used to retune M, weights, architecture, epoch selection or thresholds.

The report compares model predictions with causal persistence at horizons +1 through +5. Each precision, recall, F1, IoU/CSI and Brier entry includes confusion counts and valid-target counts. Standalone persistence can abstain where no input state is known. A separate `common_cohort` table compares model and persistence on exactly the same target pixels. Do not compare differing standalone denominators as though they were paired.

Primary offline model scores include all observed target pixels, even when an input window is sparse; the research dashboard separately enforces the existing input-coverage gates. Rolling origins and repeated horizons overlap and are correlated; they are not independent events. With zero positive test components, neither event recall nor early detection generalization can be assessed. No confirmed-HAB skill or operational reliability is claimed.

Reports: [label audit](results/bloom_events/label_audit.json), [runtime pilot](results/bloom_events/runtime_pilot.json), [frozen metrics](results/bloom_events/run_01/evaluation.md), [full counts/cohorts](results/bloom_events/run_01/evaluation.json).

## Reproduce into fresh paths

The completed artifacts must not be overwritten. The default commands were used once for `dataset_01` / `run_01`. For a separately declared new run, choose fresh paths:

```bash
MPLCONFIGDIR=/tmp/satellite-event-mpl .venv/bin/python -m preprocessing.bloom_events \
  --output-dir data/processed/bloom_events/dataset_02
MPLCONFIGDIR=/tmp/satellite-event-mpl .venv/bin/python -m training.train_bloom_events \
  --dataset-dir data/processed/bloom_events/dataset_02 \
  --output-dir outputs/bloom_events/run_02 --epochs 15
MPLCONFIGDIR=/tmp/satellite-event-mpl .venv/bin/python -m prediction.bloom_events select \
  --run-dir outputs/bloom_events/run_02
MPLCONFIGDIR=/tmp/satellite-event-mpl .venv/bin/python -m prediction.bloom_events evaluate \
  --run-dir outputs/bloom_events/run_02 --report-dir docs/results/bloom_events/run_02
```

These are reproduction commands, not permission to search hyperparameters repeatedly against the inspected test set.

## Demonstrate the completed run

```bash
cd /Users/vinayaks/Documents/HAB-Early-Warning-Submission
BLOOM_EVENT_RUN_DIR=outputs/bloom_events/run_01 MPLCONFIGDIR=/tmp/satellite-event-mpl \
  .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8014
```

Open `/dashboard/bloom.html` on that port, keep **Historical demonstration** enabled, and generate the default 2024-02-15 example. Switch all five forecast dates and download CSV. Insufficient inputs withhold inference. The baseline remains at `/dashboard/`; verified-HAB availability remains at `/dashboard/hab.html`. Do not kill an unidentified service already using port 8000.

`GET /api/bloom/status`, `POST /api/bloom/predict`, `GET /api/bloom/evaluation` and `GET /api/bloom/forecasts/<id>/summary.csv?demo=true` provide the separate workflow. The saved evaluation endpoint reads results; it does not repeat test inference. The optional field-record case study was skipped because month/locality-only records cannot support exact day/pixel matches.

## Final verification

**116 Python tests passed**, with no skips, plus dependency consistency, JavaScript syntax/regressions and whitespace checks. New real-inference HTTP checks passed **19 assertions**; the preserved baseline's HTTP check passed **24**. Actual browser checks covered successful generation, day-five map switching, CSV routing, clearing old maps on missing data and the unchanged `not_configured` verified-HAB page. All five real JSON day bindings were also checked in JavaScript. No test evaluation was repeated. The baseline scientific audit passed and all **27 protected artifacts** match their pre-change SHA-256 values.

Evidence: [new HTTP flow](results/bloom_events/http_smoke.json), [baseline HTTP flow](results/bloom_events/baseline_http_smoke.json), [preservation](results/bloom_events/preservation.json), [checks](results/bloom_events/checks.log), [browser screenshot](results/bloom_events/dashboard_day5.png). No test failed. Detection recall/PR AUC remain unassessable; probability calibration and validation-based threshold optimization were not performed because positive validation targets are absent. The optional field case study was skipped for insufficient date/location precision, not reported as a negative result.
