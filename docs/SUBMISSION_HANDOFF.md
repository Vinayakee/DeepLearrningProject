# Portable submission entry point

Run_03 is the headline satellite bloom-event forecast experiment. The frozen chlorophyll baseline and runs_01–02 remain earlier results. All original scientific statements below are retained; portability and launch instructions are in `../README.md`. Launch from this copy with `python -m submission serve --port 8020` after installation. No original checkout is needed.

# Submission handoff — 24 October 2026

## Additional controlled-input experiment: run_03

Run_03 freezes **V3**, selected solely by mean 2024 F1 from the predeclared V0–V3 ablation. Its 2025 mean observed-target F1 is **0.3798**, mean IoU/CSI **0.2345**. Mean matched-pixel F1: model **0.3423**, persistence **0.3199**. **Run_03 improves mean F1/IoU over run_02 and beats matched persistence at every horizon; report as additional improved-input evidence.**

Platt calibration was rejected on 2024 development data: second-half Brier worsened from 0.099037 to 0.101003. Scores remain raw/uncalibrated; the first-half validation F1 cutoff is 0.274019897. No threshold or input choice changed after test.

| Run | Target/input experiment | Result status |
|---|---|---|
| run_01 | Strict P90, three observed days, patch ≥3 | No positive validation/test targets; recall undefined; unchanged |
| run_02 | P75, one observed day, patch ≥2; original inputs | Observed-target F1 0.324–0.341; persistence wins matched F1/IoU; unchanged |
| run_03 | Same run_02 targets; selected causal input variant V3 | Mean F1 0.3798, mean IoU/CSI 0.2345; see matched pixel/block results |

| Variant | Added group | Channels | Parameters | Best epoch | Mean pixel F1 | Delta | Matched F1 | Persistence matched F1 | Mean block F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | Control | 16 | 8737 | 10 | 0.278723 | — | 0.360607 | 0.494624 | 0.508836 |
| V1 | Persistence state/age | 19 | 9601 | 15 | 0.379773 | +0.101050 | 0.450145 | 0.494624 | 0.613939 |
| V2 | Dynamics | 30 | 12769 | 8 | 0.371725 | -0.008048 | 0.432088 | 0.494624 | 0.610468 |
| V3 | Physical coast distance | 31 | 13057 | 10 | 0.415829 | +0.044104 | 0.476864 | 0.494624 | 0.623788 |

[All per-horizon 2025 pixel/block metrics and matched persistence](results/bloom_events_v3/evaluation.md), [2024 ablation with group changes](results/bloom_events_v3/candidate_comparison.md), [input definitions and limitations](bloom_inputs_v3.md).

**Fixed recipe:** same filter/head sizes, seed 42, masked weighted loss, training-only class weights, optimizer and validation-loss checkpoint rule as run_02. Only the input count changes first-layer parameter count. State/age can use older past observations within a calendar year; dynamics summarize trailing five days at each timestep. Coast distance uses an existing physical-land reference with local approximate kilometre projection. Bathymetry is absent; optional V4 interpolation is skipped. All feature-channel future-perturbation tests pass. Existing seasonal channels are reused.

**Presentation abstract extension:** A controlled input ablation compares the original environmental/mask/seasonal inputs with causally carried satellite-event state and observation age, environmental dynamics, and physical coast distance. The best configuration on 2024 development F1 is frozen before one 2025 assessment, with observed-target and matched-persistence pixel and 10×10 block metrics reported separately. The selected configuration yields mean pixel F1 0.380 and IoU 0.234; matched model/persistence F1 is 0.342/0.320. These remain satellite-derived targets, not verified harmful occurrence. 2025 was previously inspected, rolling forecasts are correlated, and block aggregation is not biological event detection.

**Verification:** 130 Python tests, JavaScript/dependency/whitespace checks, HTTP checks for all three runs and the baseline, and protected-artifact/scientific audits passed. All 129 protected files are unchanged; only the selected variant has a new 2025 score collection. A UTC/local-date fixture failure and the extended-input coverage adapter were fixed without changing scientific decisions or repeating test evaluation.

**Presentation notes:** show V0–V3 validation ablation → identify selected variant before test → show matched denominators and any failure to beat persistence → distinguish broad block detection from pixel extent → retain original chlorophyll baseline and field-observation blocker.

**Launch all three runs on the updated server:**

```bash
cd /Users/vinayaks/Documents/HAB-Early-Warning-Submission
BLOOM_EVENT_RUN_DIR=outputs/bloom_events/run_02 MPLCONFIGDIR=/tmp/satellite-event-mpl \
  .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8016
```

Open [satellite bloom-event forecast](http://127.0.0.1:8016/dashboard/bloom.html), select run_03, keep historical example 2024-02-15, generate and switch days/export. Run_01 and run_02 remain selectable. [Final checks](results/bloom_events_v3/checks.log), [protected hashes](results/bloom_events_v3/preservation.json), [run_03 HTTP](results/bloom_events_v3/http_run03.json).

## Preserved run_01 and run_02 submission context

**Run_01 is preserved as-is.** Strict P90, three consecutive observed days, patch ≥3 pixels: 23 training components, zero validation/test positives; recall is undefined. Its original files and reports remain unchanged.

**Run_02 is separately trained and evaluated once.** First qualifying candidate A: training monthly P75, one observed day, four-connected patch ≥2 pixels, no missing-day gap. Training/validation components: 4,364 / 847; positive labelled pixel-days: 24.529% / 10.475%. Test presence/count was inspected after label selection; 951 satellite components exist. This rule does not establish multi-day persistence. All six [candidate definitions and support statistics](results/bloom_events_v2/candidate_comparison.md) are retained.

Same 8,737-parameter ConvLSTM, five input days, 16 channels, masked weighted loss, seed 42. Training: 14 epochs, best epoch 10, 418.92 seconds. Thresholds/class weights use 2021–2023 only. Calibration and cutoff use 2024 development targets; the checkpoint also used 2024. Platt development Brier improved 0.112675 → 0.091788; calibrated cutoff 0.149193138. Model SHA256 `3e1dd685afda40808a715a69179d29a7e1847223fb4fbaa7ba9edb757d4936a2`.



| Day | Model valid N | Precision | Recall | F1 | IoU / CSI | PR AUC | Persistence available N | Persistence F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 128680 | 0.2234 | 0.6054 | 0.3264 | 0.1950 | 0.2607 | 59622 | 0.3423 |
| +2 | 129087 | 0.2328 | 0.5740 | 0.3313 | 0.1985 | 0.2882 | 58440 | 0.3413 |
| +3 | 127081 | 0.2404 | 0.5855 | 0.3408 | 0.2054 | 0.2465 | 55518 | 0.3107 |
| +4 | 127081 | 0.2290 | 0.5511 | 0.3236 | 0.1930 | 0.2634 | 53143 | 0.3040 |
| +5 | 126794 | 0.2369 | 0.5428 | 0.3298 | 0.1975 | 0.2379 | 52996 | 0.3014 |

## Matched-target comparison

| Day | Shared N | Model P | Model R | Model F1 | Model IoU / CSI | Persistence P | Persistence R | Persistence F1 | Persistence IoU / CSI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 59622 | 0.1797 | 0.5527 | 0.2712 | 0.1569 | 0.3297 | 0.3558 | 0.3423 | 0.2065 |
| +2 | 58440 | 0.2045 | 0.5074 | 0.2915 | 0.1707 | 0.3315 | 0.3516 | 0.3413 | 0.2057 |
| +3 | 55518 | 0.2044 | 0.4841 | 0.2875 | 0.1679 | 0.3002 | 0.3220 | 0.3107 | 0.1839 |
| +4 | 53143 | 0.2032 | 0.4846 | 0.2863 | 0.1671 | 0.3116 | 0.2967 | 0.3040 | 0.1792 |
| +5 | 52996 | 0.2213 | 0.4413 | 0.2948 | 0.1728 | 0.3093 | 0.2939 | 0.3014 | 0.1774 |

Standalone denominators differ. The second table compares identical observed targets. Full JSON retains TP/FP/FN/TN, abstentions, Brier and per-metric counts.


**What failed:** persistence has higher F1 and IoU/CSI on every matched horizon. The model has higher recall with lower precision. Strict-label detection recall remains unevaluable. Relaxed positives permit reporting metrics but do not establish a successful operational model. 2025 was previously inspected, so it is not an untouched holdout. No confirmed harmful occurrence, species, toxicity or biological event detection rate is established.

### Launch and demonstrate

```bash
cd /Users/vinayaks/Documents/HAB-Early-Warning-Submission
BLOOM_EVENT_RUN_DIR=outputs/bloom_events/run_02 MPLCONFIGDIR=/tmp/satellite-event-mpl \
  .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8015
```

Open [satellite bloom-event forecast](http://127.0.0.1:8015/dashboard/bloom.html). Default run_02 historical example ends 2024-02-15 and forecasts 16–20 February; switch all five days and export CSV. Select preserved run_01 to see its strict limitation. Original baseline: `/dashboard/`. Verified-HAB availability: `/dashboard/hab.html` (`not_configured`). Do not terminate unidentified existing servers; use a free port. [Fresh-path reproduction commands and architecture](satellite_bloom_events_v2.md).

### Replacement abstract

We implement a five-day satellite bloom-event forecasting prototype for the Kerala coast using 2021–2025 chlorophyll-a, sea-surface temperature, winds, rainfall and currents. A compact ConvLSTM consumes five daily grids, observation masks and seasonal features and outputs five binary pixel-score maps. Training statistics use 2021–2023; checkpointing and decision fitting use 2024. A strict P90, three-observed-day connected-patch definition yields 23 training components and no positive validation/test targets, making recall undefined. A separate bounded relaxation selects P75 daily patches of at least two pixels from training and validation label support. The second model is frozen and evaluated once on previously inspected 2025 data. Its observed-target F1 ranges from 0.324 to 0.341, but causal persistence achieves higher F1 and intersection-over-union on matched targets at every horizon. The dashboard provides actual historical inference, dates, observation coverage, run selection and exports. Satellite labels do not establish harmful species, toxicity or operational reliability; a separate verified-observation workflow remains untrained pending suitable field records.

**Presentation:** distinguish the satellite proxy from field-verified targets → show strict failure and bounded relaxation table → describe architecture/masked training → demonstrate both runs → present matched persistence comparison and limitations.

**Differences from proposal:** existing 2021–2025 inputs; five rather than fourteen input days; chlorophyll-derived pixel patches rather than verified harmful occurrence; single-day relaxed targets rather than persistent blooms; no live or operational validation. Missing data remain unknown. Connected components are not independent biological events. Calibration assessment shares 2024 development data with checkpoint selection.

**External-data blocker:** precise daily field observations with date/location/event identity, evidence of harmful species or toxicity, plus monitored negatives and coverage are still absent. No genuine verified-HAB model or metrics are claimed.

**Verification:** 122 Python tests passed; JavaScript binding/race, dependency and whitespace checks passed. Local HTTP checks passed: 19 relaxed, 19 strict, 24 baseline, plus invalid-run and cross-run export checks. All 73 snapshotted strict/baseline files are unchanged and both event-model freezes verify. Browser generation, day switching, experiment switching and missing-input clearing were checked. One initial concurrent smoke check hit the intended prediction lock; serial checks passed without code or scientific changes.

**Verification evidence:** [checks](results/bloom_events_v2/checks.log), [relaxed HTTP](results/bloom_events_v2/http_smoke_relaxed.json), [strict HTTP](results/bloom_events_v2/http_smoke_strict.json), [baseline HTTP](results/bloom_events_v2/baseline_http_smoke.json), [preservation](results/bloom_events_v2/preservation.json). These reports are separate from the unchanged run_01 evidence.

## Preserved baseline and verified-HAB handoff

The sections below retain the earlier baseline/verified-HAB context. Current satellite-event training and results are described above.

**Baseline ready to demonstrate:** the working real-data chlorophyll-anomaly ConvLSTM and dashboard.

**Implemented but not trained on real HAB labels:** the separate five-day coastal-zone HAB occurrence workflow. Public source records were obtained and audited, but do not provide sufficient adjudicated daily positives and monitored negatives. There are no verified-HAB accuracy results or deployable HAB checkpoint. Synthetic software tests must not be presented as scientific results.

## Launch and demonstrate

Use the existing verified environment; no dependency reinstall or model retraining is needed for the working demonstration.

```bash
cd /Users/vinayaks/Documents/HAB-Early-Warning-Submission
make serve
```

Open [the baseline dashboard](http://127.0.0.1:8000/dashboard/). If a previous server is already using port 8000, restart it from its original terminal to load the updates, or launch with `--port 8013` and use that port in the browser. Do not stop unrelated processes.

1. Click **Load historical example**. The default supported example ends on 2024-02-15 and forecasts 16–20 February. Alternatively, click **Show eligible dates**, choose a date, and click **Generate forecast**. The list currently contains 177 supported 2024 input dates and shows coverage for each; it does not select dates by future labels or model scores.
2. Switch through Day +1 to Day +5. Each map and score comes from actual model inference on archived environmental inputs. The historical-demo notice remains visible.
3. Switch View to **Observed proxy and input coverage**, then change forecast day. Observation context stays on the input date. Zoom and reset work. For the default example the last input day itself is unobserved, although the complete five-day window has adequate coverage; the grey observation panel correctly represents that missing observation.
4. Download the five-day CSV. It records dates, scores, coverage, freshness and non-operational status.
5. Select 2025-12-31 with historical demonstration enabled to show honest missing-data handling: the window has only 1.44% chlorophyll coverage on one observed day, so the model is withheld.
6. Open [the separate HAB page](http://127.0.0.1:8000/dashboard/hab.html). It correctly states `not_configured` and disables generation rather than substituting chlorophyll classes as confirmed HAB forecasts.

## What exists and what was trained

| Area | Delivered state |
|---|---|
| Environmental dataset | Existing 2021–2025 daily aligned chlorophyll, SST, wind U/V, rainfall and currents U/V; observation masks retained |
| Existing trained model | Frozen `outputs/improvement/final/model.keras`, selected by existing deployment metadata; unchanged |
| Baseline inputs/output | Five input days, 120×84 grid, seven fields + seven masks + two calendar channels; five future three-class anomaly maps |
| HAB label workflow | Reviewed-definition/spec contract, source/identity/date/location checks, presence/monitored absence/unknown targets; event-safe chronological splits |
| HAB model | ConvLSTM 8/4 filters, marine-zone pooling, sigmoid daily heads; `(batch,5,zones)`. Template prefers 14-day lookback |
| HAB training/evaluation | Separate fitted statistics; unknown-target masking; class weights or event-balanced sampling; validation checkpointing; calibration split; frozen test guard; logistic/persistence comparisons; per-horizon/zone and event reports |
| Reproducibility | Source/lock-file snapshots, package versions, seed 42, data manifests, hashes, fitting config and history for new training runs |
| HAB API/dashboard | Implemented, with coverage/freshness gates, five-day scores/categories, zone schematic and CSV export; no real checkpoint configured |
| Earlier verified-HAB continuation | No real HAB training; synthetic fixtures tested software only. The new, real satellite-event experiment is documented at the top of this handoff |

Corrected remaining implementation issues: event lead time uses the full catalogue and distinguishes ongoing-event detection from advance warning; persistence standalone evaluation uses its own prior-observation availability; damaged HAB artifacts return HTTP 503 while invalid requests return 422; the baseline page is clearly named chlorophyll risk and includes eligible-date selection.

## Verified scientific results — chlorophyll proxy only

These values are read from the preserved evaluation, not recomputed by rerunning the test model. Accuracy is 0.7929; macro F1 is 0.3981 across 640,783 valid rolling pixel forecasts. Per-class metrics:

| Proxy class | Precision | Recall | F1 | CSI/IoU |
|---|---:|---:|---:|---:|
| Normal | 0.9004 | 0.8856 | 0.8929 | 0.8066 |
| Elevated | 0.1969 | 0.3101 | 0.2409 | 0.1369 |
| Strong | 0.1126 | 0.0413 | 0.0604 | 0.0312 |

CSI is derived directly from each class's saved confusion counts: TP/(TP+FP+FN). Strong-score Brier is 0.06506. PR AUC is not reported in this saved final evaluation and was not invented. Scores are not approved calibrated probabilities. The previously inspected 2025 holdout is not an untouched test.

Horizon-one regional assessment has 328 origins, 78 truth-qualified cases and 250 unknown-truth cases; 161 origins fail input requirements. Mean input chlorophyll coverage is 7.26%. The model issued zero qualified warnings across 12 actual proxy-risk days, including nine abstained risk days. Zero false warnings does not establish warning skill. Rolling forecasts overlap; pixel counts are not independent bloom events. No metric in this section measures verified HAB occurrence.

The saved 2024 common-cohort comparison also shows persistence macro F1 0.5355 versus 0.5180 for the frozen model. That conditional cohort differs from the all-target results above. See [existing results](improvement_results.md) and [roadmap evaluation](roadmap_implementation.md) for full denominators and limitations.

## Data search outcome and external blocker

The [source-search report](hab_source_search.md) and [machine-readable audit](results/hab_submission/source_audit.json) document actual downloads, checksums, source records, dates, access conditions and limitations. In brief, the open Indian event workbook contains only three Kerala entries in the input period, at month/locality resolution; HAEDAT's India export has no 2021–2025 event dates. A relevant field cruise provides point reference stations around a single episode, not a representative daily zone monitoring catalogue.

**Needed to train:** original station/zone sampling logs with exact days, defensible locations/uncertainty, episode identities, reviewed species/impact evidence, and monitored absence meeting the declared target definition. KUFOS Thikkodi monitoring, CMLRE/INCOIS field records and the referenced NCCR/NIO field studies are specific leads. Requests or permissions must be handled by the user; no outside messages were sent. If only station-level negatives are obtained, explicitly change the target to monitored-site occurrence or obtain zone-representative evidence; do not label the entire zone absent.

## Reproduction commands

Preservation check and baseline inference:

```bash
.venv/bin/python -m experiments.finalize verify
.venv/bin/python prediction/predict.py --input-end-date 2024-02-15 --historical-demo
make check
```

With the server running, repeat real HTTP checks into a **fresh** evidence path:

```bash
.venv/bin/python tests/submission_smoke.py \
  --url http://127.0.0.1:8000 --output /tmp/new-submission-smoke.json
```

Only after obtaining and adjudicating real observations, complete the spec template (14-day lookback, chosen zones and chronological train/validation/calibration/test periods) and run:

```bash
.venv/bin/python -m preprocessing.hab_sequences \
  --spec /absolute/path/reviewed_hab_spec.json \
  --observations /absolute/path/verified_hab_observations.csv \
  --output-dir outputs/hab_occurrence/dataset_01
.venv/bin/python -m training.train_hab \
  --dataset-dir outputs/hab_occurrence/dataset_01 \
  --output-dir outputs/hab_occurrence/run_01 --epochs 30 --batch-size 2
.venv/bin/python -m prediction.hab_evaluation calibrate --run-dir outputs/hab_occurrence/run_01
.venv/bin/python -m prediction.hab_evaluation evaluate --run-dir outputs/hab_occurrence/run_01
HAB_OCCURRENCE_RUN_DIR=outputs/hab_occurrence/run_01 \
  .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Do not run against blank templates or create negative rows from missing reports. Existing dataset/run directories cannot be overwritten. The test guard prohibits repeat evaluation within a run, even after failure; preserve the failed evidence and declare a separate experiment for recovery. Detailed contracts: [HAB implementation](hab_implementation.md).

## Original-proposal differences

The accepted data period remains 2021–2025. The working baseline uses five input days and chlorophyll anomaly targets. The separate HAB code supports the preferred 14-day input and five future days but has not been trained on verified observations. Its output is **zone occurrence**, not pixel bloom extent or toxicity. Pixel HAB masks require suitable spatial ground truth. FastAPI/plain JavaScript replace the proposed Streamlit/Folium interface. Baselines include logistic regression, persistence and existing seasonal comparisons; no NDWI HAB baseline result is claimed. Daily live acquisition and operational warnings are not enabled.

## Presentation explanation

- **Problem:** forecast coastal conditions associated with blooms while distinguishing high chlorophyll from verified harmful occurrence.
- **Dataset:** seven daily environmental fields on a common grid; masks explicitly identify missing satellite observations. Missing HAB reports remain unknown.
- **Architecture:** spatial pooling → ConvLSTM 8 filters → normalization → ConvLSTM 4 filters → normalization → upsampling. The existing model produces five anomaly maps; the HAB extension pools features by zone and produces five daily sigmoid scores.
- **Training:** fit normalization on training data only; keep events and temporal windows within splits; mask unknown targets; select by validation loss; calibrate and choose thresholds on a separate calibration split; freeze before test assessment.
- **Results:** demonstrate actual historical chlorophyll inference, discuss the rare-class and regional warning failures honestly, and present the HAB workflow as implemented infrastructure pending verified labels—not a validated HAB predictor.
- **Contribution:** reproducible multi-source pipeline, explicit uncertainty/missing-data handling, preserved experiments and an evidence-driven path from anomaly forecasting to verified occurrence prediction.

Suggested closing sentence: “The prototype demonstrates five-day environmental anomaly forecasting. We implemented a separate verified-HAB occurrence pipeline, but a source audit showed that independent daily field labels and monitored negatives remain the requirement for scientific HAB validation.”

## Verification evidence

108 Python tests passed locally, including 14-day/five-output model software checks, label/leakage contracts, calibration, event timing, model reload, API errors and abstention. JavaScript regression checks passed. **24 localhost HTTP assertions** verified actual archived inference, five maps, observation view, CSV contents, missing inputs and stale-mode restrictions. Browser checks verified eligible-date selection, day switching, observation switching/zoom and the disabled unconfigured HAB page. Existing frozen artifacts were checked separately for preservation.

Evidence: [HTTP checks](results/hab_submission/http_smoke.json), [source audit](results/hab_submission/source_audit.json), [preservation/check manifest](results/hab_submission/verification.json), [day-five screenshot](results/hab_submission/dashboard_day5.png), [observation view](results/hab_submission/dashboard_observations.png), [HAB availability](results/hab_submission/dashboard_hab_unconfigured.png).
