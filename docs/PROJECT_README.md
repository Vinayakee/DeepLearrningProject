# ConvLSTM-Based Satellite Bloom-Event Forecasting for the Kerala Coast

**Controlled input experiment run_03:** selected V3 on 2024 only; mean 2025 observed-target F1 0.3798, IoU/CSI 0.2345. Run_03 improves mean F1/IoU over run_02 and beats matched persistence at every horizon; report as additional improved-input evidence. See [ablation and causal feature definitions](docs/bloom_inputs_v3.md) and [matched pixel/block results](docs/results/bloom_events_v3/evaluation.md). The frozen three-class chlorophyll baseline remains the primary quantitative result. All three satellite-event runs are selectable on the updated server at port **8016**.

**Preserved runs:** the two earlier separately trained **satellite bloom-event forecast** experiments are available at `/dashboard/bloom.html`. Strict run_01 is preserved: 23 training components, zero positive validation/test targets, recall undefined. Relaxed run_02 uses training-defined P75 daily patches ≥2 pixels, selected on train/2024 label support; it has evaluable positives but **persistence beats matched F1/IoU at all five horizons**. Model test F1 is 0.324–0.341 on all observed targets. 2025 was previously inspected. These are satellite proxies, not confirmed harmful occurrence. See [run_02 definitions, metrics and limitations](docs/satellite_bloom_events_v2.md) and [strict run_01](docs/satellite_bloom_events.md).

Launch all three pages on a free local port:

```bash
BLOOM_EVENT_RUN_DIR=outputs/bloom_events/run_02 MPLCONFIGDIR=/tmp/satellite-event-mpl \
  .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8016
```

| Workflow | Actually trained? | Page |
|---|---|---|
| Satellite bloom-event forecast | Separate strict run_01 and relaxed run_02; select either run | `/dashboard/bloom.html` |
| Original chlorophyll-anomaly baseline | Existing frozen three-class model, preserved | `/dashboard/` |
| Verified-HAB occurrence | No verified checkpoint; remains `not_configured` | `/dashboard/hab.html` |

**Submission entry point:** [launch commands, verified results and presentation notes](docs/SUBMISSION_HANDOFF.md). The [verified-HAB source search](docs/hab_source_search.md) documents actual public downloads and why a real HAB model has not yet been trained. The working chlorophyll model and frozen evaluation are preserved.

**Verified HAB occurrence implementation:** a separate zone-level training, calibration, evaluation and API/dashboard workflow is documented in [HAB implementation](docs/hab_implementation.md). It learns from adjudicated HAB observations rather than chlorophyll thresholds. Verified HAB records are still required before this new model can be trained; the existing chlorophyll experiment below is preserved. Open `/dashboard/hab.html` for the new mode.

A student deep-learning project forecasting **chlorophyll anomaly risk** along the Kerala coast. Seven environmental fields, their observation flags and month sine/cosine features from the previous five days produce three risk-class scores for each of the next five days. These are proxy labels, not confirmed harmful species, toxicity or field-verified HAB events.

The existing data acquisition, alignment, two-layer ConvLSTM approach and dashboard are retained. Corrected training and forecasts use separate artifacts; original labels, checkpoints and outputs remain available as historical references.

The [October roadmap implementation](docs/roadmap_implementation.md) adds a verified native Sentinel pilot, geographic mask comparison, seasonal/persistence comparisons with temporal block uncertainty, observation/coverage maps, CSV exports, acquisition retry and orphan-lock recovery. The bounded pilot remains experimental; the active checkpoint and saved 2025 evaluation are preserved.

The latest [post-freeze improvements](docs/post_freeze_improvements.md) add STRONG diagnostics, a controlled sampling experiment, 2024 temperature-calibration assessment and quality-controlled real-data/live-update interfaces. The baseline and original frozen 2025 evaluation remain preserved. [Required real-data formats](docs/real_data_interfaces.md) document what remains dependent on unavailable observations.

The [runtime integration guide](docs/runtime_workflows.md) covers configurable acquisition/polling, guarded QC Sentinel research inference, verified-field matching, supplied-data independent evaluation and disabled-by-default SMTP/SMS providers. Health/API metadata explicitly distinguishes decision artifacts from tested, active and approved probability calibration. These integrations do not claim available fresh observations or operational warning reliability.

The [latest integration verification](docs/integration_completion.md) records 82 passing tests, actual historical/staged/API checks and the remaining external-data limitations.

## Install and data prerequisites

Tested on macOS ARM64 with Python 3.13 and CPU TensorFlow. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
```

The dashboard needs no Node/npm build. `requirements.txt` is sufficient to run the application; `requirements-dev.txt` adds the HTTP test client. Download dependencies are separate in `requirements-download.txt`. `requirements-lock.txt` records all packages from the tested clean application/test environment; `requirements-download-lock.txt` also records optional acquisition packages. Install a lock file directly for exact versions on the tested Python/platform.

**Data and trained weights are ignored by Git.** A code-only clone does not contain the dataset. Use the existing aligned data, transfer them from the project owner, or acquire and preprocess raw products as described in [data preparation](docs/data_preparation.md). Never substitute generated/random arrays for scientific results. The aligned inputs must contain:

- `data/processed/dates_2021_2025.txt`: 1,826 consecutive dates, 2021-01-01 through 2025-12-31.
- `latitude_kerala_2021_2025.npy` and `longitude_kerala_2021_2025.npy`: the existing 120 × 84 grid.
- Seven `<name>_2021_2025.npy` arrays and corresponding `<name>_mask_2021_2025.npy` arrays, each `(1826,120,84)`. Names: `chlorophyll`, `sst`, `wind_u`, `wind_v`, `rainfall`, `current_u`, `current_v`.

Masks contain 0/1; absent values remain NaN. January/Sentinel download diagnostics are manual acquisition tools, not the automated regression suite.

## Run the existing completed system

Use the already fitted artifacts; do not rerun completed experiments or regenerate labels over a frozen run.

```bash
source .venv/bin/activate
python -m experiments.finalize verify
python prediction/predict.py --input-end-date 2024-02-15 --historical-demo
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/dashboard/** and enable **Historical demonstration**. All five maps and scores come from actual model inference. Missing required inputs return **Insufficient data for reliable prediction** without executing the model. Low-coverage exploration cannot issue reliable warning labels.

“Find covered historical date” selects a real 2024 input window meeting the frozen coverage requirements. Source, last successful update and live-availability status are shown separately. `/api/system`, `/api/covered-dates` and `/api/live/prediction` expose those engineering states without rerunning the frozen evaluation. No live acquisition has succeeded yet; production notification delivery remains disabled.

The dataset ends in December 2025. It supplies no current coastal warnings. Regional warning skill is insufficient, so operational warning delivery remains disabled even if input dates become fresh.

```bash
python -m unittest discover -s tests -v
python -m pip check
python tests/http_smoke.py
```

The HTTP smoke command requires the server above to be running. See [training/reproduction commands](docs/improvement_commands.md) for the completed controlled experiment sequence and safe new experiment names. Existing experiment directories and the original baseline checkpoint cannot be overwritten. Final selection, freeze and test-start guards prevent retuning or repeating the final evaluation.

## Methodology

The retained raw-data preprocessors align NASA Aqua MODIS daily chlorophyll, Copernicus SST/currents and ERA5 wind/rainfall to the same dates/grid. SST is Celsius, rainfall mm/day, currents/wind m/s and chlorophyll mg/m³.

Training is 2021–2023, validation 2024, testing 2025. Monthly median climatology, marine support, observation minimums, P75/P90 relative-anomaly thresholds, normalization and class weights are fitted from training years only. Each five-day target follows its input window, and the complete ten-day window stays inside its split. Windows with entirely absent targets are excluded.

Training support uses marine-product availability and at least 20 training chlorophyll observations per cell. A cell needs at least three observations in its training calendar month for that month's labels. Missing labels stay 255 and become all-zero targets; the loss and accuracy exclude them. See [label details](docs/label_preprocessing.md).

One `EnvironmentalPipeline` transforms training, validation, testing and inference. Chlorophyll/rainfall use log1p; each field uses training median and interquartile range, then clips standardized values to ±8. Missing inputs become zero in standardized space **with a separate availability flag**. The final model receives seven fields plus seven flags and two cyclical month features. The original 14-channel checkpoint remains preserved. Calendar encoding is shared across training, validation and inference.

The architecture remains small and explainable:

```text
5 × 120 × 84 × 16 inputs
  → spatial average pooling to 60 × 42
  → ConvLSTM (8 filters, all input steps) + LayerNormalization
  → ConvLSTM (4 filters, final temporal representation) + LayerNormalization
  → bilinear upsampling to 120 × 84
  → 1×1 convolution with 5 × 3 logits
  → reshape to 5 × 120 × 84 × 3 and class softmax
```

Every future day uses the final representation of all five input days. Adam starts at 0.001 with gradient clipping; class weights are normalized inverse-frequency power 0.75: [0.7973403, 2.6660632, 3.6133980], with expected training-pixel loss mass matched to the square-root baseline. Weighted cross-entropy is divided once by valid target count, with no second sample-weight mask. Accuracy accumulates valid pixels globally.

## Evaluation and warnings

The baseline was frozen before development. 2024-only diagnostics showed both learned suppression of elevated scores and additional suppression by the old strong override. Class weights, decision boundaries and seasonal inputs were tested separately before combining retained changes. The final 8,787-parameter model keeps ConvLSTM filters 8/4; the two calendar inputs add only their necessary input weights.

The final frozen decision is `argmax(p × [1, 1.2840254166877414, 0.7788007830714049])`. This adjusts class boundaries, not the raw probabilities. It was selected on 2024 macro F1 while monitoring all classes and false warnings. It is not probability calibration. Exact configurations, corrected-label/preprocessing hashes, histories and checkpoints are saved separately from baseline.

**2025 was already inspected in earlier work. It cannot be called untouched.** This improvement round excluded it from development; the final frozen configuration was evaluated once on that previously inspected holdout, without subsequent tuning.

[Final results](docs/improvement_results.md) provide baseline/final accuracy, macro F1, all class precision/recall/F1/support, confusion matrices, horizons, Brier score, coverage/abstentions, regional warning counts and false-alarm analysis. Archived baseline reports also contain majority and persistence references. Rolling pixel forecasts overlap and are not independent events. Labels indicate chlorophyll anomalies, not verified toxic blooms.

Regional warnings retain training P75/P90 thresholds (13.720735% and 24.172538%) for predicted strong-pixel fractions on the training-supported domain. Observed target days below 5% marine coverage are unknown. Primary warning-day evaluation uses horizon one; unavailable-input high-risk days count as misses. A common available-input cohort separates coverage effects from model effects.

On the frozen 2025 evaluation, macro F1 improved from 0.3833 to 0.3981 and elevated F1 from zero to 0.2409, but strong F1 fell from 0.2574 to 0.0604. Regional strong recall was zero in both 2024 and 2025; the final model issued no qualified regional warnings in 2025. Reducing false warnings therefore does not prove a useful regional detector. Operational warning delivery is disabled; prototype alerts are available only for research demonstration. Missing input observations are explicitly flagged, never relabeled normal. Forecasts recheck freshness on every request; unsupported cells are grey.

No SMS/push provider, automatic data refresh or paid service is configured. Independent verified observations and separate regional validation are required before operational use. See [diagnosis and controls](docs/improvement_methodology.md).

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Existing API status and dashboard link |
| `GET /api/health` | Training/artifact availability and latest input date |
| `GET /api/prediction` | Current cached forecast; stale data blocked |
| `GET /api/prediction?demo=true` | Explicit historical forecast with freshness/coverage limitations |
| `POST /api/predict` | Generate a forecast from real aligned inputs |
| `GET /api/alerts?demo=true` | Historical dashboard alert entries clearly marked non-operational |
| `GET /api/evaluation` | Actual generated validation/test metrics |
| `/dashboard/`, `/forecasts/<id>/day_<n>.png` | Existing frontend and dated forecast maps |

POST body: `{"input_end_date":"2024-02-15","historical_demo":true}`. Missing required observations return a structured `insufficient_data` result with no maps or predictions. A date requires five available input days; invalid input returns 422, missing/inconsistent artifacts 503, concurrent predictions 409. The app serves frontend/assets itself, so no second frontend server or hard-coded API hostname is needed.

For an additional real HTTP integration check while the API is running:

```bash
python tests/http_smoke.py
```

## Structure and artifacts

```text
backend/                  FastAPI and existing dashboard serving
frontend/                 Existing HTML/CSS/JavaScript dashboard
preprocessing/            Acquisition/alignment, corrected labels, quality report, shared pipeline
models/                   Compact ConvLSTM source and preserved/new checkpoints
training/                 Shared generator, masked loss/metric, trainer
prediction/               Evaluation, baseline metrics, inference/maps/warnings
config.py                 Shared paths, channels, dates and freshness configuration
experiments/              Baseline freeze, controlled development, final freeze/test and reports
tests/                    Offline regression and HTTP integration tests
docs/                     Methodology, data preparation and verified completion report
data/processed/training_only_labels/  Corrected labels and provenance
data/sequences/corrected/             Index, training normalization and manifest
outputs/corrected/                    Training history, evaluation/calibration and plots
outputs/data_quality/                 Coverage Markdown/JSON/CSV
outputs/forecasts/<id>/               Arrays, dated five-day maps and bundle metadata
```

The active checkpoint is `outputs/improvement/final/model.keras`, selected by `outputs/improvement/final/deployment.json`. The original `models/corrected_convlstm.keras` and complete `outputs/improvement/baseline` snapshot are preserved. Final preprocessing lives in `outputs/improvement/final_preprocessing`; labels are the unchanged frozen corrected labels. Artifacts record hashes and source file identities so mismatched labels, normalization, model or forecasts fail visibly. Dataset copies should be followed by regeneration/retraining; preserve timestamps if transferring a complete already-fitted artifact set.

## Configuration and limitations

`HAB_USE_BASELINE=1` selects the preserved original run. `HAB_RUN_DIR`, `HAB_SEQUENCE_DIR` and `HAB_LABEL_DIR` override matching artifact paths. `HAB_MODEL_PATH` overrides the checkpoint (relative paths resolve from the repository root; it must match the completed training record). `HAB_MAX_DATA_AGE_DAYS` defaults to 3, a prototype safeguard. Acquisition alone requires NASA `.netrc`, Copernicus Marine login, CDS `.cdsapirc`, or ignored `.env` containing `CDSE_CLIENT_ID`/`CDSE_CLIENT_SECRET` for Sentinel tools. The normal dashboard/training path needs none of these credentials.

MODIS has extensive cloud/missing coverage. Sentinel-3 coverage is reported but its exports lack retrieval science/quality flags, so it remains excluded from modelling. Marine support is an availability proxy rather than an authoritative coastline. The model does not confirm HAB species/toxicity, ingest new data automatically, offer real-time monitoring or send phone notifications. Forecast skill must be assessed against persistence, particularly for rare strong blooms; accuracy alone is insufficient.

Practical future improvements are quality-controlled satellite fusion, verified field HAB labels, better coastal support, event-based evaluation, probability calibration and a separately validated fresh-data ingestion/notification workflow. See [final improvement results](docs/improvement_results.md) for actual runs and remaining limitations. The earlier [corrected-pipeline completion report](docs/project_completion.md) is retained as baseline history.
