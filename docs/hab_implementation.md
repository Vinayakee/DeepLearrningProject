# Verified HAB occurrence workflow

The new workflow predicts **documented HAB occurrence within a defined coastal zone on each of five future days**. It is separate from the existing chlorophyll anomaly model. Environmental inputs alone do not establish harmful species or toxicity. The [24 October source search](hab_source_search.md) obtained real public event reports but no suitable adjudicated daily training catalogue with monitored absences, so no real HAB model was fitted or promoted.

## 1. Define observations and zones

Copy `docs/examples/hab_spec.template.json` into your own dataset configuration. Replace every definition/zone placeholder, supply disjoint chronological `train`, `validation`, `calibration` and `test` periods, and define nonoverlapping `[west,south,east,north]` rectangular analysis zones. Bounds must contain marine cells on the existing grid. The template bounds are an example study box, not a validated coastal zone. The template now prefers **14 input days**; the implementation supports configurable 2–60 input days and five output days. Choose the lookback before testing.

The event definition must state what evidence establishes a HAB: for example, adjudicated harmful-species observations against an applicable locally reviewed threshold or verified harmful bloom impacts. Occurrence does not imply toxin presence or severity. The negative definition must explain the monitoring needed to establish absence across a zone. Review these definitions with the evidence provider; software cannot authenticate a field claim from its CSV alone.

Use the columns in `docs/examples/hab_observations.template.csv`. This file intentionally contains no invented records. Store actual observations outside tracked docs, for example `data/hab_observations.csv` (ignored by Git).

| Field | Contract |
|---|---|
| `record_id` | Unique observation identity |
| `event_id` | Required for positive observations; the same episode across zones/days must use the same ID |
| `zone_id` | Matches a configured analysis zone |
| `observed_at_utc`, `available_at_utc` | Explicit UTC, publication follows observation; no future publication |
| `latitude`, `longitude` | Actual observation/reference location within its zone |
| `confirmed_hab` | Adjudicated `true` or `false`, never inferred from a missing report |
| `observation_scope` | `point` or `complete_zone`; negative zone labels require `complete_zone` |
| `definition_id` | Exact reviewed event-definition ID in the spec |
| `species` | Identification when known; may be blank if supported by another accepted evidence type |
| `verification_method`, `source_reference`, `verified_by` | Required auditable evidence and adjudicator |

A confirmed point occurrence establishes at least one documented occurrence in its zone; point absence cannot establish zone absence. A zone-day with no observation is **unknown (255)**. Conflicting statuses or event IDs require explicit adjudication. Each CSV record applies to its observed date only; intervals are not silently stretched into daily labels. Event IDs crossing splits fail preparation. Supply split boundaries around complete events, leaving a gap if needed. Records in gaps are rejected; omit them from this experiment's input catalogue while preserving the source catalogue separately.

This first version accepts explicitly adjudicated zone assignments, not uncertain-location records or arbitrary bloom polygons. Resolve uncertainty upstream and preserve source evidence; do not assign a precise zone when the evidence cannot support it. Rectangles are analysis areas, not predicted bloom extent. Detailed spatial masks require a separate pixel-label extension.

## 2. Prepare the dataset

Run from the repository root with the existing environment:

```bash
.venv/bin/python -m preprocessing.hab_sequences \
  --spec /absolute/path/hab_spec.json \
  --observations /absolute/path/hab_observations.csv \
  --output-dir outputs/hab_occurrence/dataset_01
```

The command uses the actual aligned arrays and coordinates under `data/processed` by default (`--processed-dir` can select another directory using the same array filenames). It derives marine support and fits environmental median/IQR statistics from the declared training period only. It uses seven fields, seven observation flags and month sine/cosine; invalid inputs are filled in standardized space with flags retained. It never loads the old chlorophyll target labels or their normalization.

Outputs include spec/records, labels and event IDs, marine support, sequence index, preprocessing, file hashes and a split audit with per-zone label counts. Every complete input-plus-five-target window stays inside one split. Empty-target windows are omitted; unknown targets remain masked. All four splits need usable windows with both observed classes before training; each zone separately needs both classes in usable training targets. Counts alone do not establish sufficient statistical power: review event counts, spatial sampling and representative monitored negatives before interpreting results.

Fitted artifact files and coordinate/date files use SHA-256. Aligned environmental arrays use file size and modification timestamp identity, as in the existing project; this is a change detector, not cryptographic content integrity against same-size/same-timestamp edits. Do not mutate an experiment's input archive.

## 3. Train a separate candidate

```bash
.venv/bin/python -m training.train_hab \
  --dataset-dir outputs/hab_occurrence/dataset_01 \
  --output-dir outputs/hab_occurrence/run_01 \
  --epochs 30 --batch-size 2
```

The model retains ConvLSTM filters 8/4, spatial pooling, normalization and bilinear restoration, then pools spatial features within each supported zone. A shared dense sigmoid head produces `(batch,5,zones)` scores. It does not estimate toxic species, concentrations or bloom boundaries. It starts a new HAB model; it does not relabel or overwrite the old checkpoint.

Masked binary cross-entropy excludes unknown targets. Default positive weight is the square root of the training-window negative/positive ratio. Optional `--event-balanced` instead samples windows with approximately equal event mass and equal total negative-only-window mass, using unit loss weights to avoid double correction. Windows containing multiple events share sampling mass; this is not independent event replication. Model selection uses validation loss only. Normalization and logistic-baseline coefficients use training data only. A training-prevalence reference is retained in `baselines.json`.

Training also fits one logistic-regression environmental baseline per horizon; single-class horizons explicitly fall back to training prevalence for that horizon. Data/history, model identity, exact dependency versions, seed and source/lock-file snapshots are recorded. Existing dataset/run directories cannot be overwritten.

## 4. Calibrate, freeze and evaluate

```bash
.venv/bin/python -m prediction.hab_evaluation calibrate \
  --run-dir outputs/hab_occurrence/run_01
.venv/bin/python -m prediction.hab_evaluation evaluate \
  --run-dir outputs/hab_occurrence/run_01
```

Calibration uses the separate calibration split with both observed classes and sufficient input coverage. It fits a pooled Platt mapping separately for the ConvLSTM and logistic baseline. Nonpositive ConvLSTM discrimination is rejected; a nonpositive baseline calibration instead retains raw baseline scores and explicitly records that calibration was not applied. The High threshold maximizes calibration-split F1 (precision and higher threshold break ties); Low is the highest threshold attaining at least 90% calibration-split recall, capped at High. Thresholds may coincide. LOW/MODERATE/HIGH describe occurrence estimates, not biological severity. Objectives are development choices, not validated warning guarantees.

Calibration and thresholds are fitted on the same declared fitting split. Their reliability is assessed only afterward on test observations. Verified observation sampling can bias calibration; it does not automatically provide population-wide probabilities. The run is frozen before test collection. Evaluation is guarded by exclusive `test_started.json`: it cannot be repeated or retuned within the same run, including after an interrupted test. Preserve failed-run evidence and use a separate declared experiment for recovery.

Reports include observed-target confusion counts, F1, CSI, PR AUC, ROC AUC, Brier score, reliability bins, each horizon and zone, coverage abstentions, positive-target recall including abstentions, event recall and nominal lead before the first recorded event day. Lead-time references now come from the **full verified event catalogue**, not only evaluated targets. Advance warning, same-day detection, ongoing-event detection and unavailable timing references are distinguished; advance-warning recall is reported separately from any event detection. False alarms count only verified-negative forecast zone-days; unknown days cannot establish false alarms. Repeated horizons/origins are correlated. Events absent from all usable split-safe windows are outside the assessed cohort; the report explicitly counts them separately from evaluated events.

Persistence uses the latest available verified observation within the input window and abstains otherwise; observations published after issuance are excluded. Its standalone report uses its own observation availability, without incorrectly requiring satellite coverage. Model/logistic results share a coverage-qualified target cohort, and a separate common persistence-supported cohort is reported. Fixed 14-calendar-day origin-block bootstrap intervals retain all zones/horizons together; they are conditional on observation coverage and may underrepresent dependence of long events. Operational product publication latency remains unverified for the historical environmental archive, so nominal lead times do not establish operational warning lead.

## 5. Configure the API and dashboard

```bash
HAB_OCCURRENCE_RUN_DIR=outputs/hab_occurrence/run_01 \
  .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/dashboard/hab.html` or follow the link from the existing dashboard. Without a configured run the page explains that verified observations and model fitting are required; it does not substitute chlorophyll predictions.

| Endpoint | Purpose |
|---|---|
| `GET /api/hab/status` | Lightweight run availability metadata (not a full integrity audit) |
| `POST /api/hab/predict` | Actual zone inference after frozen-artifact, coverage and freshness checks |
| `GET /api/hab/evaluation` | Saved held-out report after artifact checks; no repeat evaluation |

POST uses `{"input_end_date":"YYYY-MM-DD","historical_demo":true}` for an archived date. The dashboard displays each future day, interactive zone rectangles, experimental calibrated estimates, occurrence categories and zone input coverage, plus five-day CSV export. A zone requires at least 5% mean chlorophyll coverage, two observed chlorophyll days and one observed day for every other field in its input window. These inherited coverage thresholds remain prototype safeguards. If every zone fails, inference is not executed; failed zones never receive displayed scores. Future-dated input is rejected, and stale input requires historical demonstration mode.

No model or label catalogue is bundled. No new live-source adapter, production warning approval, automated acquisition or notification delivery is enabled. The existing live path still belongs to the frozen chlorophyll experiment. HAB inference currently uses the separate historical archive workflow. Any fresh operational HAB integration needs verified source timing, independent warning validation and a separate deployment review.

## Verification

`tests/test_hab_occurrence.py` uses explicitly synthetic temporary fixtures only to test software contracts. It does not produce scientific model results. Run the complete existing suite plus the new tests and JavaScript checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check frontend/hab.js
node tests/hab_dashboard.mjs
```

Implementation verification: 106 Python tests passed with no skips in the local environment, including tiny synthetic-model training/reload, deterministic calibration contracts, held-out evaluation guards, API inference, tamper rejection and missing-input non-execution. JavaScript syntax, the new HAB dashboard regression and existing request-order/interaction regressions passed. Dependency and whitespace checks passed. The new dashboard/assets/status routes returned HTTP 200 and the unconfigured state was visually inspected in a browser. The preserved scientific baseline passed its integrity audit without new test-set inference. No real HAB model was trained; synthetic test scores are not scientific performance results.
