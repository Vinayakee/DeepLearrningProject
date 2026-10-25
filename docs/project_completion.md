# Verified project completion — 12 September 2026

This document records the earlier baseline completion. The subsequent improvement experiments, active model, 38-test suite and warning limitations are documented in [improvement results](improvement_results.md). Use the [current commands](improvement_commands.md); the historical training commands below cannot overwrite the now-frozen checkpoints.

The existing project now runs from real aligned data through corrected labels, full-data ConvLSTM training, held-out evaluation, inference, FastAPI, the existing dashboard and dashboard warning states. This completes a reproducible research demonstration; it does not establish operational HAB forecasting reliability.

## What changed and what was preserved

Training-only label fitting removes held-out leakage while preserving the existing relative-anomaly formula and three classes. Training, testing and inference share normalization and seven availability flags. The scalar masked loss is applied once, accuracy accumulates actual valid pixels, and future targets follow all five inputs. The forecast head now uses the final temporal representation for every future day. Hard-coded evaluation values/maps/thresholds are replaced by generated artifacts and a single validated decision rule. APIs perform inference, check artifact consistency, reject invalid inputs, recheck data age and distinguish historical demonstrations. The existing dashboard receives actual values, coverage, dates and warnings.

Original raw/aligned data, original labels/climatology, two original checkpoints, old forecasts/maps and original sequence artifacts remain preserved. Twenty original generated artifacts were checked against the pre-change SHA-256 snapshot; all were unchanged. Existing frontend layout, cards, environmental descriptions and five-day interaction remain. Original acquisition/regridding scripts are retained; the unsafe duplicate trainer is clearly marked as a reference and no longer executes accidentally.

## Phase validation

| Phase | Implemented and actually checked |
|---|---|
| Labels/data | Train-only monthly medians/P75/P90, fixed marine support, monthly minimums, valid/missing labels; nine preprocessing regression tests and real coverage report |
| Shared pipeline/loss | Real seven-field batches, 14 input channels, matching transforms, split-safe future windows; masked-loss padding/empty-target tests and real generator consistency |
| Training sanity | Eight real mini-batches through a small sanity network showed loss 0.3836 → 0.1587; separate two-epoch ConvLSTM smoke run completed and reloaded |
| ConvLSTM | Output `(batch,5,120,84,3)`, probabilities sum to one, each future head responds to the final input; save/reload regression passed |
| Full training | All 1,012 training sequences and 325 validation sequences, best checkpoint retained, actual checkpoint reloaded and real-batch inference passed |
| Evaluation | 2024-only threshold selection, full validation and 328 held-out 2025 sequences, baseline/class/horizon/probability evaluation generated |
| API/warnings | HTTP validation, missing/inconsistent artifacts, concurrency, age boundaries, masked summaries, normal/elevated/strong rules, stale cache blocking |
| Real integration | Real checkpoint → historical inference → risk arrays → all five PNG routes → cached API → dashboard alert records |
| Browser | Actual historical elevated and strong warning displays, day-five date/map/metrics update, environmental values and missing-observation text, stale blocking and full-size map link |
| Setup/cleanup | Clean temporary Python 3.13 environment installed core/dev and optional acquisition requirements; pip check passed, source syntax and git whitespace checks passed; secret files remain ignored |

## Dataset and preprocessing

1,826 daily inputs cover 2021–2025 on the existing 120 × 84 grid. Marine availability support has 5,691 cells; training chlorophyll observation minimums retain 5,683 stable support cells. Monthly support varies with the three-observation requirement. Total minimum is 20 training observations. Labels 255 remain absent from scoring and never become normal targets.

| Split | Dates | Sequences | Normal labels | Elevated labels | Strong labels |
|---|---|---:|---:|---:|---:|
| train | 2021-01-01–2023-12-31 | 1012 | 423344 | 84669 | 56450 |
| validation | 2024-01-01–2024-12-31 | 325 | 148271 | 10783 | 7221 |
| test | 2025-01-01–2025-12-31 | 328 | 115745 | 10831 | 6972 |

Normalization uses training median/IQR per variable, log1p for chlorophyll/rainfall and standardized clipping at ±8. Missing values become zero plus a separate 0 availability flag. Marine filtering applies to every input field and forecast summaries exclude unsupported seasonal cells.

MODIS mean coverage is 4.88% of the whole grid and 8.58% of marine support, with 962 empty days. Sentinel coverage is broader (37.18% of marine support), but exports lack retrieval science/quality flags and independent georeferencing metadata; it remains excluded from the model. Reports retain source disagreement and boundary diagnostics without inventing observations.

## Architecture and actual training

The model has **8,211 parameters**: 5 × 120 × 84 × 14 inputs → average pooling to 60 × 42 → ConvLSTM8 (all time steps) + LayerNormalization → ConvLSTM4 (final representation) + LayerNormalization → bilinear upsampling → 1×1 convolution with 15 logits → reshape into five independent three-class softmax heads. The compact filters/pooling make full training practical on the available CPU. Every future head receives the entire five-day history, unlike the old input-aligned sequence head.

Actual full run: seed 42, Adam initial LR 0.001, batch size 2, four intra-op CPU threads, gradient clipnorm 1, square-root inverse-frequency weights [1.0, 2.236065336545842, 2.738512523515752]. Requested 30 epochs; early stopping completed **14**, best validation epoch **10**, elapsed **10.95 minutes**. ReduceLROnPlateau and best-val-loss checkpointing were active.

Training loss 1.4043 → 1.2501; best validation loss 1.0012. Checkpoint: `models/corrected_convlstm.keras`, SHA-256 `0dff8dd0ded97db428cb9694a03a1caf20aa2605322d32ff3dd90992fbea678d`. The checkpoint was loaded successfully in both the existing and clean environments.

## Actual held-out results

The strong-class threshold is **0.225**, selected exclusively on 2024 validation (strong binary F1 0.2523). 2025 contains **640,783 valid pixel forecasts** over rolling origins/horizons.

| 2025 method | Accuracy | Macro F1 | Strong precision | Strong recall | Strong F1 |
|---|---:|---:|---:|---:|---:|
| model | 0.7876 | 0.3833 | 0.1714 | 0.5170 | 0.2574 |
| argmax | 0.8584 | 0.3191 | 0.1236 | 0.0160 | 0.0283 |
| persistence | 0.8380 | 0.3839 | 0.2369 | 0.0916 | 0.1321 |
| majority | 0.8637 | 0.3089 | 0.0000 | 0.0000 | 0.0000 |

Pixel-forecast metrics over rolling windows; overlapping target dates are scored at distinct origins/horizons, not independent observations. Probabilities are model scores, not guaranteed calibrated.

Strong-probability Brier score: 0.0557

Legacy results used a different label definition/support and preprocessing. They are not a controlled comparison with this corrected test set.

### Test class performance (shared threshold decision)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.9055 | 0.8801 | 0.8926 | 553413 |
| Elevated risk | 0.0000 | 0.0000 | 0.0000 | 53264 |
| Strong bloom risk | 0.1714 | 0.5170 | 0.2574 | 34106 |

Confusion matrix; rows actual, columns predicted normal/elevated/strong:

```text
487041  10  66362
34374  0  18890
16473  0  17633
```

**Model limitation:** the elevated class has zero recall/F1. Strong recall improves over persistence, but precision is low and overall accuracy/macro F1 do not beat persistence. The model is a working prototype with limited rare-class skill, not a validated warning product. No parameters were tuned against test performance after observing these results.

## API, dashboard, alerts and actual end-to-end example

FastAPI serves the existing dashboard and dated forecast images from one origin. Every important endpoint passed real HTTP testing: root, health, dashboard HTML/CSS/JS, invalid POST (422), real inference POST (200), all five maps (200), stale cached forecast, explicit-demo cached forecast, alerts and evaluation. Tests also verify artifact failures (503), concurrent prediction rejection (409) and date/coverage/missing-data behavior.

Real latest-data example: input **2025-12-27–2025-12-31** → forecasts **2026-01-01–2026-01-05**, bundle `76eab63b69aa422b8c1a9a918f95afc7`. Five historical strong warning entries were generated. Chlorophyll mean input coverage is **1.44%**, latest day has no observed proxy labels, and data are stale. These states are prominently displayed, and default cached requests do not return current-warning maps/results. The dense forecast must not be mistaken for an observed satellite bloom. A separate 2025-02-15 input produced actual elevated warning states and was verified in the browser.

Regional elevated/strong thresholds are **13.72% / 24.17%**, fitted from 332 training days using observed strong-label fractions. They are demonstration heuristics. Normal generates no alert; elevated/strong entries appear on the dashboard/API. Historical alert entries have `operational=false`. No previous SMS/push implementation existed; no external notifications or credentials were added.

## Tests, reproducibility and remaining limitations

**27 automated tests passed**, including real data/model/API integration, with no skipped tests in the completion run. The standalone HTTP check exercised 16 requests against the listening API. Python source parse, JavaScript syntax and git diff whitespace checks passed. Core/dev and optional acquisition dependencies installed in a clean temporary environment, and both that environment and the existing `.venv` passed `pip check` after aligning boto3/botocore with the async acquisition dependency. Upstream deprecation warnings do not affect the verified test results.

Code-only checkouts still require real data and training: arrays/checkpoints are deliberately ignored. The exact clean environments are recorded in root lock files. Raw remote downloads were not repeated; source accounts/licences/service availability remain prerequisites for acquisition. No automated live ingestion, event-level HAB confirmation, calibrated public-risk probabilities, authoritative coastline, operational regional warning validation, SMS/push delivery or continuous monitoring is claimed.

## Exact commands

Run from the repository root with real aligned data present:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
python preprocessing/create_chl_climatology.py
python preprocessing/create_final_hab_labels.py
python preprocessing/report_data_quality.py
python -m preprocessing.pipeline
python -m unittest discover -s tests -v
python -m training.train --smoke --epochs 2
python -m training.train --epochs 30 --batch-size 2 --threads 4
python -m prediction.evaluation
python prediction/predict.py --historical-demo
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/dashboard/` and select Historical demonstration. The frontend needs no separate server/build. After training/evaluation, rerun automated tests to include real artifact integration. In another activated terminal while the API is listening: `python tests/http_smoke.py`. An input date can be selected in the dashboard or passed with `--input-end-date 2025-02-15 --historical-demo` to the prediction CLI.

Existing artifacts have already been trained/evaluated/generated: to view this completed workspace, only start the backend. Regenerating fitted metadata requires completing retraining/evaluation/prediction together; old artifact combinations are blocked.

Actual training/evaluation/calibration/HTTP/test records are copied under `docs/results/` so evidence remains with the source even though bulky outputs/weights are ignored. Generated figures are in `outputs/corrected/`.
