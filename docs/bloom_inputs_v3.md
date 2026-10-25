# Satellite bloom-event forecast: controlled input experiment

This experiment preserves run_01, run_02 and the frozen three-class chlorophyll baseline. The original baseline remains the primary quantitative result. Run_02 is the submission satellite-event result unless the explicitly declared improvement criteria are met.

## Fixed experiment

- Targets are exactly run_02: monthly training P75, one observed day, four-connected patch of at least two pixels, no gap. All missing/uncertain targets remain 255. No labels are made from imputed inputs.
- Chronological training 2021–2023, validation 2024, test 2025; no complete input/target window crosses a split. All feature scaling, caps and class weights use training only. Physical coastline geometry is an existing static local reference.
- Same 8/4-filter ConvLSTM, five sigmoid daily heads, masked weighted binary loss, sqrt training-negative/positive weight, seed 42, batch two, Adam 0.001 with clipnorm one, 15-epoch ceiling, patience-four stopping and validation-loss checkpoint selection. The only model-size difference is unavoidable first-layer input kernels for extra channels; filter and output sizes are fixed.
- V1 is timed first for three epochs. The pilot estimates roughly 457 seconds for 15 epochs; four runs plus assessment fit comfortably within the three-hour limit, so the original 15-epoch ceiling is retained for all variants.
- Variant selection uses mean per-horizon F1 on 2024 observed targets only. As in run_02, Platt calibration and decision fitting use first-half 2024, assessed on the second half. Checkpoint selection uses full 2024, so this is development evidence, not independent calibration validation. No 2025 model metrics are used in any input or decision selection.

## Added channels and missingness

| Variant | Added group | Total channels |
|---|---|---:|
| V0 | Reproduced run_02 control, including existing seasonal sine/cosine | 16 |
| V1 | Latest known observed satellite-event state, known flag, time since that observation | 19 |
| V2 | V1 plus chlorophyll change/validity, SST spatial front/validity, wind speed, unit U/V direction and validity, trailing mean U/V and history validity | 30 |
| V3 | V2 plus physical distance to coast | 31 |

State is carried causally within each calendar year and resets at year boundaries. It can remember observations earlier than the five-day input window; this expands past context. Unknown state is zero-filled only with a separate known flag. Observation age is capped/scaled by the training P95; unknown age is zero with known flag false. This is satellite observation state, not verified field presence/absence.

At each recurrent timestep, dynamics use that day's trailing at-most-five-day window. Thus the forecast-origin summary uses exactly the five input days; earlier timesteps can summarize older past observations. Chlorophyll change is latest observed concentration minus the average of observed concentrations, requiring at least two observations. SST gradient requires finite current-day spatial neighbours and uses approximate physical kilometre spacing. Wind history uses only valid observed U/V pairs. Invalid numerical features are zero-filled with explicit flags. Unit direction is undefined at calm wind and receives a false flag.

Numerical features use train-only P1/P99 caps and median/IQR scaling; observation age uses P95 as described. Binary flags and unit directions retain their natural scale. Coast distance is computed from the existing `outputs/roadmap_20261015/reference/land.geojson`, using segment distances in a local equirectangular kilometre approximation; points inside physical land are zero. This is not distance to the observational support boundary. No bathymetry/depth grid was found locally; none was downloaded. Day-of-year sine/cosine already exist. Optional V4 interpolation and Sentinel merging are skipped; no filled values enter targets.

Every new channel has an origin-causality subtest: changing all future environmental values, masks and event states does not change that origin's feature values. Additional tests cover missing state, observed-only chlorophyll change, physical coastline distance and matched block denominators. Software fixtures establish code contracts, not scientific accuracy.

## Block metric declared before test

Fixed non-overlapping 10×10 blocks, anchored at the grid's top-left. Partial edge blocks are padded invalid. A block requires at least ten valid target pixels; truth and predicted block labels are positive if any event occurs among exactly those valid pixels. For matched comparisons, valid pixels must also have a known causal persistence state, and at least ten of these shared pixels are required. This aggregates pixel patches and can substantially raise apparent skill; it is not independent biological event detection.

The timestamped [protocol](results/bloom_events_v3/protocol_before_test.json) records this definition before test evaluation. Full per-horizon pixel and block reports retain confusion counts, precision, recall, F1, IoU/CSI, denominators and persistence abstentions. IoU and CSI are equivalent binary measures. The existing persistence baseline uses its five-day window; new input state may remember older observations, a context difference explicitly retained here.

## Evidence and launch

- [2024 ablation](results/bloom_events_v3/candidate_comparison.md)
- [Selected-variant 2025 evaluation](results/bloom_events_v3/evaluation.md)
- [Feature records and fitted statistics](../data/processed/bloom_events_v3/features.json)
- [Selected run and source snapshots](../outputs/bloom_events/run_03/)

```bash
cd /Users/vinayaks/Documents/HAB-Early-Warning-Submission
BLOOM_EVENT_RUN_DIR=outputs/bloom_events/run_02 MPLCONFIGDIR=/tmp/satellite-event-mpl \
  .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8016
```

Open `/dashboard/bloom.html`; select run_03 for the controlled input experiment, or preserved run_01/run_02. Historical input end 2024-02-15 forecasts 16–20 February. The run is disabled until training, selection, freeze and evaluation finish. Baseline remains `/dashboard/`; the field-verified workflow remains unavailable at `/dashboard/hab.html`.

All 2025 results must disclose prior inspection, correlated rolling pixel forecasts and satellite-derived labels. No harmful species, toxicity, verified harmful occurrence or operational reliability is established. No further tuning follows the selected 2025 evaluation.

## Frozen outcome and interpretation

V3 had the highest mean 2024 pixel F1 (0.415829), versus V0 0.278723, V1 0.379773 and V2 0.371725. V2 dynamics alone did not improve V1 and V2 was not selected. V3 is the predeclared cumulative combination including dynamics plus coast distance; this experiment does **not** isolate a benefit from dynamics or test V1-plus-coast without dynamics. No undeclared pruning/retraining variant was added after viewing results. Thus the instruction to discard every non-improving intermediate group is not independently resolved within this fixed cumulative design; the primary maximum-2024-F1 rule chose V3. This is an additional experiment with that limitation, not evidence that every retained group helps.

The one 2025 evaluation has higher observed-target F1/IoU than run_02 and higher matched-pixel and matched-block F1/IoU than persistence at each horizon. Mean 2024 matched F1 still favours persistence, so the 2025 advantage is not stable proof of generalization. Original frozen chlorophyll results remain the primary quantitative result. Offline metrics include all within-split windows with known targets; dashboard inference separately applies the original environmental observation-coverage gate. Filled/carried features do not count as new observations in that gate.
