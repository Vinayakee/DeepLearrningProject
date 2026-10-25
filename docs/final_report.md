# Run_03 final report

**Satellite bloom-event forecast; the frozen three-class chlorophyll baseline remains the primary quantitative result.**

Created `preprocessing/bloom_inputs_v3.py`, `training/train_bloom_inputs_v3.py`, two test modules, `data/processed/bloom_events_v3/`, `outputs/bloom_events/run_03/`, and this report directory. Dashboard/API, README and submission handoff include run_03 alongside unchanged runs 01/02.

## 2024 ablation and selection

| Variant | Added group | Channels | Mean observed-target F1 |
|---|---|---:|---:|
| V0 | Control | 16 | 0.278723 |
| V1 | Carried event state + known flag + age | 19 | 0.379773 |
| V2 | Dynamics | 30 | 0.371725 |
| V3 | Physical coast distance | 31 | 0.415829 |

**Chosen: V3**, highest 2024 mean F1, frozen before one selected 2025 evaluation. V0 exactly reproduced run_02 training/validation history. Same targets, filter/head sizes, loss, class weights, seed and stopping recipe; parameter count increases from input kernels only (8,737→13,057). V1 three-epoch pilot projected ~457 seconds/run, retaining the 15-epoch ceiling. V3 trained 14 epochs, checkpoint epoch 10.

Platt calibration was rejected on 2024 development data: second-half Brier worsened from 0.099037 to 0.101003. Scores remain raw/uncalibrated; the first-half validation F1 cutoff is 0.274019897. No threshold or input choice changed after test.

## 2025 pixels

| Day | All valid pixels | Precision | Recall | F1 | IoU/CSI | Matched pixels | Model / persistence F1 | Model / persistence IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 128680 | 0.2750 | 0.6510 | 0.3866 | 0.2396 | 59622 | 0.3546 / 0.3423 | 0.2155 / 0.2065 |
| +2 | 129087 | 0.2957 | 0.6087 | 0.3980 | 0.2485 | 58440 | 0.3710 / 0.3413 | 0.2277 / 0.2057 |
| +3 | 127081 | 0.2708 | 0.6315 | 0.3791 | 0.2339 | 55518 | 0.3410 / 0.3107 | 0.2056 / 0.1839 |
| +4 | 127081 | 0.2748 | 0.5986 | 0.3766 | 0.2320 | 53143 | 0.3308 / 0.3040 | 0.1982 / 0.1792 |
| +5 | 126794 | 0.2592 | 0.5808 | 0.3584 | 0.2183 | 52996 | 0.3140 / 0.3014 | 0.1862 / 0.1774 |

## 2025 blocks

10×10 fixed blocks need ≥10 known target pixels. Truth/prediction is positive if any of those pixels is positive. Matched blocks use only the target-known/persistence-known intersection, again ≥10 pixels. This rule was timestamped before test.

| Day | All valid blocks | Model F1 | Model IoU/CSI | Matched blocks | Model / persistence F1 | Model / persistence IoU |
|---|---:|---:|---:|---:|---:|---:|
| +1 | 2296 | 0.6084 | 0.4372 | 1257 | 0.5922 / 0.5756 | 0.4206 / 0.4041 |
| +2 | 2310 | 0.5965 | 0.4251 | 1242 | 0.5817 / 0.5759 | 0.4101 / 0.4044 |
| +3 | 2276 | 0.5839 | 0.4123 | 1180 | 0.5662 / 0.5392 | 0.3948 / 0.3691 |
| +4 | 2276 | 0.5824 | 0.4108 | 1127 | 0.5652 / 0.5271 | 0.3939 / 0.3578 |
| +5 | 2270 | 0.5659 | 0.3946 | 1122 | 0.5462 / 0.5206 | 0.3757 / 0.3519 |

Observed-target mean F1 improved from run_02 ~0.3304 to 0.3798; mean IoU/CSI is 0.2345. Run_03 beats matched persistence F1/IoU at all five horizons for pixels and blocks. Treat this as additional input-experiment evidence: 2025 was previously inspected, and mean matched **2024** F1 still favoured persistence (0.4946 versus V3 0.4769).

## What failed / was skipped

- Dynamics V2 did not improve over V1. The winning predeclared V3 is cumulative and still includes dynamics; their individual benefit or removal is **not** established. Fully removing every non-improving intermediate group was not resolved within the fixed candidate list; no undeclared pruning experiment or further test tuning was performed.
- Calibration was rejected; raw scores are not calibrated probabilities. No biological event detection/extent or harmful-species/toxicity validation is claimed.
- Bathymetry was unavailable locally; optional V4 interpolation was skipped. No Sentinel merging or downloads.
- Rolling forecasts are correlated. Block aggregation is not independent biological event detection. Carried state can remember older observations than the original five-day persistence comparator. Offline evaluation includes sparse input windows; the dashboard separately gates on original observed inputs.
- Initial software checks exposed a local/UTC date fixture mismatch and a coverage helper limited to 16 channels. The fixture and new-run adapter were corrected; scientific decisions and test results were preserved.

**Verification:** 130 Python tests, JavaScript/dependency/whitespace checks, 19 HTTP checks for each event run plus 24 baseline checks. All 129 protected files unchanged; baseline scientific audit and all three freezes pass. No variant except frozen V3 has test scores. Verified field-observation workflow remains untrained.

## Run the demo

```bash
cd /Users/vinayaks/Documents/HAB-Early-Warning-Submission
BLOOM_EVENT_RUN_DIR=outputs/bloom_events/run_02 MPLCONFIGDIR=/tmp/satellite-event-mpl \
  .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8016
```

Open [dashboard](http://127.0.0.1:8016/dashboard/bloom.html), choose run_03, keep historical example 2024-02-15, generate, switch days and export CSV. The task-owned server is already running on 8016. No retraining is required.

[Full results](evaluation.md) · [Ablation](candidate_comparison.md) · [Preservation](preservation.json) · [Handoff](../../SUBMISSION_HANDOFF.md)

![Verified run_03 day-five dashboard](dashboard_run03_day5.png)

Automation cleanup: the user explicitly approved deletion, and the completed recurring automation was deleted on 25 October 2026. Scientific results, models and the dashboard remain unchanged.
