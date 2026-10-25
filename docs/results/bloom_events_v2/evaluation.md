# Satellite bloom-event forecast — relaxed run_02

**2025 was previously inspected. Satellite-derived proxy labels; no verified-HAB or operational claim.**

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

- Chlorophyll-derived satellite patches, not verified HAB species, toxicity or harmful occurrence.
- The selected definition allows one observed day: it detects spatial patches, not persistent multi-day blooms.
- Unobserved and uncertain target pixels stay unknown; monitoring absence is not inferred.
- Rolling pixel forecasts and connected components are correlated; component counts are not independent biological events.
- Model metrics on all observed targets include sparse input windows; dashboard inference is separately coverage-gated.
- Calibration and threshold selection are 2024 development fits; the checkpoint also used 2024, so calibration assessment is not independent.
- 2025 was previously inspected, including strict-run results; this is not an untouched holdout.
- Satellite publication latency is unverified; nominal forecast horizons do not establish operational warning lead.
