# Satellite bloom-event forecast — input-only run_03

**Satellite-derived proxy targets; 2025 previously inspected. Rolling pixel forecasts are correlated, not independent biological events.**

Chosen on 2024 mean observed-target F1: V3. No 2025 scores used for variant or decision selection.

| Day | Valid pixel N | P | R | F1 | IoU/CSI | Shared pixel N | Matched model F1 | Persistence F1 | Matched model IoU | Persistence IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 128680 | 0.2750 | 0.6510 | 0.3866 | 0.2396 | 59622 | 0.3546 | 0.3423 | 0.2155 | 0.2065 |
| +2 | 129087 | 0.2957 | 0.6087 | 0.3980 | 0.2485 | 58440 | 0.3710 | 0.3413 | 0.2277 | 0.2057 |
| +3 | 127081 | 0.2708 | 0.6315 | 0.3791 | 0.2339 | 55518 | 0.3410 | 0.3107 | 0.2056 | 0.1839 |
| +4 | 127081 | 0.2748 | 0.5986 | 0.3766 | 0.2320 | 53143 | 0.3308 | 0.3040 | 0.1982 | 0.1792 |
| +5 | 126794 | 0.2592 | 0.5808 | 0.3584 | 0.2183 | 52996 | 0.3140 | 0.3014 | 0.1862 | 0.1774 |

Platt calibration was rejected on 2024 development data: second-half Brier worsened from 0.099037 to 0.101003. Scores remain raw/uncalibrated; the first-half validation F1 cutoff is 0.274019897. No threshold or input choice changed after test.

## 10×10 blocks

Fixed non-overlapping grid blocks (edge blocks padded invalid), at least 10 valid target pixels. Positive if any valid target pixel is an event; prediction any positive on exactly those pixels. Matched blocks restrict both truth and predictions to pixels where persistence is known; require at least 10 of those pixels. Block aggregation can greatly inflate apparent performance and is not independent event detection.

| Day | Valid blocks | P | R | F1 | IoU/CSI | Shared blocks | Matched model F1 | Persistence F1 | Matched model IoU | Persistence IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 2296 | 0.5337 | 0.7073 | 0.6084 | 0.4372 | 1257 | 0.5922 | 0.5756 | 0.4206 | 0.4041 |
| +2 | 2310 | 0.5430 | 0.6618 | 0.5965 | 0.4251 | 1242 | 0.5817 | 0.5759 | 0.4101 | 0.4044 |
| +3 | 2276 | 0.5175 | 0.6697 | 0.5839 | 0.4123 | 1180 | 0.5662 | 0.5392 | 0.3948 | 0.3691 |
| +4 | 2276 | 0.5251 | 0.6538 | 0.5824 | 0.4108 | 1127 | 0.5652 | 0.5271 | 0.3939 | 0.3578 |
| +5 | 2270 | 0.5035 | 0.6459 | 0.5659 | 0.3946 | 1122 | 0.5462 | 0.5206 | 0.3757 | 0.3519 |

Run_03 improves mean F1/IoU over run_02 and beats matched persistence at every horizon; report as additional improved-input evidence.

Full JSON includes all P/R/F1/IoU/CSI, confusion counts, pixel Brier and abstentions for both matched cohorts. Known-label state input can carry observations beyond five input days; the preserved persistence comparator uses the original five-day window. Added recurrent input kernels increase parameter count with channel count; filters and heads are fixed. Calibration assessment shares 2024 development data with checkpoint selection. Nominal horizons do not prove data publication latency.
