# Satellite bloom-event forecast — frozen 2025 evaluation

**2025 was previously inspected. These labels are satellite-derived; no confirmed HAB skill is claimed.**

Observed test components: 0. Recall and PR AUC are not estimable without positive truth.

| Horizon | Model valid targets | Precision | Recall | F1 | IoU / CSI | Persistence valid targets | Persistence F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| +1 | 122407 | 0.000000 | N/A | 0.000000 | 0.000000 | 55608 | N/A |
| +2 | 122815 | 0.000000 | N/A | 0.000000 | 0.000000 | 54370 | N/A |
| +3 | 120888 | 0.000000 | N/A | 0.000000 | 0.000000 | 51414 | N/A |
| +4 | 120888 | 0.000000 | N/A | 0.000000 | 0.000000 | 49459 | N/A |
| +5 | 120601 | 0.000000 | N/A | 0.000000 | 0.000000 | 49315 | N/A |

Model and standalone persistence use different availability cohorts. Use the paired `common_cohort` JSON counts for matched comparisons. Undefined metrics are N/A, never invented zeros.

- Chlorophyll-derived satellite event labels, not confirmed HAB occurrence or toxicity.
- Strict observed three-day common-patch rule; missing candidates stay unknown, not no-event.
- Rolling pixel forecasts overlap and are correlated, not independent events.
- Primary model metrics include all observed targets, including weak input windows; dashboard availability is separately coverage-gated.
- No positive validation/test targets means detection generalization and calibration cannot be established.
- Input publication latency is unverified; nominal horizons do not prove operational warning lead time.
