# 2024 input-only ablation

| Variant | Added group | Channels | Parameters | Best epoch | Mean pixel F1 | Delta | Matched F1 | Persistence matched F1 | Mean block F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | Control | 16 | 8737 | 10 | 0.278723 | — | 0.360607 | 0.494624 | 0.508836 |
| V1 | Persistence state/age | 19 | 9601 | 15 | 0.379773 | +0.101050 | 0.450145 | 0.494624 | 0.613939 |
| V2 | Dynamics | 30 | 12769 | 8 | 0.371725 | -0.008048 | 0.432088 | 0.494624 | 0.610468 |
| V3 | Physical coast distance | 31 | 13057 | 10 | 0.415829 | +0.044104 | 0.476864 | 0.494624 | 0.623788 |

Chosen variant: **V3**, using maximum mean per-horizon 2024 observed-target F1 before any run_03 2025 metric inspection.

V2 did not improve mean validation F1 over the preceding nested variant; that addition has no demonstrated incremental benefit.

Nested ablation; a selected later variant cannot isolate any intermediate non-improving group without an undeclared extra variant. Such attribution is not claimed.

Full per-horizon pixel, block and matched-persistence validation metrics are retained in candidate_comparison.json and each variants/V*/validation.json. Decision fitting uses first-half 2024, checkpoint full 2024, so comparisons remain development evidence. V0 is freshly trained using the same seed handling as V1–V3; no test results were used.
