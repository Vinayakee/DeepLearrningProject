# Integration completion verification

Overall status: **PARTIALLY COMPLETE**. The software integrations below are implemented and tested. Fresh retrieval, usable QC Sentinel products, verified field observations and a new independent evaluation dataset remain unavailable. Frozen detector performance remains inadequate for operational warnings. No claim of real-world live validation or new scientific improvement is made.

## Implemented in this round

- `operations.acquisition`: configurable real staged-directory or source-plugin acquisition with explicit missing/invalid configuration states.
- `operations.provenance`: product identity, UTC acquisition/publication ordering and valid-frame timestamp checks.
- `operations.update`: protected one-shot/polling updates, Sentinel research bridge, distinct coverage/staleness failures, preserved last success and gated delivery.
- `operations.fusion`: existing strict Sentinel QC and MODIS-preserving gap fill connected to staged inference; source/QC/overlap reports and hashed cell-day provenance. Invalid/missing/stale Sentinel falls back to MODIS, never fake measurements. Default disabled; operational alerts blocked for fusion.
- `operations.hindcast`: real supplied-stage frozen inference with no training, no 2025 reuse, no output overwrite and honest missing-input handling.
- `operations.validation`: verified-field date/location/horizon matching, binary event metrics with explicit abstentions, optional species/severity/toxin retention, and supplied-label independent class/regional evaluation with source-forecast integrity checks. New geographic grids are unsupported rather than silently resized.
- `operations.notifications`: rich dry-run payloads, STARTTLS SMTP and HTTPS SMS-gateway transports, persistent delivery claims and separate provider acceptance. Approval is checked against frozen deployment, not merely caller JSON. Default disabled; no external message sent.
- Health/system probability metadata now agrees: decision artifact existence is separate from calibration artifact/tested/active/approved state.
- Existing frontend uses actual backend source/coverage/calibration/update state and polls fresh research bundles. Existing historical functionality remains intact.
- Environment template, reproducibility/workflow docs and explicit separation of legacy fixed-date research scripts from supported runtime orchestration.

## Verification evidence

**82 Python tests passed, zero failures, no skipped tests in this environment.** The original 56 were preserved. JavaScript request-order regression and actual-backend-JSON DOM rendering passed; syntax parsing covered all 97 Python files, JS syntax passed, `pip check` found no broken requirements and `git diff --check` passed. All baseline/final/post-freeze scientific hashes and saved 2025 evaluation bytes matched their archived evidence.

| Case | What was actually verified | Result |
|---|---|---|
| A: historical | Real 2024-02-15 API input → shared transform → trained ConvLSTM → five maps/summary/cache/API | Passed; all five PNGs HTTP200 |
| B: missing | Actual 2024-05-08 staged archive window, fixed test clock | Missing-data state, model not called |
| C: insufficient/low | Actual 2024-03-07 (one chlorophyll day) and 2024-03-11 (two days, 4.779% mean coverage), fixed test clock | Withheld model; separate insufficient/low states |
| D: stale | Actual old staged inputs at current execution time, cache default mode | Withheld inference/forecast; explicit stale state |
| E: staged/live interfaces | Actual historical arrays/model through staged update, five maps and live API under a fixed historical test clock, plus separate hindcast comparison | Passed; TEST provenance manifest, not fresh acquisition evidence |
| F: invalid Sentinel | Bad units/timing, clouds, missing and stale contract fixtures | MODIS fallback; no invalid Sentinel participation |
| G: valid Sentinel | Strict QC-passing contract fixture gap fill/provenance plus integration wiring | Passed engineering contract; real QC Sentinel data unavailable, no fusion skill claim |
| H: fields | Verified-schema contract fixtures, target/coordinate/horizon matching, duplicates, out-of-grid, abstention and toxin validation | Passed; actual verified records unavailable |
| I: independent | Supplied-label contract fixture, per-class/regional metrics, 2025 rejection and tampered prediction rejection | Passed contract; actual independent evaluation unavailable |
| J: notifications | Rich dry-run/gate fixtures; real historical bundle dry-run; mocked SMTP/SMS transports and idempotency | Passed; actual stale bundle blocked; no external sends |
| K: dashboard/API | Actual restarted health/system metadata, live-unavailable API and real forecast JSON through all five dashboard day renderers | Passed data binding; no new browser layout validation claimed |

Fixtures remain labeled tests and temporary files; they are never used to publish scientific metrics or create field ground truth. Fixed-clock tests explicitly do not change/archive observation dates or claim freshness today.

## Commands actually executed

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/http_smoke.py --input-end-date 2024-02-15
.venv/bin/python -m experiments.finalize verify
.venv/bin/python -m pip check
.venv/bin/python -m operations.update
.venv/bin/python -m operations.validation field --data data/verified_hab_records.csv --output outputs/operations/field_unavailable.json
.venv/bin/python -m operations.validation independent --data data/independent --output outputs/operations/independent_unavailable.json
.venv/bin/python -m operations.notifications --bundle outputs/operations/actual_api_forecast.json
.venv/bin/python -m operations.hindcast --input-dir data/nonexistent_live_stage --output-dir outputs/operations/missing_hindcast
git diff --check
```

Both standalone JavaScript checks used the bundled Node runtime: `tests/dashboard_race.mjs`, `tests/dashboard_render.mjs outputs/operations/actual_api_forecast.json`; `--check frontend/script.js` passed. Direct HTTP health/system equality and source/live checks were run too. Missing real-data commands correctly reported unavailable/not_configured; existing report paths deliberately refuse overwrite on repetition.

Saved engineering evidence: [Python log](results/operations/tests.log), [HTTP smoke](results/operations/http_smoke.json), [metadata](results/operations/api_metadata.json), [notification dry-run](results/operations/notification_dry_run.json), [field-unavailable report](results/operations/field_unavailable.json), [independent-unavailable report](results/operations/independent_unavailable.json). These are not new evaluation metrics. Detailed real input/provider schemas and enabled-mode commands are in [runtime workflows](runtime_workflows.md).

## Performance and limitations retained

No scientific experiment, class-weight/seasonal/sampling run or 2025 evaluation was repeated. Prior sampling and regional-support changes were rejected objectively and remain rejected. Saved final 2025 P/R/F1: NORMAL .9004/.8856/.8929; ELEVATED .1969/.3101/.2409; STRONG .1126/.0413/.0604. MacroF1 .3981. Regional any-risk TP0/FN12/FP0, strong TP0/FN8; zero false alarms do not imply useful detection. Original 2024 MODIS coverage remains 8.30%; no new coverage gain is claimed.

Probability calibration artifact/test exists but activation/operational approval remain false. All forecasts remain chlorophyll proxies, not confirmation of harmful species/toxicity. Point field samples cannot establish regional risk fractions. Source-native QC/georeferencing/timing cannot be proved solely by a manifest. Real source plugins/credentials must implement those claims. Current live state remains not_configured, with no successful fresh update. SMTP/SMS provider acceptance tests use mocks; recipient delivery is not claimed. Polling architecture is opt-in, not a deployed scheduler/service; crash locks require owner inspection.

Nonfatal warnings: FastAPI/httpx and Keras/NumPy deprecations, test-checkpoint optimizer variable warning and NumPy/netCDF extension compatibility warning; real read/inference tests nevertheless passed. Broad geographic transfer, reliable STRONG warnings, calibrated operational confidence and new independent field validation remain unestablished.

**IMPLEMENTATION STILL REQUIRES EXTERNAL DATA/DEPENDENCIES.**
