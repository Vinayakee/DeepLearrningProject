# Runtime integration and reproducibility

This extends the existing historical research workflow without replacing its frozen model, normalization, labels, experiments or 2025 reports. It does not claim a reliable operational HAB detector. No new field/satellite observations were acquired. Tests with explicitly marked fixtures are engineering checks, not validation evidence.

## Code-only clone and required artifacts

Use Python 3.13 on the tested macOS ARM64 platform, install `requirements-dev.txt` (or the existing tested lock), and run `python -m pip check`. No new dependency or frontend build is required. Optional original download tools use `requirements-download.txt` and source credentials.

Data/weights are intentionally excluded from Git. Obtain the real aligned seven-channel arrays, masks, coordinates and dates described in README; the corrected label snapshot; frozen final preprocessing, model, training/decision/freeze files; and baseline snapshot if verifying the baseline. Preserve a complete matching artifact set. Missing files return unavailable/503 or a failed update, never substitute random data. Live preprocessing currently validates the existing historical fitted archive too; model weights alone are insufficient.

`.env.example` lists runtime/provider configuration without secrets. Export variables in the shell or service environment; no dotenv loader is implicit. Keep real credentials only in ignored `.env` or a secret store. Frozen scientific artifacts remain the source of deployment settings; do not alter them to enable alerts.

## Historical mode

```bash
source .venv/bin/activate
python -m experiments.finalize verify
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `/dashboard/`, select 2024-02-15 and Historical demonstration, Generate forecast; or use Find covered historical date. Existing `/api/predict`, cache, maps and evaluation APIs remain supported. Old data cannot become current warnings. Live mode polls the API every 60 seconds while no manual generation is running; demonstration mode is not polled into new forecasts.

## Real staged/live inputs

The exact staged NPZ/JSON contract is in [real-data interfaces](real_data_interfaces.md). Arrays retain seven original fields, actual WGS84 reference coordinates, binary masks and five consecutive UTC calendar dates; missing recent frames may remain unavailable. Every channel needs product-ID lists, native QC references, actual observation end and publication/availability timestamps. Publication cannot precede acquisition or exceed execution/first target time. Claimed valid frames cannot postdate their source observation end.

```bash
python -m operations.update
python -m operations.update --input-dir /absolute/path/to/real/stage
```

The first command reports not_configured unless `HAB_LIVE_INPUT_DIR` or `HAB_LIVE_ADAPTER=module:function` is supplied. The plugin callable downloads/quality-controls real products and returns a staged directory. Configure credentials according to the chosen service; no generic plugin can invent product availability or processing policy. The supported orchestration does not import historical batch downloaders.

Validation → frozen preprocessing → original coverage/freshness gates → frozen ConvLSTM → unchanged decisions/regional summary → integrity-checked bundle/API → dashboard → notification gate are connected. Missing, insufficient, low-coverage, stale and invalid timing are distinct states, without forced inference. Exceptions preserve the previous successful-update identity but mark the current update unavailable. Source/provenance hashes are retained. `/api/live/prediction` serves only valid fresh research bundles; `/api/system` exposes source, age and last update.

## Sentinel research mode

```bash
python -m operations.update --input-dir /absolute/path/to/real/stage --sentinel-dir /absolute/path/to/real/qc_sentinel --research-fusion
```

Sentinel NPZ/JSON fields and version-specific native flag requirements are documented in real-data interfaces. Source acquisition/availability timestamps are additionally required. Only QC-passing observations within the input window participate. Source data older than the frozen three-day limit relative to the input end are ignored. Invalid, absent, all-rejected and unaligned Sentinel inputs explicitly fall back to the untouched MODIS bundle; if that bundle lacks sufficient coverage inference is withheld.

Valid Sentinel fills only missing MODIS cells; observed MODIS values, proxy labels and model weights are never replaced. Save a 0=missing/1=MODIS/2=Sentinel cell-day provenance array and its hash, source IDs, QC counts and overlap agreement. Combined source timestamps conservatively use the latest of both sources. The API/dashboard identify actual source usage.

This is explicitly research inference integration, disabled by default. QC alone does not establish cross-sensor compatibility or forecast skill. Fusion cannot enable operational warnings. No real usable QC Sentinel bundle exists here; no coverage gain or fusion-performance improvement is claimed. Compare overlap/coverage and class/regional skill on genuinely supplied training/development data before any deployment decision.

## Scheduling

```bash
python -m operations.update --watch --interval-seconds 3600
```

This is an opt-in polling worker, not an installed scheduler. It reads the same configured adapter/stage and uses a cross-process `outputs/live/update.lock` to prevent overlapping updates. Use a supervisor/external scheduler to run this command with the chosen environment and capture stdout. Stop with Ctrl-C. No background service or remote acquisition was enabled here. After an abnormal process termination, inspect the PID in update.lock and remove it only after confirming its owner is gone; locks are not silently stolen.

## Supplied-input hindcasts and independent class evaluation

```bash
python -m operations.hindcast --input-dir /absolute/path/to/real/stage --output-dir outputs/independent/unique_forecast
python -m operations.validation independent --data /absolute/path/to/evaluation_bundle --output outputs/independent/unique_report.json
```

Hindcast inference uses real supplied QC inputs and the frozen pipeline, requires sufficient coverage and verified source timing, but deliberately permits old inputs for research evaluation. It never claims live/fresh warnings. It refuses any input/target in 2025 and refuses output overwrite. Generate one forecast per required origin from real supplied stages; no training or tuning occurs.

Independent `evaluation.npz` contains `dates` (unique increasing target dates after 2025), `latitude`, `longitude`, `actual`, `predicted` (T,H,W classes 0/1/2 or missing255), `available` (T binary), `ocean` (H,W binary subset of fitted support). Metadata JSON requires `independent:true`, `label_definition`, `label_provenance`, frozen `model_sha256`, `decision_rule_sha256`, and `forecast_directories` with one actual saved horizon-one forecast per date. Paths are resolved as supplied; use absolute paths. Predictions, availability, target/input timing and hashes are checked against those saved outputs. Do not fabricate class labels from unadjudicated toxin/species records or fit thresholds on evaluation data.

Reports include all three class P/R/F1/support and confusion matrix, regional any/strong metrics, observation coverage, available days, unknown regional truth and abstentions. Empty observations report insufficient_data; absent classes retain zero support and cannot establish class skill. Region masks can evaluate locations inside the existing grid. A genuinely different geographic grid is reported unsupported_region: this Kerala-trained model cannot be claimed transferable merely by resizing input arrays. All supplied labels must have a scientific definition comparable to the proxy-class task for class/regional interpretation.

No post-2025 independent data is available locally. The interface is tested with contract fixtures; no new independent model score is claimed. Original 2025 results remain preserved and no new experiment was chosen using them.

## Verified field matching

```bash
python -m operations.validation field --data /absolute/path/to/verified.csv --forecast-dir /absolute/path/to/real/forecast --purpose independent --output outputs/field/unique_report.json
```

Use the existing verified CSV schema with explicit true/false HAB status, UTC observation time, coordinates and verification provenance. Optional species, severity, toxin_value/toxin_units are retained as observations without inventing a severity-to-risk mapping. Independent records/forecast dates must be after 2025; development mode permits 2024 records only.

Match actual target UTC date and nearest grid center within the real footprint; out-of-grid points stay unmatched, unsupported coastal cells abstain. Duplicate record/horizon forecasts are rejected; choose one origin reproducibly. Each horizon reports available-forecast binary P/R/F1/TP/FP/FN/TN and event detection counting abstentions as misses, plus match records and unmatched reasons. Positive proxy class >=1 is a prediction to compare with verified presence, not confirmation of harmful species. Point samples cannot establish regional absence or a strong-pixel fraction, so field regional metrics explicitly remain unavailable. Lead-time provenance is reported separately; manifest timestamps alone do not prove the original source claims.

## Notification providers

```bash
python -m operations.notifications --bundle /absolute/path/to/bundle.json
```

Default dry-run prints rich payloads (region, risk, forecast date/horizon, generation timestamp, source, QC) and whether the operational gate is satisfied. Missing/insufficient/low/stale data is blocked even in dry-run. Historical stale forecasts cannot preview current delivery. Local/demo outbox from the original interface remains available.

SMTP uses configurable host/port/from/to and optional username/password with mandatory STARTTLS and certificate checking. SMS uses a configured HTTPS gateway accepting `{to,message}` JSON plus Bearer token and Idempotency-Key; it must return `{accepted:true,message_id:...}`. This is a gateway contract, not a claim that every SMS vendor accepts the same schema.

External sending additionally requires `--deliver` (or worker `HAB_NOTIFICATION_DRY_RUN=0`), `HAB_NOTIFICATIONS_ENABLED=1`, a configured provider, fresh sufficient non-demo data AND genuine frozen operational validation. Current validation is false, so external delivery remains blocked. No email/SMS was sent. Provider acceptance is distinguished from confirmed recipient delivery. Persisted claims prevent duplicate sends; uncertain failures are retained for inspection rather than blindly retrying non-idempotent SMTP.

## Calibration and legacy utilities

Health and system endpoints share artifact_exists, tested, active, operationally_approved and separate decision_rule_artifact_exists metadata. Legacy calibration_available now means deployed probability calibration and is false. A decision artifact is not probability calibration. Saved 2024 temperature assessment remains tested but inactive; raw scores remain raw scores.

Supported runtime: `operations/`, `backend/`, `prediction/service.py`, shared fitted preprocessing/model. Historical batch/reference tools: `preprocessing/download_*`, `preprocess_5years_*`, `preprocess.py`, `preprocess_month.py`, `align_data.py`, early label/visualization utilities and `training/train_backup.py`. Their fixed dates preserve historical reproducibility; do not schedule/import them as runtime acquisition. No legacy utility was deleted or rewritten.

## Scientific boundary

No controlled experiment was repeated: saved weights/rules/seasonal/calibration/sampling comparisons are in the prior reports. Sampling and regional-support variants worsened validation and remain rejected. Current STRONG and regional skill remain poor. Corrected proxy labels are not field-confirmed blooms; source-specific regional acquisition/publication timing, additional QC observations and verified events are required. Integration code cannot by itself improve the frozen detector's skill.
