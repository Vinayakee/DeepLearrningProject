# Satellite bloom-event forecast — submission copy

**Headline experiment: run_03 (V3 causal inputs).** The frozen chlorophyll baseline and runs_01–02 are earlier results. This is a submission copy; the original project is not needed at runtime and is blocked from runtime access. Scientific checkpoints, saved metrics and frozen source files are preserved without refitting or reevaluation.

## Install

Tested with **Python 3.13 on macOS Apple Silicon**. Node.js is needed for the JavaScript regression checks; no Node packages are required. Extract the ZIP with a utility that preserves Unix symlinks (macOS Archive Utility or `unzip`), then:

```bash
cd HAB-Early-Warning-Submission
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

`requirements.txt` contains pinned direct runtime dependencies. `requirements-verified.txt` records the complete freshly installed verification environment. Raw-data acquisition dependencies are optional in `requirements-download.txt`.

## Run the demo

```bash
python -m submission serve --port 8020
```

Open [satellite bloom-event forecast](http://127.0.0.1:8020/dashboard/bloom.html). Select **run_03**, leave input end **2024-02-15** and **Historical demonstration** enabled, generate, switch Day +1 through +5, inspect maps and export CSV. Run_01/run_02 remain selectable. A 2025-12-31 input end demonstrates insufficient observed inputs. Stop the server with Ctrl+C; use a different free port if needed.

- Frozen chlorophyll baseline: `/dashboard/`.
- Verified-observation workflow: `/dashboard/hab.html`; implemented but **untrained/not configured**.
- Saved results: `/api/bloom/evaluation?run_id=run_03` (also run_01/run_02).

The launcher supplies copy-only path relocation for immutable metadata and restores original array timestamps after ZIP extraction. **Use this launcher or `make serve`**, rather than invoking the legacy server command directly. `sitecustomize.py` also enables relocation when the project is explicitly on `PYTHONPATH`. No symlink points into the original project, and no frozen hash check is disabled.

## Results and limitations

| Result | Preserved finding |
|---|---|
| **run_03 / V3** | 2024 mean observed-target F1 0.415829. 2025 observed-target mean F1 0.3798, IoU/CSI 0.2345; matched pixel/block F1 and IoU exceed persistence at all five horizons. Scores are uncalibrated because calibration was rejected. |
| run_02 | 2025 observed-target F1 about 0.324–0.341; persistence has higher matched F1/IoU at all horizons. |
| run_01 | Strict labels have zero positive validation/test targets; recall is undefined. |
| Frozen chlorophyll baseline | Earlier three-class result and its original saved evaluation are retained. |

2025 was previously inspected. Rolling pixel forecasts are correlated, and block aggregation is not independent biological event detection. Labels are chlorophyll-derived satellite targets. V2 dynamics did not improve over V1; the winning cumulative V3 does not isolate their benefit/removal. No harmful species, toxicity or operational validation is established. All detailed claims, counts and limitations remain in the original reports.

Start with [submission handoff](docs/SUBMISSION_HANDOFF.md), [final report](docs/final_report.md), [full pixel/block metrics](results/bloom_events_v3/evaluation.md), [2024 ablation](results/bloom_events_v3/candidate_comparison.md), [limitations](docs/limitations.md), [abstract](docs/abstract.md), and [presentation notes](docs/presentation_notes.md).

## Tests and frozen audit

```bash
python -m submission check
python -m submission audit
```

Equivalently, `make check` / `make audit` with `.venv` installed. The full suite contains 133 tests: **132 pass; one optional raw-MODIS test is skipped** because large raw downloads are excluded. Dependency and JavaScript checks pass. All three event-model HTTP flows and the original baseline are checked separately during packaging; saved scientific test evaluations are not rerun. Archive extraction also passes the frozen audit and loads all three event runs independently, with the original checkout blocked. See `.submission/verification.json` for the verification record and `data/README.md` for the skipped raw-product filename/checksum.

## Folder map

The existing relative layout is retained because imports and immutable manifests depend on it. `dashboard`, `results`, and `models/run_*` are internal aliases, not duplicate copies.

- `preprocessing/`, `training/`, `prediction/`, `backend/`: preparation, fitting, evaluation/inference and API.
- `frontend/` (`dashboard/` alias): dashboard assets.
- `data/`: required aligned arrays, frozen targets/features and small public source-audit evidence.
- `models/`: model definitions plus aliases to frozen checkpoint directories under `outputs/`.
- `outputs/improvement/`: frozen chlorophyll baseline, final checkpoint and fitting metadata.
- `outputs/bloom_events/run_01`, `run_02`, `run_03`: final checkpoints, decisions, freezes and source snapshots.
- `docs/results/` (`results/` alias): final scientific reports, tables and figures.
- `docs/`: handoff, methodology, abstract, presentation notes and field-source audit.
- `tests/`: original regression tests plus three packaging portability checks.
- `submission/`, `sitecustomize.py`, `.submission/`: copy-only launcher, relocation and packaging/audit records.

### Tree (depth 3)

```text
HAB-Early-Warning-Submission/
  .github/
    workflows/
      checks.yml
  .submission/
    aliases.json
    copy_changes.json
    exclusions.json
    inventory.json
    original-preservation.json
    original-tree-before.json.gz
    package.json
    relocation.json
    selected_files.json
    source_file_identity.json
    timestamps.json
    tree-depth-3.txt
    verification.json
  backend/
    app.py
  dashboard -> frontend
  data/
    processed/
      bloom_events/
      bloom_events_v2/
      bloom_events_v3/
      training_only_labels/
      chlorophyll_2021_2025.npy
      chlorophyll_mask_2021_2025.npy
      current_u_2021_2025.npy
      current_u_mask_2021_2025.npy
      current_v_2021_2025.npy
      current_v_mask_2021_2025.npy
      dates_2021_2025.txt
      latitude_kerala_2021_2025.npy
      longitude_kerala_2021_2025.npy
      rainfall_2021_2025.npy
      rainfall_mask_2021_2025.npy
      sentinel3_chl_2021_2025.npy
      sentinel3_mask_2021_2025.npy
      sst_2021_2025.npy
      sst_mask_2021_2025.npy
      wind_u_2021_2025.npy
      wind_u_mask_2021_2025.npy
      wind_v_2021_2025.npy
      wind_v_mask_2021_2025.npy
    raw/
      hab_source_audit_20261024/
    sequences/
      corrected/
    README.md
  docs/
    examples/
      hab_observations.template.csv
      hab_spec.template.json
    results/
      bloom_events/
      bloom_events_v2/
      bloom_events_v3/
      dashboard_interaction/
      hab_submission/
      improvement/
      operations/
      roadmap/
      sentinel_access/
      submission/
      upgrade/
      calibration.json -> ../../outputs/improvement/baseline/calibration.json
      data_quality.json -> ../../outputs/data_quality/report.json
      data_quality.md -> ../../outputs/data_quality/report.md
      dependency_check.txt
      evaluation.json -> ../../outputs/improvement/baseline/evaluation.json
      training_config.json -> ../../outputs/improvement/baseline/training_config.json
      training_history.csv -> ../../outputs/improvement/baseline/training_history.csv
    abstract.md
    bloom_inputs_v3.md
    dashboard_interaction_fix.md
    data_preparation.md
    final_report.md -> results/bloom_events_v3/final_report.md
    hab_implementation.md
    hab_source_search.md
    improvement_commands.md
    improvement_integration_checks.md
    improvement_methodology.md
    improvement_results.md -> ../outputs/improvement/final/evaluation.md
    integration_completion.md
    label_preprocessing.md
    limitations.md
    PACKAGING_INVENTORY.md
    post_freeze_improvements.md
    presentation_notes.md
    project_completion.md
    PROJECT_README.md
    real_data_interfaces.md
    roadmap_implementation.md
    runtime_workflows.md
    satellite_bloom_events.md
    satellite_bloom_events_v2.md
    sentinel_access_report.md
    SUBMISSION_HANDOFF.md
  experiments/
    common.py
    development.py
    diagnose.py
    false_alarms.py
    finalize.py
    freeze_baseline.py
    pilot_comparison.py
    regional.py
    report.py
    roadmap_evaluation.py
    satellite.py
    seasonal.py
    select.py
    train.py
  frontend/
    bloom.html
    bloom.js
    hab.css
    hab.html
    hab.js
    index.html
    script.js
    style.css
  models/
    frozen_chlorophyll -> ../outputs/improvement/final
    run_01 -> ../outputs/bloom_events/run_01
    run_02 -> ../outputs/bloom_events/run_02
    run_03 -> ../outputs/bloom_events/run_03
    best_convlstm.keras
    best_convlstm_backup.keras
    bloom_events.py
    convlstm.py
    corrected_convlstm.keras
    hab_convlstm.py
  operations/
    __init__.py
    acquisition.py
    baseline_audit.py
    forecast_views.py
    fusion.py
    geography.py
    hab_data_audit.py
    hindcast.py
    metadata.py
    notifications.py
    provenance.py
    sentinel_download.py
    sentinel_native.py
    sentinel_pilot.py
    update.py
    validation.py
  outputs/
    bloom_events/
      run_01/
      run_02/
      run_03/
    data_quality/
      daily_coverage.csv
      report.json
      report.md
    improvement/
      baseline/
      final/
      final_preprocessing/
      selection.json
    operations/
      actual_api_forecast.json
      api_metadata.json
      field_unavailable.json
      independent_unavailable.json -> field_unavailable.json
      integration_manifest.json
      notification_dry_run.json
    roadmap_20261015/
      reference/
    sentinel_access/
      access_probe.json
      aws_2024_probe.json
      aws_actual_2024_probe.json
      aws_catalogue_probe.json
      aws_download_probe.json
      aws_february_probe.json
      aws_native_files_probe.json
      aws_paths_probe.json
      download_result.json
    upgrade/
      balanced_sampling/
      calibration_2024.json
      calibration_2024.png
      data_audit.json
      freeze.json
      integration.json
      regional_support_2024.csv
      regional_support_2024.json
      sampling_comparison_2024.json
      strong_diagnostics_2024.json
  prediction/
    bloom_events.py
    bloom_events_v2.py
    bloom_service.py
    coverage.py
    decision.py
    evaluate.py
    evaluate_metrics.py
    evaluation.py
    final_results.py
    hab_evaluation.py
    hab_service.py
    metrics.py
    plot_confusion_matrix.py
    plot_training_history.py
    plots.py
    predict.py
    regional.py
    risk.py
    service.py
    tune_threshold.py
    visualize_prediction.py
  preprocessing/
    align_data.py
    bloom_events.py
    bloom_events_v2.py
    bloom_inputs_v3.py
    check_sentinel3_coverage.py
    check_sentinel3_january.py
    create_chl_climatology.py
    create_final_hab_labels.py
    create_hab_labels.py
    create_sequences.py
    data_utils.py
    download_currents_5years.py
    download_data.py
    download_era5_rainfall.py
    download_era5_wind.py
    download_sentinel3_5years.py
    download_sst_5years.py
    features.py
    hab_labels.py
    hab_sequences.py
    pipeline.py
    preprocess.py
    preprocess_5years_chlorophyll.py
    preprocess_5years_currents.py
    preprocess_5years_rainfall.py
    preprocess_5years_sst.py
    preprocess_5years_wind.py
    preprocess_month.py
    report_data_quality.py
    test_currents.py
    test_sentinel3_olci.py
    verify_all_data.py
    visualize_labels.py
    visualize_ocean_mask.py
  results -> docs/results
  submission/
    __init__.py
    __main__.py
    bootstrap.py
  tests/
    bloom_dashboard.mjs
    bloom_http_smoke.py
    dashboard_interaction.mjs
    dashboard_race.mjs
    dashboard_render.mjs
    dashboard_views.mjs
    hab_dashboard.mjs
    submission_smoke.py
    test_bloom_blocks_v3.py
    test_bloom_events.py
    test_bloom_events_v2.py
    test_bloom_inputs_v3.py
    test_end_to_end.py
    test_forecast_api.py
    test_hab_occurrence.py
    test_improvement.py
    test_label_preprocessing.py
    test_operations.py
    test_roadmap.py
    test_submission_portability.py
    test_training_pipeline.py
    test_upgrade.py
  training/
    data_generator.py
    losses.py
    train.py
    train_backup.py
    train_bloom_events.py
    train_bloom_inputs_v3.py
    train_hab.py
  upgrade/
    __init__.py
    audit.py
    calibration.py
    compare.py
    coverage_windows.py
    diagnostics.py
    field_records.py
    live.py
    modis.py
    notifications.py
    regional_experiment.py
    repair_month_strata.py
    report.py
    sampling.py
    satellite.py
    verify.py
  .env.example
  .gitignore
  config.py
  Makefile
  README.md
  requirements-dev.txt
  requirements-download-lock.txt
  requirements-download.txt
  requirements-lock.txt
  requirements-verified.txt
  requirements.txt
  sitecustomize.py
```

## Included, excluded and size

The complete folder is approximately **1946.14 MiB**; the ZIP is approximately **622.40 MiB**. Installed verification dependencies are outside the folder and are not bundled.

The folder exceeds 500 MB because the frozen run_03 feature cache is about **1,053 MiB**, the aligned environmental archive/observation masks require several hundred MiB, and earlier frozen label/fitting dependencies remain necessary for unchanged integrity checks. Safe internal aliases save about **91 MiB**. The large raw environmental downloads and cached prediction tensors are excluded.

To reduce this further, the largest candidate is the ~1 GiB run_03 feature cache: regenerate it from unchanged inputs using frozen statistics rather than shipping it. A second option is a validated lossless compressed-input cache. Neither was applied silently, because both need an additional storage adapter and exact-value verification. See `.submission/package.json`, `.submission/aliases.json` and `data/README.md`.


Excluded: Git history, installed environments (including the accidental `#`, `first`, `only`, `time` environments), private `.env`, caches, logs, progress/HTTP dumps, generated forecast bundles, large raw MODIS/ERA5/CMEMS/Sentinel downloads, old intermediate checkpoints and stored prediction tensors. Final saved evaluations, fitting metadata, scientific figures, selected checkpoints and source audits remain. See `.submission/exclusions.json` and [data instructions](data/README.md).

Small earlier artifacts explicitly referenced by frozen integrity manifests remain, including the balanced-sampling checkpoint. Small legacy root-model checkpoints are retained and flagged for provenance/legacy references; they are not the default or headline model. Existing optional Sentinel processed placeholders and earlier fitting directories are retained and flagged for compatibility. Safe duplicate paths are internal aliases only; arrays with distinct timestamp identities remain separate. See `.submission/aliases.json` and `.submission/relocation.json`.

The verification environment was freshly installed outside this folder, not copied from the original. No environment, package cache or raw download is included in the zip. `docs/PROJECT_README.md` retains the previous README as historical context; launch this copy using the instructions above.

### Historical workflows outside the packaged demo

Raw-product acquisition/preprocessing and the optional raw-source integration test need the excluded downloads and provider credentials described in `data/README.md`. Historical post-hoc scripts that expect saved prediction tensors require separately regenerated caches in a new experiment directory; those large caches are intentionally not bundled. The saved numerical reports and frozen-count audit remain available. No scientific test-set evaluation was repeated for packaging.
