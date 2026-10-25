# Satellite bloom-event forecast — Kerala coast

**Headline: run_03 (V3 causal inputs).** Earlier results are the frozen chlorophyll-anomaly baseline and runs_01–02. The verified-observation workflow remains **untrained / not configured**.

This repository versions source, tests, reports and small configuration files. The complete frozen processed data/checkpoint bundle is attached to the GitHub Release; it is not committed as large Git blobs. Existing repository history is preserved.

## Get the complete working project

Verified on **macOS Apple Silicon with Python 3.13**. Node.js **22+** is used for the JavaScript regression checks; no npm packages are needed.

```bash
git clone --branch codex/submission-run03 https://github.com/Vinayakee/DeepLearrningProject.git
cd DeepLearrningProject
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/setup_runtime.py
python -m submission serve --port 8020
```

Open **http://127.0.0.1:8020/dashboard/bloom.html**. Run_03 is selected by default. Keep input end **2024-02-15** and **Historical demonstration** checked, generate, switch +1…+5 and export CSV. Run_01/run_02 are also selectable. Frozen chlorophyll baseline: `/dashboard/`; untrained field-verified workflow: `/dashboard/hab.html`.

The setup command downloads the versioned ZIP, validates its SHA256, and restores only data/checkpoints/output metadata. It does not replace working code or refit anything. Alternatively download the release ZIP manually and run `python scripts/setup_runtime.py --archive /path/to/HAB-Early-Warning-Submission.zip`. The complete ZIP can also be extracted and run directly using its own README.

## Results — unchanged frozen evidence

| Workflow | Finding |
|---|---|
| **run_03 / V3** | 2024 mean pixel F1 **0.415829**. Single 2025 evaluation: mean observed-target F1 **0.3798**, IoU/CSI **0.2345**; matched pixel/block F1 and IoU exceed persistence at all five horizons. Calibration rejected; scores remain raw. |
| run_02 | Observed-target 2025 F1 approximately **0.324–0.341**; persistence has higher matched F1/IoU at all horizons. |
| run_01 | Zero positive validation/test targets; recall undefined. |
| Frozen chlorophyll baseline | Earlier three-class result and original evaluation preserved. |

2025 was previously inspected; rolling pixel forecasts are correlated. V2 dynamics did not improve V1; the winning cumulative V3 does not isolate their benefit/removal. Block aggregation is not independent biological event detection. Satellite-derived labels do not establish harmful species, toxicity or operational warning reliability.

[Submission handoff](docs/SUBMISSION_HANDOFF.md) · [Final report](docs/final_report.md) · [Full pixel/block results](docs/results/bloom_events_v3/evaluation.md) · [2024 ablation](docs/results/bloom_events_v3/candidate_comparison.md) · [Limitations](docs/limitations.md) · [Data and source audit](data/README.md)

## Test and audit

After restoring the runtime bundle:

```bash
python -m submission check
python -m submission audit
```

The verified submission ran **133 tests: 132 passed, one optional raw-MODIS source test skipped** because raw downloads are deliberately excluded. Synthetic software tests are not scientific accuracy results. Saved scientific test-set evaluations are not repeated by setup or these checks.

GitHub Actions is **not enabled**, avoiding automatic runner usage on pushes. A manual-only template is retained in `docs/ci/checks.yml.example` for later opt-in use. A local Python environment is not included in Git or the release.

## Team collaboration

Create a separate branch for each change, push it and open a pull request. Until the submission PR is merged, use `codex/submission-run03` as shown above. Do not force-push or overwrite frozen results. See [team workflow](docs/TEAM_WORKFLOW.md).

Runtime inputs, generated forecasts, environments and credentials are ignored by Git. Do not `git add -f` the 1 GiB feature cache. Local machine-wide audit listings are deliberately excluded from the public release; all scientific data/checkpoints/reports remain included.

## Layout

- `preprocessing/`, `training/`, `prediction/`, `backend/`: existing import layout, scientific workflow and API.
- `frontend/`: dashboard.
- `submission/`, `.submission/`, `sitecustomize.py`: portability launcher and frozen path/timestamp metadata.
- `scripts/setup_runtime.py`: checksum-verified release data setup.
- `docs/`, `docs/results/`: reports, comparison tables, presentation notes and provenance.
- `tests/`: software and integration checks.
- `data/`, `outputs/`, model binaries: restored from the release; not large tracked Git objects.

No original protected project file is moved or edited by this team repository setup.
