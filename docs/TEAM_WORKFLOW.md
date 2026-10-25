# Team workflow

Use `codex/submission-run03` until its pull request is reviewed and merged. Each teammate creates a new branch for a change, commits code/documentation, pushes, and opens a pull request. Do not force-push or rewrite the existing project history.

The 2021–2025 runtime arrays and frozen checkpoint bundle are distributed through the versioned GitHub Release. Run `python scripts/setup_runtime.py` after cloning to restore them without replacing your working code. Generated forecasts, local environments, credentials and array caches remain ignored by Git.

The headline is **satellite bloom-event forecast run_03**; the frozen chlorophyll baseline and runs_01–02 are earlier results. Verified-observation workflow remains untrained/not configured. Do not retune the model against previously inspected 2025 results or modify frozen scientific evidence to make an experiment pass.

CI is not enabled. A manual-only `workflow_dispatch` template is retained in `docs/ci/checks.yml.example`, so publishing a branch starts no runner. Its scientific integration checks require the release runtime bundle; synthetic software tests are not scientific performance evidence.
