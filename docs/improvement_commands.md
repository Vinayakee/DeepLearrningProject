# Running and reproducing the improvement work

Use the existing repository and aligned real dataset. Do not create a replacement project. Large observations, fitted labels and checkpoints are not committed to Git.

## Run the completed local system

```bash
source .venv/bin/activate
python -m experiments.finalize verify
python prediction/predict.py --input-end-date 2024-02-15 --historical-demo
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `/dashboard/` on that server and enable Historical demonstration. The dataset ends in 2025; current warnings remain unavailable. A missing-input request returns explicit insufficient data, not a normal-risk prediction.

```bash
python -m unittest discover -s tests -v
python -m pip check
```

The active deployment pointer is `outputs/improvement/final/deployment.json`. It selects the final checkpoint, preprocessing, label directory and report directory. `HAB_USE_BASELINE=1` selects the preserved original working baseline for inspection. `HAB_MODEL_PATH`, `HAB_RUN_DIR`, `HAB_SEQUENCE_DIR` and `HAB_LABEL_DIR` are explicit path overrides; all fitted artifacts must match.

## Completed experiments: inspect, do not repeat

This local project already contains the completed experiment directories. Training commands refuse to overwrite them. Final selection, freeze and test-start records prevent repeating or retuning the final evaluation. Read the saved JSON/Markdown instead. `python -m prediction.evaluation` refuses to retune an active frozen final run.

The original checkpoint remains `models/corrected_convlstm.keras`, with an additional protected snapshot under `outputs/improvement/baseline`. Legacy `best_convlstm.keras` checkpoints are also preserved. Baseline training cannot overwrite the frozen original checkpoint; smoke output remains separate.

## Reproduction sequence for a separate artifact set

These commands document the actual experiment order, rather than instructions to rerun completed local experiments. A separate artifact set requires the existing real aligned observations and a completed baseline. A code-only checkout must first acquire/transfer the real dataset and fit corrected labels/preprocessing as described in `data_preparation.md` and `label_preprocessing.md`. Do not replace absent data with random arrays.

```bash
python -m experiments.freeze_baseline
python -m experiments.diagnose
python -m experiments.train --name inverse --weight-power 1
python -m experiments.train --name intermediate --weight-power 0.75
python -m experiments.development weights
python -m experiments.development rules
python -m experiments.seasonal --declare
python -m experiments.train --name seasonal_root --weight-power 0.5 --seasonal
python -m experiments.seasonal
python -m experiments.development rules --names seasonal_root
```

Calendar features passed the declared retention criterion in this round, so their combination was evaluated only after measuring the individual effects:

```bash
python -m experiments.train --name seasonal_combination --weight-power 0.75 --seasonal
python -m experiments.development rules --names seasonal_combination
python -m experiments.satellite
python -m experiments.regional --names baseline inverse intermediate seasonal_root seasonal_combination
python -m experiments.false_alarms
python -m experiments.select
python -m experiments.finalize train
python -m experiments.finalize freeze
python -m experiments.finalize test
python -m experiments.finalize activate
```

The selected decision boundaries are fixed before final retraining. Final inference and training both use `EnvironmentalPipeline` with the same fitted normalization, flags and optional calendar encoding. The final test command evaluates the frozen configuration once; it refuses a second start. No rule/weight/feature adjustment follows that result.

To train an additional development experiment without overwriting a completed checkpoint, use a new name, for example:

```bash
python -m experiments.train --name additional_development --weight-power 0.75 --seasonal
```

This trains on 2021–2023 and validates on 2024 only; it does not replace production or access 2025. Do not use it to tune against the already reported test results.

Reproducibility records include the seed, optimizer, class weights, callbacks, split dates, source hashes, training history and checkpoint hash. Hardware/library differences may change floating-point results; exact package versions are in the existing lock files.

2025 was inspected before this improvement request. It is a previously inspected holdout, not an untouched test. Future independent generalization assessment needs a new period or external verified HAB observations.
