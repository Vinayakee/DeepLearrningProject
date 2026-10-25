"""Apply training-fitted monthly anomaly thresholds to all observed dates."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import data_utils as data
else:
    import data_utils as data


def calculate_labels(chlorophyll, mask, dates, climatology, stable_mask,
                     train_start=data.TRAIN_START, train_end=data.TRAIN_END):
    data.validate_pair(chlorophyll, mask)
    if (len(dates) != len(chlorophyll) or climatology.shape != (12, *chlorophyll.shape[1:])
            or stable_mask.shape != chlorophyll.shape[1:]):
        raise ValueError("Dates, climatology, and stable mask must match the chlorophyll grid.")
    selected = data.training_selection(dates, train_start, train_end)
    months = data.month_numbers(dates)
    anomaly = np.full(chlorophyll.shape, np.nan, dtype=np.float32)
    p75 = np.full(12, np.nan, dtype=np.float64)
    p90 = np.full(12, np.nan, dtype=np.float64)
    labels = np.full(chlorophyll.shape, 255, dtype=np.uint8)

    for month in range(1, 13):
        baseline = climatology[month - 1]
        for index in np.flatnonzero(months == month):
            observed = ((mask[index] > 0) & np.asarray(stable_mask, dtype=bool)
                        & np.isfinite(chlorophyll[index]) & (chlorophyll[index] > 0)
                        & np.isfinite(baseline) & (baseline > 0))
            # Preserve the original relative anomaly formula and missing values.
            anomaly[index][observed] = (chlorophyll[index][observed] - baseline[observed]) / baseline[observed]
        training_values = anomaly[selected & (months == month)]
        training_values = training_values[np.isfinite(training_values)]
        if not len(training_values):
            continue
        p75[month - 1], p90[month - 1] = np.percentile(training_values, [75, 90])
        for index in np.flatnonzero(months == month):
            valid = np.isfinite(anomaly[index])
            labels[index][valid & (anomaly[index] < p75[month - 1])] = 0
            labels[index][valid & (anomaly[index] >= p75[month - 1]) & (anomaly[index] < p90[month - 1])] = 1
            labels[index][valid & (anomaly[index] >= p90[month - 1])] = 2
    if not np.isfinite(p75).any():
        raise ValueError("No training anomalies available to fit label thresholds.")
    return anomaly, labels, p75, p90


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=data.PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=data.LABEL_DIR,
                        help="Directory containing training-only climatology; also receives labels.")
    args = parser.parse_args()
    data.validate_output_directory(args.processed_dir, args.output_dir)
    metadata = json.loads((args.output_dir / "climatology_metadata.json").read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported climatology metadata; regenerate training-only climatology.")
    if (metadata["source_directory"] != str(args.processed_dir.resolve())
            or metadata["source_fingerprint"] != data.source_fingerprint(args.processed_dir)):
        raise ValueError("Climatology inputs have changed; regenerate climatology before labels.")
    for name, expected_hash in metadata["array_sha256"].items():
        if data.file_sha256(args.output_dir / name) != expected_hash:
            raise ValueError(f"Climatology artifact {name} has changed; regenerate climatology.")
    dates = data.load_dates(args.processed_dir)
    _, _, shape = data.load_grid(args.processed_dir, dates)
    if metadata["shape"] != list(shape):
        raise ValueError("Climatology metadata does not match the current grid and dates.")
    chl, mask = data.load_pair(args.processed_dir, "chlorophyll", shape)
    climatology = np.load(args.output_dir / "monthly_chl_climatology_2021_2025.npy", allow_pickle=False)
    stable_mask = np.load(args.output_dir / "stable_ocean_mask.npy", allow_pickle=False)
    anomaly, labels, p75, p90 = calculate_labels(chl, mask, dates, climatology, stable_mask,
                                               metadata["train_start"], metadata["train_end"])
    arrays = {"chlorophyll_anomaly_2021_2025.npy": anomaly, "hab_labels_2021_2025.npy": labels,
              "hab_p75_monthly.npy": p75, "hab_p90_monthly.npy": p90}
    for name, array in arrays.items():
        np.save(args.output_dir / name, array)
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_definition": "Chlorophyll anomaly proxy: 0 below training monthly P75, 1 from P75 to P90, 2 at/above P90; 255 unobserved or unsupported. Does not establish species, toxicity, or confirmed HAB events.",
        "train_start": metadata["train_start"], "train_end": metadata["train_end"],
        "climatology_metadata_sha256": data.file_sha256(args.output_dir / "climatology_metadata.json"),
        "array_sha256": {name: data.file_sha256(args.output_dir / name) for name in arrays},
        "monthly_thresholds": [{"month": month + 1,
                                "p75": float(p75[month]) if np.isfinite(p75[month]) else None,
                                "p90": float(p90[month]) if np.isfinite(p90[month]) else None} for month in range(12)],
        "yearly_labels": {},
    }
    for year in np.unique(dates.astype("datetime64[Y]")):
        values = labels[dates.astype("datetime64[Y]") == year]
        counts = {str(class_id): int((values == class_id).sum()) for class_id in (0, 1, 2, 255)}
        report["yearly_labels"][str(year)] = counts
        print(f"{year}: {counts}")
    (args.output_dir / "label_metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Training-fitted labels saved to {args.output_dir}; existing training/model artifacts remain available.")


if __name__ == "__main__":
    main()
