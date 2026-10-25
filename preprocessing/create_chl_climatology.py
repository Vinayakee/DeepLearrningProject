"""Fit the existing monthly chlorophyll medians using training years only."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import numpy as np

if __package__:
    from . import data_utils as data
else:
    import data_utils as data

# Preserve the original minimum of 20 total observations. A monthly median
# additionally requires three observed days to avoid a one-day baseline.
MIN_OBSERVATIONS = 20
MIN_MONTHLY_OBSERVATIONS = 3


def calculate_climatology(chlorophyll, mask, dates, ocean_mask,
                          train_start=data.TRAIN_START, train_end=data.TRAIN_END,
                          min_observations=MIN_OBSERVATIONS,
                          min_monthly_observations=MIN_MONTHLY_OBSERVATIONS):
    data.validate_pair(chlorophyll, mask)
    if len(dates) != len(chlorophyll) or ocean_mask.shape != chlorophyll.shape[1:]:
        raise ValueError("Dates and ocean mask must match the chlorophyll grid.")
    if min_observations < 1 or min_monthly_observations < 1:
        raise ValueError("Minimum observation counts must be positive.")
    selected = data.training_selection(dates, train_start, train_end)
    months = data.month_numbers(dates)
    counts = np.zeros(chlorophyll.shape[1:], dtype=np.uint32)
    for index in np.flatnonzero(selected):
        counts += ((mask[index] > 0) & np.isfinite(chlorophyll[index]) & (chlorophyll[index] > 0))
    stable = (counts >= min_observations) & np.asarray(ocean_mask, dtype=bool)
    if not stable.any():
        raise ValueError("No ocean pixels meet the minimum training observation count.")

    climatology = np.full((12, *stable.shape), np.nan, dtype=np.float32)
    monthly_counts = np.zeros(climatology.shape, dtype=np.uint32)
    for month in range(1, 13):
        indices = np.flatnonzero(selected & (months == month))
        if not len(indices):
            continue
        values = chlorophyll[indices]
        observed = (mask[indices] > 0) & np.isfinite(values) & (values > 0)
        monthly_counts[month - 1] = observed.sum(axis=0)
        # All-NaN cells are expected outside the supported observation domain.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            median = np.nanmedian(np.where(observed, values, np.nan), axis=0)
        median[~stable | (monthly_counts[month - 1] < min_monthly_observations)] = np.nan
        climatology[month - 1] = median
    if not np.isfinite(climatology).any():
        raise ValueError("No monthly climatology meets the observation requirements.")
    return {
        "monthly_chl_climatology_2021_2025.npy": climatology,
        "monthly_chl_observation_count.npy": monthly_counts,
        "stable_ocean_mask.npy": stable.astype(np.uint8),
        "ocean_support_mask.npy": np.asarray(ocean_mask, dtype=np.uint8),
        "pixel_observation_count.npy": counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=data.PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=data.LABEL_DIR)
    parser.add_argument("--train-start", default=data.TRAIN_START)
    parser.add_argument("--train-end", default=data.TRAIN_END)
    parser.add_argument("--min-observations", type=int, default=MIN_OBSERVATIONS)
    parser.add_argument("--min-monthly-observations", type=int, default=MIN_MONTHLY_OBSERVATIONS)
    args = parser.parse_args()
    data.validate_output_directory(args.processed_dir, args.output_dir)
    dates = data.load_dates(args.processed_dir)
    _, _, shape = data.load_grid(args.processed_dir, dates)
    chl, mask = data.load_pair(args.processed_dir, "chlorophyll", shape)
    selected = data.training_selection(dates, args.train_start, args.train_end)
    ocean = data.derive_ocean_support(args.processed_dir, shape, selected)
    arrays = calculate_climatology(chl, mask, dates, ocean, args.train_start, args.train_end,
                                   args.min_observations, args.min_monthly_observations)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(args.output_dir / name, array)
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(args.processed_dir.resolve()),
        "source_fingerprint": data.source_fingerprint(args.processed_dir),
        "train_start": args.train_start, "train_end": args.train_end,
        "fitted_first_date": str(dates[selected][0]), "fitted_last_date": str(dates[selected][-1]),
        "training_days": int(selected.sum()), "shape": list(shape),
        "min_observations": args.min_observations,
        "min_monthly_observations": args.min_monthly_observations,
        "ocean_mask_method": "Union of finite, masked SST/current cells in training years; availability proxy, not an authoritative coastline.",
        "array_sha256": {name: data.file_sha256(args.output_dir / name) for name in arrays},
    }
    (args.output_dir / "climatology_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Training-only climatology: {metadata['fitted_first_date']} to {metadata['fitted_last_date']} ({selected.sum()} days)")
    print(f"Ocean support pixels: {ocean.sum()}; stable observed ocean pixels: {arrays['stable_ocean_mask.npy'].sum()}")
    for month, values in enumerate(arrays["monthly_chl_climatology_2021_2025.npy"], 1):
        print(f"Month {month:02d}: {np.isfinite(values).sum()} supported climatology pixels")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
