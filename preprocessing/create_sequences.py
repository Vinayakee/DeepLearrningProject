import os
import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/sequences"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Past 5 days → future 5 days
INPUT_DAYS = 5
OUTPUT_DAYS = 5

# ============================================================
# LOAD DATES
# ============================================================

dates = np.loadtxt(
    os.path.join(
        PROCESSED_DIR,
        "dates_2021_2025.txt"
    ),
    dtype=str
)

dates = pd.to_datetime(dates)

print("=" * 70)
print("5-DAY CONVLSTM SEQUENCE PREPARATION")
print("=" * 70)

print(
    f"Date range: {dates[0].date()} → {dates[-1].date()}"
)

print(
    f"Total days: {len(dates)}"
)

# ============================================================
# LOAD ENVIRONMENTAL DATA
# ============================================================

print("\nLoading environmental datasets...")

data_files = {
    "chlorophyll": "chlorophyll_2021_2025.npy",
    "sst": "sst_2021_2025.npy",
    "wind_u": "wind_u_2021_2025.npy",
    "wind_v": "wind_v_2021_2025.npy",
    "rainfall": "rainfall_2021_2025.npy",
    "current_u": "current_u_2021_2025.npy",
    "current_v": "current_v_2021_2025.npy",
}

data = {}

for name, filename in data_files.items():

    path = os.path.join(
        PROCESSED_DIR,
        filename
    )

    print(f"Loading {name}...")

    data[name] = np.load(
        path,
        mmap_mode="r"
    )

    print(
        f"  Shape: {data[name].shape}"
    )

# ============================================================
# LOAD OBSERVATION MASKS
# ============================================================

print("\nLoading observation masks...")

mask_files = {
    "chlorophyll": "chlorophyll_mask_2021_2025.npy",
    "sst": "sst_mask_2021_2025.npy",
    "wind_u": "wind_u_mask_2021_2025.npy",
    "wind_v": "wind_v_mask_2021_2025.npy",
    "rainfall": "rainfall_mask_2021_2025.npy",
    "current_u": "current_u_mask_2021_2025.npy",
    "current_v": "current_v_mask_2021_2025.npy",
}

masks = {}

for name, filename in mask_files.items():

    path = os.path.join(
        PROCESSED_DIR,
        filename
    )

    masks[name] = np.load(
        path,
        mmap_mode="r"
    )

    print(
        f"  {name}: {masks[name].shape}"
    )

# ============================================================
# LOAD HAB LABELS
# ============================================================

print("\nLoading HAB labels...")

labels = np.load(
    os.path.join(
        PROCESSED_DIR,
        "hab_labels_2021_2025.npy"
    ),
    mmap_mode="r"
)

print(
    "HAB labels:",
    labels.shape
)

# ============================================================
# TRAIN / VALIDATION / TEST PERIODS
# ============================================================

# Chronological split
#
# TRAIN      : 2021-2023
# VALIDATION : 2024
# TEST       : 2025
#

train_start = pd.Timestamp("2021-01-01")
train_end = pd.Timestamp("2023-12-31")

val_start = pd.Timestamp("2024-01-01")
val_end = pd.Timestamp("2024-12-31")

test_start = pd.Timestamp("2025-01-01")
test_end = pd.Timestamp("2025-12-31")

# ============================================================
# CALCULATE TRAINING MEDIANS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING TRAINING-ONLY NORMALIZATION VALUES")
print("=" * 70)

print(
    "Important: validation/test data are NOT used here."
)

training_indices = np.where(
    (dates >= train_start)
    & (dates <= train_end)
)[0]

print(
    "Training days:",
    len(training_indices)
)

training_medians = {}

for name in data:

    print(
        f"\nCalculating median for {name}..."
    )

    # We cannot load the entire 3-D array into memory.
    # Sample every 7th training day.

    sample_indices = training_indices[::7]

    values = []

    for idx in sample_indices:

        day = np.asarray(
            data[name][idx],
            dtype=np.float32
        )

        valid = day[
            np.isfinite(day)
        ]

        if len(valid) > 0:

            # Sample at most 10,000 values/day
            if len(valid) > 10000:

                rng = np.random.default_rng(42)

                selected = rng.choice(
                    valid,
                    size=10000,
                    replace=False
                )

                values.extend(
                    selected.tolist()
                )

            else:

                values.extend(
                    valid.tolist()
                )

    if len(values) == 0:

        median_value = 0.0

    else:

        median_value = float(
            np.median(values)
        )

    training_medians[name] = median_value

    print(
        f"{name} training median: "
        f"{median_value:.6f}"
    )

# ============================================================
# SAVE NORMALIZATION VALUES
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "training_medians.npy"
    ),
    training_medians
)

print(
    "\n✓ Training medians saved."
)

# ============================================================
# CREATE SEQUENCE INDEXES
# ============================================================

print("\n" + "=" * 70)
print("CREATING SEQUENCE INDEXES")
print("=" * 70)

sequence_rows = []

# The final target day must be:
#
# start + 5 input days + 5 future days - 1
#
# Example:
#
# Input:
# Jan 1 → Jan 5
#
# Target:
# Jan 6 → Jan 10
#

max_start = (
    len(dates)
    - INPUT_DAYS
    - OUTPUT_DAYS
    + 1
)

for start_idx in range(max_start):

    input_start = start_idx
    input_end = (
        start_idx
        + INPUT_DAYS
        - 1
    )

    target_start = (
        start_idx
        + INPUT_DAYS
    )

    target_end = (
        target_start
        + OUTPUT_DAYS
        - 1
    )

    input_start_date = dates[
        input_start
    ]

    input_end_date = dates[
        input_end
    ]

    target_start_date = dates[
        target_start
    ]

    target_end_date = dates[
        target_end
    ]

    # --------------------------------------------------------
    # Determine split
    # --------------------------------------------------------

    if (
        input_start_date >= train_start
        and target_end_date <= train_end
    ):

        split = "train"

    elif (
        input_start_date >= val_start
        and target_end_date <= val_end
    ):

        split = "validation"

    elif (
        input_start_date >= test_start
        and target_end_date <= test_end
    ):

        split = "test"

    else:

        # This removes windows crossing
        # train/validation/test boundaries.

        continue

    # --------------------------------------------------------
    # Check future label availability
    # --------------------------------------------------------

    target_valid_pixels = []

    for day_idx in range(
        target_start,
        target_end + 1
    ):

        day_labels = np.asarray(
            labels[day_idx]
        )

        valid_count = np.sum(
            day_labels != 255
        )

        target_valid_pixels.append(
            int(valid_count)
        )

    total_target_valid = sum(
        target_valid_pixels
    )

    # --------------------------------------------------------
    # Skip completely empty targets
    # --------------------------------------------------------

    if total_target_valid == 0:

        continue

    sequence_rows.append({

        "split": split,

        "input_start_idx": input_start,

        "input_end_idx": input_end,

        "target_start_idx": target_start,

        "target_end_idx": target_end,

        "input_start": input_start_date.strftime(
            "%Y-%m-%d"
        ),

        "input_end": input_end_date.strftime(
            "%Y-%m-%d"
        ),

        "target_start": target_start_date.strftime(
            "%Y-%m-%d"
        ),

        "target_end": target_end_date.strftime(
            "%Y-%m-%d"
        ),

        "target_valid_pixels": total_target_valid,

    })

# ============================================================
# SAVE SEQUENCE INDEX
# ============================================================

sequence_df = pd.DataFrame(
    sequence_rows
)

sequence_path = os.path.join(
    OUTPUT_DIR,
    "sequence_index.csv"
)

sequence_df.to_csv(
    sequence_path,
    index=False
)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE SUMMARY")
print("=" * 70)

print(
    f"Total sequences: "
    f"{len(sequence_df)}"
)

for split in [
    "train",
    "validation",
    "test"
]:

    count = np.sum(
        sequence_df["split"] == split
    )

    print(
        f"{split.capitalize()} sequences: "
        f"{count}"
    )

# ============================================================
# EXAMPLE
# ============================================================

if len(sequence_df) > 0:

    print("\n" + "=" * 70)
    print("FIRST SEQUENCE")
    print("=" * 70)

    first = sequence_df.iloc[0]

    print(
        f"Input : "
        f"{first['input_start']} → "
        f"{first['input_end']}"
    )

    print(
        f"Target: "
        f"{first['target_start']} → "
        f"{first['target_end']}"
    )

    print(
        f"Split : "
        f"{first['split']}"
    )

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE PREPARATION COMPLETE")
print("=" * 70)

print(
    "✓ 5-day input windows created"
)

print(
    "✓ 5-day prediction windows created"
)

print(
    "✓ Chronological train/validation/test split"
)

print(
    "✓ No temporal leakage across splits"
)

print(
    "✓ Missing target pixels preserved"
)

print(
    "✓ Training-only normalization values calculated"
)

print(
    f"✓ Saved: {sequence_path}"
)

print("=" * 70)