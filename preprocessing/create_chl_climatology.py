import numpy as np
import os
from datetime import datetime


# ============================================================
# SETTINGS
# ============================================================

CHL_FILE = "data/processed/chlorophyll_2021_2025.npy"
MASK_FILE = "data/processed/chlorophyll_mask_2021_2025.npy"
DATES_FILE = "data/processed/dates_2021_2025.txt"

OUTPUT_DIR = "data/processed"

# Minimum number of observations required for a pixel
MIN_OBSERVATIONS = 20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("5-YEAR CHLOROPHYLL CLIMATOLOGY")
print("=" * 65)

print("\nLoading chlorophyll...")

chlorophyll = np.load(CHL_FILE)

print("Shape:", chlorophyll.shape)

mask = np.load(MASK_FILE)

print("Mask shape:", mask.shape)


# ============================================================
# LOAD DATES
# ============================================================

with open(DATES_FILE, "r") as f:
    date_strings = [line.strip() for line in f if line.strip()]

dates = [
    datetime.strptime(d, "%Y-%m-%d")
    for d in date_strings
]

print("Number of dates:", len(dates))


# ============================================================
# VERIFY
# ============================================================

if len(dates) != chlorophyll.shape[0]:
    raise ValueError(
        "Number of dates does not match CHL time dimension."
    )


# ============================================================
# MONTH ARRAY
# ============================================================

months = np.array([
    d.month
    for d in dates
])

print("Months identified.")


# ============================================================
# PERMANENT OBSERVATION COUNT
# ============================================================

observation_count = np.sum(mask, axis=0)

stable_ocean_mask = (
    observation_count >= MIN_OBSERVATIONS
)

print()
print("OBSERVATION COVERAGE")
print("--------------------")

print(
    "Total grid pixels:",
    observation_count.size
)

print(
    "Stable pixels:",
    np.sum(stable_ocean_mask)
)

print(
    "Stable pixel percentage:",
    np.mean(stable_ocean_mask) * 100
)


# ============================================================
# CREATE MONTHLY CLIMATOLOGY
# ============================================================

height = chlorophyll.shape[1]
width = chlorophyll.shape[2]

monthly_climatology = np.full(
    (12, height, width),
    np.nan,
    dtype=np.float32
)

monthly_observation_count = np.zeros(
    (12, height, width),
    dtype=np.uint16
)


print()
print("CREATING MONTHLY CLIMATOLOGY")
print("----------------------------")


for month in range(1, 13):

    indices = np.where(months == month)[0]

    print(
        f"Month {month:02d}: "
        f"{len(indices)} daily observations"
    )

    month_data = chlorophyll[indices]

    month_mask = mask[indices].astype(bool)

    # Count observations for each pixel
    counts = np.sum(
        month_mask,
        axis=0
    )

    monthly_observation_count[month - 1] = counts

    # Median is robust to extreme bloom values
    climatology = np.nanmedian(
        np.where(
            month_mask,
            month_data,
            np.nan
        ),
        axis=0
    )

    # Only retain stable pixels
    climatology[
        ~stable_ocean_mask
    ] = np.nan

    monthly_climatology[
        month - 1
    ] = climatology.astype(np.float32)


# ============================================================
# SAVE CLIMATOLOGY
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "monthly_chl_climatology_2021_2025.npy"
    ),
    monthly_climatology
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "monthly_chl_observation_count.npy"
    ),
    monthly_observation_count
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "stable_ocean_mask.npy"
    ),
    stable_ocean_mask.astype(np.uint8)
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "pixel_observation_count.npy"
    ),
    observation_count.astype(np.uint16)
)


# ============================================================
# PRINT STATISTICS
# ============================================================

print()
print("=" * 65)
print("CLIMATOLOGY STATISTICS")
print("=" * 65)

for month in range(12):

    values = monthly_climatology[month]

    valid = values[np.isfinite(values)]

    if len(valid) == 0:

        print(
            f"Month {month + 1:02d}: "
            "No valid climatology"
        )

        continue

    print(
        f"Month {month + 1:02d}: "
        f"min={np.min(valid):.3f}, "
        f"median={np.median(valid):.3f}, "
        f"max={np.max(valid):.3f}"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 65)
print("FILES CREATED")
print("=" * 65)

print(
    "monthly_chl_climatology_2021_2025.npy"
)

print(
    "monthly_chl_observation_count.npy"
)

print(
    "stable_ocean_mask.npy"
)

print(
    "pixel_observation_count.npy"
)

print()
print("CLIMATOLOGY COMPLETE!")