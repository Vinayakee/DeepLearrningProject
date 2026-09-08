import numpy as np
import os
from datetime import datetime


# ============================================================
# SETTINGS
# ============================================================

CHL_FILE = "data/processed/chlorophyll_2021_2025.npy"
MASK_FILE = "data/processed/chlorophyll_mask_2021_2025.npy"
DATES_FILE = "data/processed/dates_2021_2025.txt"

CLIM_FILE = (
    "data/processed/"
    "monthly_chl_climatology_2021_2025.npy"
)

STABLE_MASK_FILE = (
    "data/processed/stable_ocean_mask.npy"
)

OUTPUT_DIR = "data/processed"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("FINAL HAB LABEL GENERATION")
print("=" * 65)

chlorophyll = np.load(CHL_FILE)

mask = np.load(MASK_FILE).astype(bool)

climatology = np.load(CLIM_FILE)

stable_mask = np.load(
    STABLE_MASK_FILE
).astype(bool)


print("CHL shape:", chlorophyll.shape)
print("Climatology shape:", climatology.shape)


# ============================================================
# LOAD DATES
# ============================================================

with open(DATES_FILE, "r") as f:
    date_strings = [
        line.strip()
        for line in f
        if line.strip()
    ]

dates = [
    datetime.strptime(
        d,
        "%Y-%m-%d"
    )
    for d in date_strings
]


# ============================================================
# CREATE MONTH ARRAY
# ============================================================

months = np.array([
    d.month
    for d in dates
])


# ============================================================
# CREATE ANOMALY ARRAY
# ============================================================

anomaly = np.full(
    chlorophyll.shape,
    np.nan,
    dtype=np.float32
)


# ============================================================
# CALCULATE MONTHLY RELATIVE ANOMALY
# ============================================================

print()
print("CALCULATING MONTHLY CHLOROPHYLL ANOMALY")
print("----------------------------------------")


for month in range(1, 13):

    indices = np.where(
        months == month
    )[0]

    monthly_clim = climatology[
        month - 1
    ]

    for index in indices:

        observed = (
            mask[index]
            &
            stable_mask
            &
            np.isfinite(chlorophyll[index])
            &
            np.isfinite(monthly_clim)
            &
            (chlorophyll[index] > 0)
            &
            (monthly_clim > 0)
        )

        anomaly[index][observed] = (
            (
                chlorophyll[index][observed]
                -
                monthly_clim[observed]
            )
            /
            monthly_clim[observed]
        )

    print(
        f"Month {month:02d}: "
        f"{len(indices)} days processed"
    )


# ============================================================
# CALCULATE GLOBAL ANOMALY THRESHOLDS
# ============================================================

valid_anomaly = anomaly[
    np.isfinite(anomaly)
]

print()
print("=" * 65)
print("ANOMALY STATISTICS")
print("=" * 65)

print(
    "Valid anomaly values:",
    len(valid_anomaly)
)

print(
    "Minimum:",
    np.percentile(valid_anomaly, 1)
)

print(
    "Median:",
    np.percentile(valid_anomaly, 50)
)

print(
    "75th percentile:",
    np.percentile(valid_anomaly, 75)
)

print(
    "90th percentile:",
    np.percentile(valid_anomaly, 90)
)

print(
    "Maximum:",
    np.percentile(valid_anomaly, 99)
)


# ============================================================
# MONTHLY THRESHOLDS
# ============================================================

p75_monthly = np.zeros(12)

p90_monthly = np.zeros(12)


print()
print("MONTHLY THRESHOLDS")
print("------------------")


for month in range(1, 13):

    indices = np.where(
        months == month
    )[0]

    values = anomaly[
        indices
    ]

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        p75_monthly[month - 1] = np.nan
        p90_monthly[month - 1] = np.nan

        print(
            f"Month {month:02d}: "
            "NO DATA"
        )

        continue

    p75 = np.percentile(
        values,
        75
    )

    p90 = np.percentile(
        values,
        90
    )

    p75_monthly[month - 1] = p75
    p90_monthly[month - 1] = p90

    print(
        f"Month {month:02d}: "
        f"P75={p75:.3f}, "
        f"P90={p90:.3f}"
    )


# ============================================================
# CREATE HAB LABELS
# ============================================================

labels = np.full(
    anomaly.shape,
    255,
    dtype=np.uint8
)


print()
print("CREATING HAB LABELS")
print("-------------------")


for month in range(1, 13):

    indices = np.where(
        months == month
    )[0]

    p75 = p75_monthly[
        month - 1
    ]

    p90 = p90_monthly[
        month - 1
    ]

    if not np.isfinite(p75):
        continue

    for index in indices:

        valid = np.isfinite(
            anomaly[index]
        )

        # Normal
        labels[index][
            valid &
            (anomaly[index] < p75)
        ] = 0

        # Elevated bloom risk
        labels[index][
            valid &
            (anomaly[index] >= p75) &
            (anomaly[index] < p90)
        ] = 1

        # Strong bloom risk
        labels[index][
            valid &
            (anomaly[index] >= p90)
        ] = 2


# ============================================================
# SAVE FILES
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "chlorophyll_anomaly_2021_2025.npy"
    ),
    anomaly
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "hab_labels_2021_2025.npy"
    ),
    labels
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "hab_p75_monthly.npy"
    ),
    p75_monthly
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "hab_p90_monthly.npy"
    ),
    p90_monthly
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("=" * 65)
print("FINAL LABEL DISTRIBUTION")
print("=" * 65)


valid_labels = labels[
    labels != 255
]

unique, counts = np.unique(
    valid_labels,
    return_counts=True
)


for label, count in zip(
    unique,
    counts
):

    percentage = (
        count /
        len(valid_labels) *
        100
    )

    if label == 0:
        name = "Normal"

    elif label == 1:
        name = "Elevated Risk"

    elif label == 2:
        name = "Strong Bloom Risk"

    else:
        name = "Unknown"

    print(
        f"Class {label} "
        f"({name}): "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


print()
print(
    "Missing:",
    np.sum(labels == 255)
)


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 65)
print("FINAL CHECK")
print("=" * 65)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Anomaly shape:",
    anomaly.shape
)

print()
print("Created:")

print(
    "chlorophyll_anomaly_2021_2025.npy"
)

print(
    "hab_labels_2021_2025.npy"
)

print(
    "hab_p75_monthly.npy"
)

print(
    "hab_p90_monthly.npy"
)

print()
print("FINAL HAB LABEL GENERATION COMPLETE!")