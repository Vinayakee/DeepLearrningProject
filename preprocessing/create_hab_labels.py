import numpy as np
import os

# --------------------------------------------------
# FILE PATH
# --------------------------------------------------

CHL_FILE = "data/processed/chlorophyll_oct2024.npy"
MASK_FILE = "data/processed/chlorophyll_mask_oct2024.npy"

OUTPUT_LABEL = "data/processed/hab_labels_oct2024.npy"
OUTPUT_ANOMALY = "data/processed/chlorophyll_anomaly_oct2024.npy"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading chlorophyll data...")

chlorophyll = np.load(CHL_FILE)
chl_mask = np.load(MASK_FILE)

print("Chlorophyll shape:", chlorophyll.shape)


# --------------------------------------------------
# MASK INVALID VALUES
# --------------------------------------------------

chlorophyll = chlorophyll.astype(np.float32)

chlorophyll[~chl_mask.astype(bool)] = np.nan


# --------------------------------------------------
# CALCULATE TEMPORAL BASELINE
# --------------------------------------------------
# Since we currently have only October 2024,
# this is a TEST baseline.
#
# Later, when we have several years of data,
# this will be replaced by a multi-year
# seasonal climatology.

baseline = np.nanmedian(chlorophyll, axis=0)

print("Baseline calculated.")


# --------------------------------------------------
# CALCULATE CHLOROPHYLL ANOMALY
# --------------------------------------------------

anomaly = (chlorophyll - baseline) / (baseline + 1e-6)

print("Anomaly calculated.")


# --------------------------------------------------
# CALCULATE PERCENTILES
# --------------------------------------------------

valid_values = anomaly[np.isfinite(anomaly)]

p50 = np.percentile(valid_values, 50)
p75 = np.percentile(valid_values, 75)
p90 = np.percentile(valid_values, 90)

print()
print("Anomaly thresholds:")
print("50th percentile:", p50)
print("75th percentile:", p75)
print("90th percentile:", p90)


# --------------------------------------------------
# CREATE HAB RISK LABELS
# --------------------------------------------------

labels = np.zeros_like(anomaly, dtype=np.uint8)

# Class 1 = elevated bloom risk
labels[
    np.isfinite(anomaly) &
    (anomaly >= p75) &
    (anomaly < p90)
] = 1

# Class 2 = strong bloom risk
labels[
    np.isfinite(anomaly) &
    (anomaly >= p90)
] = 2


# Missing pixels = 255
labels[~np.isfinite(anomaly)] = 255


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

np.save(OUTPUT_LABEL, labels)
np.save(OUTPUT_ANOMALY, anomaly.astype(np.float32))


# --------------------------------------------------
# CHECK RESULTS
# --------------------------------------------------

print()
print("LABEL DISTRIBUTION")

valid_labels = labels[labels != 255]

unique, counts = np.unique(valid_labels, return_counts=True)

for label, count in zip(unique, counts):
    percentage = count / len(valid_labels) * 100

    if label == 0:
        name = "Normal"
    elif label == 1:
        name = "Elevated risk"
    elif label == 2:
        name = "Strong bloom risk"
    else:
        name = "Unknown"

    print(
        f"Class {label} ({name}): "
        f"{count} pixels ({percentage:.2f}%)"
    )

print()
print("Missing pixels:", np.sum(labels == 255))

print()
print("FILES CREATED:")
print(OUTPUT_LABEL)
print(OUTPUT_ANOMALY)

print()
print("HAB LABEL GENERATION COMPLETE!")