import os
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

PROCESSED_DIR = "data/processed"

EXPECTED_SHAPE = (1826, 120, 84)

# ============================================================
# DATASETS
# ============================================================

datasets = {
    "Chlorophyll": "chlorophyll_2021_2025.npy",
    "SST": "sst_2021_2025.npy",
    "Wind U": "wind_u_2021_2025.npy",
    "Wind V": "wind_v_2021_2025.npy",
    "Rainfall": "rainfall_2021_2025.npy",
    "Current U": "current_u_2021_2025.npy",
    "Current V": "current_v_2021_2025.npy",
    "HAB Labels": "hab_labels_2021_2025.npy",
}

# ============================================================
# START
# ============================================================

print("=" * 70)
print("FINAL 7-CHANNEL DATASET VERIFICATION")
print("=" * 70)

loaded = {}
all_files_present = True
all_shapes_correct = True

# ============================================================
# CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("FILE CHECK")
print("=" * 70)

for name, filename in datasets.items():

    path = os.path.join(
        PROCESSED_DIR,
        filename
    )

    if not os.path.exists(path):

        print(f"❌ {name}: FILE NOT FOUND")
        print(f"   {filename}")

        all_files_present = False

    else:

        data = np.load(path)

        loaded[name] = data

        print(f"✓ {name}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")

# ============================================================
# SHAPE CHECK
# ============================================================

print("\n" + "=" * 70)
print("SHAPE CHECK")
print("=" * 70)

for name, data in loaded.items():

    if data.shape == EXPECTED_SHAPE:

        print(
            f"✓ {name}: "
            f"{data.shape}"
        )

    else:

        print(
            f"❌ {name}: "
            f"{data.shape} "
            f"(expected {EXPECTED_SHAPE})"
        )

        all_shapes_correct = False

# ============================================================
# DATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATE CHECK")
print("=" * 70)

dates_path = os.path.join(
    PROCESSED_DIR,
    "dates_2021_2025.txt"
)

dates_ok = False

if os.path.exists(dates_path):

    dates = np.loadtxt(
        dates_path,
        dtype=str
    )

    print("✓ Date file exists")
    print(f"Number of dates: {len(dates)}")
    print(f"First date: {dates[0]}")
    print(f"Last date: {dates[-1]}")

    if (
        len(dates) == 1826
        and dates[0] == "2021-01-01"
        and dates[-1] == "2025-12-31"
    ):

        print(
            "✓ Date range is correct"
        )

        dates_ok = True

    else:

        print(
            "❌ Date range is incorrect"
        )

else:

    print("❌ Date file not found")

# ============================================================
# GRID CHECK
# ============================================================

print("\n" + "=" * 70)
print("GRID CHECK")
print("=" * 70)

lat_path = os.path.join(
    PROCESSED_DIR,
    "latitude_kerala_2021_2025.npy"
)

lon_path = os.path.join(
    PROCESSED_DIR,
    "longitude_kerala_2021_2025.npy"
)

grid_ok = False

if (
    os.path.exists(lat_path)
    and os.path.exists(lon_path)
):

    lat = np.load(lat_path)
    lon = np.load(lon_path)

    print(
        f"Latitude shape: {lat.shape}"
    )

    print(
        f"Longitude shape: {lon.shape}"
    )

    print(
        f"Latitude: "
        f"{lat.min():.2f} → {lat.max():.2f}"
    )

    print(
        f"Longitude: "
        f"{lon.min():.2f} → {lon.max():.2f}"
    )

    if (
        lat.shape == (120,)
        and lon.shape == (84,)
    ):

        print(
            "✓ Common 120 × 84 grid confirmed"
        )

        grid_ok = True

    else:

        print(
            "❌ Grid dimensions incorrect"
        )

else:

    print(
        "❌ Latitude/longitude files missing"
    )

# ============================================================
# MISSING DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("MISSING DATA CHECK")
print("=" * 70)

for name, data in loaded.items():

    if name == "HAB Labels":

        missing = np.sum(
            data == 255
        )

        total = data.size

        percentage = (
            missing / total
        ) * 100

        print(
            f"{name}: "
            f"{missing:,} missing "
            f"({percentage:.2f}%)"
        )

    else:

        missing = np.isnan(data).sum()

        total = data.size

        percentage = (
            missing / total
        ) * 100

        print(
            f"{name}: "
            f"{percentage:.2f}% NaN"
        )

# ============================================================
# HAB LABEL CHECK
# ============================================================

print("\n" + "=" * 70)
print("HAB LABEL CHECK")
print("=" * 70)

if "HAB Labels" in loaded:

    labels = loaded["HAB Labels"]

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    for value, count in zip(
        unique,
        counts
    ):

        if value == 0:
            label_name = "Normal"

        elif value == 1:
            label_name = "Elevated Risk"

        elif value == 2:
            label_name = "Strong Bloom Risk"

        elif value == 255:
            label_name = "Missing"

        else:
            label_name = "Unknown"

        print(
            f"Class {value} "
            f"({label_name}): "
            f"{count:,}"
        )

# ============================================================
# CURRENT CHECK
# ============================================================

print("\n" + "=" * 70)
print("OCEAN CURRENT CHECK")
print("=" * 70)

for name in [
    "Current U",
    "Current V"
]:

    if name not in loaded:
        continue

    data = loaded[name]

    valid = data[
        ~np.isnan(data)
    ]

    if len(valid) > 0:

        print(
            f"{name}:"
        )

        print(
            f"  Min: {valid.min():.4f} m/s"
        )

        print(
            f"  Max: {valid.max():.4f} m/s"
        )

        print(
            f"  Mean: {valid.mean():.4f} m/s"
        )

# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

if (
    all_files_present
    and all_shapes_correct
    and dates_ok
    and grid_ok
    and len(loaded) == 8
):

    print(
        "✅ ALL 7 ENVIRONMENTAL CHANNELS ARE PRESENT"
    )

    print(
        "✅ HAB LABELS ARE PRESENT"
    )

    print(
        "✅ ALL DATASETS USE 1826 DAYS"
    )

    print(
        "✅ ALL DATASETS USE THE SAME 120 × 84 GRID"
    )

    print(
        "✅ DATES: 2021-01-01 → 2025-12-31"
    )

    print(
        "✅ READY FOR SEQUENCE CREATION"
    )

else:

    print(
        "❌ SOMETHING NEEDS TO BE CHECKED"
    )

print("=" * 70)