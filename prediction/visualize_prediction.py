import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# FILES
# ============================================================

PREDICTION_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "latest_prediction.npy"
)

LAT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "latitude_kerala_2021_2025.npy"
)

LON_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "longitude_kerala_2021_2025.npy"
)

DATE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "dates_2021_2025.txt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "maps"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

STRONG_BLOOM_THRESHOLD = 0.50

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CREATING 5-DAY HAB PREDICTION MAPS")
print("=" * 70)

print("\nLoading prediction...")

prediction = np.load(PREDICTION_FILE)

print("Prediction shape:", prediction.shape)

# Expected:
# (1, 5, 120, 84, 3)

if prediction.ndim != 5:
    raise ValueError(
        f"Unexpected prediction shape: {prediction.shape}"
    )

# Remove batch dimension
prediction = prediction[0]

print("Prediction after removing batch:", prediction.shape)

# ============================================================
# LOAD LAT/LON
# ============================================================

latitude = np.load(LAT_FILE)
longitude = np.load(LON_FILE)

print("\nGrid:")
print("Latitude :", latitude.shape)
print("Longitude:", longitude.shape)

# ============================================================
# LOAD DATES
# ============================================================

dates = np.loadtxt(
    DATE_FILE,
    dtype=str
)

last_input_date = dates[-1]

print("\nLast available input date:", last_input_date)

# ============================================================
# CREATE FUTURE DATES
# ============================================================

from datetime import datetime, timedelta

last_date = datetime.strptime(
    last_input_date,
    "%Y-%m-%d"
)

future_dates = [
    last_date + timedelta(days=i)
    for i in range(1, 6)
]

# ============================================================
# EXTRACT CLASS PROBABILITIES
# ============================================================

# Class definitions:
# 0 = Normal
# 1 = Elevated Risk
# 2 = Strong Bloom Risk

normal_probability = prediction[:, :, :, 0]

elevated_probability = prediction[:, :, :, 1]

strong_probability = prediction[:, :, :, 2]

# ============================================================
# CREATE MAPS
# ============================================================

print("\nCreating forecast maps...")

for day in range(5):

    date = future_dates[day]

    strong_map = strong_probability[day]

    # --------------------------------------------------------
    # Convert probability into risk classes
    # --------------------------------------------------------

    risk_map = np.zeros_like(strong_map)

    # Strong Bloom
    risk_map[
        strong_map >= STRONG_BLOOM_THRESHOLD
    ] = 2

    # Elevated Risk
    elevated_mask = (
        (strong_map >= 0.20)
        & (strong_map < STRONG_BLOOM_THRESHOLD)
    )

    risk_map[elevated_mask] = 1

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    # Use longitude/latitude extent
    extent = [
        longitude.min(),
        longitude.max(),
        latitude.min(),
        latitude.max()
    ]

    image = ax.imshow(
        strong_map,
        extent=extent,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    ax.set_title(
        f"HAB Early Warning - Day +{day + 1}\n"
        f"Forecast Date: {date.strftime('%Y-%m-%d')}"
    )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    colorbar = plt.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Strong Bloom Probability"
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"day_{day + 1}.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        f"✓ Day +{day + 1}: "
        f"{date.strftime('%Y-%m-%d')} "
        f"-> {output_file}"
    )

# ============================================================
# SAVE RISK MAPS
# ============================================================

risk_output = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "risk_classes.npy"
)

np.save(
    risk_output,
    risk_map
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("✓ 5-DAY PREDICTION MAPS CREATED")
print("=" * 70)

print("\nMaps saved in:")
print("outputs/maps/")

print("\nGenerated:")

for day in range(1, 6):
    print(
        f"  ✓ day_{day}.png"
    )

print("\nStrong Bloom threshold:", STRONG_BLOOM_THRESHOLD)

print("=" * 70)