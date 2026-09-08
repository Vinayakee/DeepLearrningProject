import os
import zipfile
import tempfile
import numpy as np
import xarray as xr

# ============================================================
# SETTINGS
# ============================================================

RAW_DIR = "data/raw/rainfall"
PROCESSED_DIR = "data/processed"

YEARS = [2021, 2022, 2023, 2024, 2025]

# ============================================================
# LOAD TARGET CHLOROPHYLL GRID
# ============================================================

print("Loading target grid...")

target_lat = np.load(
    "data/processed/latitude_kerala_2021_2025.npy"
)

target_lon = np.load(
    "data/processed/longitude_kerala_2021_2025.npy"
)

dates = np.loadtxt(
    "data/processed/dates_2021_2025.txt",
    dtype=str
)

print("Target latitude:", target_lat.shape)
print("Target longitude:", target_lon.shape)
print("Number of dates:", len(dates))

# ============================================================
# CREATE OUTPUT ARRAYS
# ============================================================

num_days = len(dates)
height = len(target_lat)
width = len(target_lon)

rainfall = np.full(
    (num_days, height, width),
    np.nan,
    dtype=np.float32
)

rainfall_mask = np.zeros(
    (num_days, height, width),
    dtype=np.uint8
)

# Date -> index lookup
date_to_index = {
    date: i for i, date in enumerate(dates)
}

# ============================================================
# PROCESS ALL MONTHS
# ============================================================

processed_files = 0
failed_files = 0

for year in YEARS:

    for month in range(1, 13):

        month_str = f"{month:02d}"

        filename = (
            f"era5_rainfall_{year}_{month_str}.zip"
        )

        filepath = os.path.join(
            RAW_DIR,
            filename
        )

        print("\n" + "=" * 60)
        print(f"PROCESSING: {year}-{month_str}")
        print("=" * 60)

        if not os.path.exists(filepath):
            print("✗ File not found:", filepath)
            failed_files += 1
            continue

        try:

            # ------------------------------------------------
            # IMPORTANT:
            # The CDS files are actually NetCDF/HDF5 files,
            # despite having .zip extension.
            # ------------------------------------------------

            ds = xr.open_dataset(
                filepath,
                engine="h5netcdf"
            )

            print("Variables:", list(ds.data_vars))

            # Find precipitation variable
            if "tp" in ds.data_vars:
                var_name = "tp"
            else:
                var_name = list(ds.data_vars)[0]

            data = ds[var_name]

            # ------------------------------------------------
            # SORT COORDINATES
            # ------------------------------------------------

            if "latitude" in data.coords:
                data = data.sortby("latitude")

            if "longitude" in data.coords:
                data = data.sortby("longitude")

            # ------------------------------------------------
            # CONVERT METERS → MILLIMETERS
            # ------------------------------------------------

            # ERA5 total precipitation is stored in metres.
            # Convert to mm for easier interpretation.

            data = data * 1000.0

            # ------------------------------------------------
            # INTERPOLATE TO CHL GRID
            # ------------------------------------------------

            data_interp = data.interp(
                latitude=target_lat,
                longitude=target_lon,
                method="linear"
            )

            # ------------------------------------------------
            # SAVE EACH DAY
            # ------------------------------------------------

            for time_value in data_interp.valid_time.values:

                date_string = str(
                    time_value
                )[:10]

                if date_string not in date_to_index:
                    continue

                index = date_to_index[date_string]

                day_data = data_interp.sel(
                    valid_time=time_value
                ).values

                rainfall[index] = day_data.astype(
                    np.float32
                )

                rainfall_mask[index] = (
                    ~np.isnan(day_data)
                ).astype(np.uint8)

            ds.close()

            processed_files += 1

            print(
                f"✓ {year}-{month_str} processed"
            )

        except Exception as e:

            failed_files += 1

            print(
                f"✗ {year}-{month_str} FAILED"
            )

            print("Error:", e)

# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 60)
print("SAVING RAINFALL DATA")
print("=" * 60)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

np.save(
    os.path.join(
        PROCESSED_DIR,
        "rainfall_2021_2025.npy"
    ),
    rainfall
)

np.save(
    os.path.join(
        PROCESSED_DIR,
        "rainfall_mask_2021_2025.npy"
    ),
    rainfall_mask
)

# ============================================================
# DIAGNOSTICS
# ============================================================

print("\n" + "=" * 60)
print("FINAL CHECK")
print("=" * 60)

print(
    "Rainfall shape:",
    rainfall.shape
)

print(
    "Rainfall mask shape:",
    rainfall_mask.shape
)

missing_percentage = (
    np.isnan(rainfall).sum()
    / rainfall.size
) * 100

print(
    f"Rainfall missing: {missing_percentage:.2f}%"
)

valid_values = rainfall[
    ~np.isnan(rainfall)
]

if len(valid_values) > 0:

    print(
        f"Rainfall min: {valid_values.min():.4f} mm"
    )

    print(
        f"Rainfall max: {valid_values.max():.4f} mm"
    )

    print(
        f"Rainfall mean: {valid_values.mean():.4f} mm"
    )

print(
    "Processed files:",
    processed_files
)

print(
    "Failed files:",
    failed_files
)

print("\n✓ RAINFALL PREPROCESSING COMPLETE!")