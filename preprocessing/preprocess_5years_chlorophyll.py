import os
import glob
import numpy as np
import xarray as xr
from datetime import date, timedelta


# ============================================================
# SETTINGS
# ============================================================

CHL_DIR = "data/raw/Chlorophyll-a"
OUTPUT_DIR = "data/processed"

LAT_MIN = 8.0
LAT_MAX = 13.0
LON_MIN = 74.0
LON_MAX = 77.5

START_DATE = date(2021, 1, 1)
END_DATE = date(2025, 12, 31)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GET ALL DATES
# ============================================================

dates = []

current = START_DATE

while current <= END_DATE:
    dates.append(current)
    current += timedelta(days=1)

total_days = len(dates)

print("=" * 65)
print("5-YEAR CHLOROPHYLL PREPROCESSING")
print("=" * 65)

print("Total calendar days:", total_days)
print("Date range:", START_DATE, "to", END_DATE)


# ============================================================
# FIND AVAILABLE FILES
# ============================================================

files = glob.glob(
    os.path.join(
        CHL_DIR,
        "AQUA_MODIS.202*.L3m.DAY.CHL.chlor_a.4km.nc"
    )
)

print("MODIS files found:", len(files))


# ============================================================
# CREATE FILE LOOKUP
# ============================================================

file_lookup = {}

for filepath in files:

    filename = os.path.basename(filepath)

    # Example:
    # AQUA_MODIS.20241005.L3m.DAY.CHL.chlor_a.4km.nc

    try:

        date_string = filename.split(".")[1]

        file_date = date(
            int(date_string[0:4]),
            int(date_string[4:6]),
            int(date_string[6:8])
        )

        file_lookup[file_date] = filepath

    except Exception:

        print("Could not read date:", filename)


print("Valid files indexed:", len(file_lookup))


# ============================================================
# FIND FIRST AVAILABLE FILE
# ============================================================

first_date = None

for d in dates:

    if d in file_lookup:
        first_date = d
        break


if first_date is None:
    raise RuntimeError("No MODIS files found.")


print("First available file:", first_date)


# ============================================================
# READ REFERENCE GRID
# ============================================================

print()
print("Reading reference grid...")

reference_file = file_lookup[first_date]

with xr.open_dataset(
    reference_file,
    engine="h5netcdf"
) as ds:

    print("Variables:", list(ds.data_vars))

    chl = ds["chlor_a"]

    lat_name = "lat"
    lon_name = "lon"

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    # Handle latitude ordering
    if lat[0] > lat[-1]:

        lat_indices = np.where(
            (lat >= LAT_MIN) &
            (lat <= LAT_MAX)
        )[0]

    else:

        lat_indices = np.where(
            (lat >= LAT_MIN) &
            (lat <= LAT_MAX)
        )[0]

    lon_indices = np.where(
        (lon >= LON_MIN) &
        (lon <= LON_MAX)
    )[0]

    reference_lat = lat[lat_indices]
    reference_lon = lon[lon_indices]

    sample = chl.values

    # Determine dimensions
    if sample.ndim == 2:
        sample_crop = sample[
            np.ix_(lat_indices, lon_indices)
        ]

    else:
        sample_crop = sample[
            0,
            np.ix_(lat_indices, lon_indices)
        ]

    height, width = sample_crop.shape


print()
print("Kerala grid:")
print("Latitude:", height)
print("Longitude:", width)
print("Grid shape:", (height, width))


# ============================================================
# INITIALIZE ARRAYS
# ============================================================

chlorophyll = np.full(
    (total_days, height, width),
    np.nan,
    dtype=np.float32
)

chlorophyll_mask = np.zeros(
    (total_days, height, width),
    dtype=np.uint8
)


# ============================================================
# PROCESS EVERY DAY
# ============================================================

successful = 0
missing = 0
failed = []

for index, current_date in enumerate(dates):

    print(
        f"[{index + 1}/{total_days}] "
        f"{current_date}",
        end=" "
    )

    filepath = file_lookup.get(current_date)

    # --------------------------------------------------------
    # Missing file
    # --------------------------------------------------------

    if filepath is None:

        print("→ MISSING")

        missing += 1

        continue

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    try:

        with xr.open_dataset(
            filepath,
            engine="h5netcdf"
        ) as ds:

            data = ds["chlor_a"].values

            if data.ndim == 2:

                cropped = data[
                    np.ix_(lat_indices, lon_indices)
                ]

            else:

                cropped = data[
                    0,
                    np.ix_(lat_indices, lon_indices)
                ]

            cropped = cropped.astype(
                np.float32,
                copy=False
            )

            # ------------------------------------------------
            # Clean invalid values
            # ------------------------------------------------

            valid = (
                np.isfinite(cropped) &
                (cropped > 0)
            )

            cleaned = np.full(
                cropped.shape,
                np.nan,
                dtype=np.float32
            )

            cleaned[valid] = cropped[valid]

            chlorophyll[index] = cleaned

            chlorophyll_mask[index] = valid.astype(
                np.uint8
            )

            successful += 1

            valid_percentage = (
                np.sum(valid) /
                valid.size *
                100
            )

            print(
                f"→ OK "
                f"({valid_percentage:.1f}% valid)"
            )

    except Exception as e:

        print("→ ERROR")

        failed.append(
            f"{current_date}: {str(e)}"
        )


# ============================================================
# SAVE DATA
# ============================================================

print()
print("=" * 65)
print("SAVING DATA")
print("=" * 65)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "chlorophyll_2021_2025.npy"
    ),
    chlorophyll
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "chlorophyll_mask_2021_2025.npy"
    ),
    chlorophyll_mask
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "latitude_kerala_2021_2025.npy"
    ),
    reference_lat
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "longitude_kerala_2021_2025.npy"
    ),
    reference_lon
)


# ============================================================
# SAVE DATE LIST
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "dates_2021_2025.txt"
    ),
    "w"
) as f:

    for d in dates:
        f.write(d.strftime("%Y-%m-%d") + "\n")


# ============================================================
# SAVE PROCESSING REPORT
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "chlorophyll_processing_report.txt"
    ),
    "w"
) as f:

    f.write("5-YEAR MODIS CHLOROPHYLL PROCESSING REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Start date: {START_DATE}\n")
    f.write(f"End date: {END_DATE}\n")
    f.write(f"Calendar days: {total_days}\n")
    f.write(f"Files found: {len(file_lookup)}\n")
    f.write(f"Successfully processed: {successful}\n")
    f.write(f"Missing: {missing}\n")
    f.write(f"Processing errors: {len(failed)}\n")

    if failed:

        f.write("\nERRORS:\n")

        for item in failed:
            f.write(item + "\n")


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 65)
print("PREPROCESSING COMPLETE")
print("=" * 65)

print("Calendar days:", total_days)
print("Successfully processed:", successful)
print("Missing:", missing)
print("Errors:", len(failed))

print()
print("Final array shape:")
print(chlorophyll.shape)

print()
print("Created:")

print(
    "data/processed/chlorophyll_2021_2025.npy"
)

print(
    "data/processed/chlorophyll_mask_2021_2025.npy"
)

print(
    "data/processed/latitude_kerala_2021_2025.npy"
)

print(
    "data/processed/longitude_kerala_2021_2025.npy"
)

print(
    "data/processed/dates_2021_2025.txt"
)

print()
print("DONE!")