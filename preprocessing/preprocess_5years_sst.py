import os
import numpy as np
import xarray as xr
from datetime import datetime


# ============================================================
# SETTINGS
# ============================================================

START_YEAR = 2021
END_YEAR = 2025

SST_DIR = "data/raw/sst"

PROCESSED_DIR = "data/processed"

CHL_LAT_FILE = (
    "data/processed/"
    "latitude_kerala_2021_2025.npy"
)

CHL_LON_FILE = (
    "data/processed/"
    "longitude_kerala_2021_2025.npy"
)

DATES_FILE = (
    "data/processed/"
    "dates_2021_2025.txt"
)


# ============================================================
# LOAD TARGET GRID
# ============================================================

print("=" * 65)
print("5-YEAR SST PREPROCESSING")
print("=" * 65)

print("\nLoading chlorophyll grid...")

target_lat = np.load(CHL_LAT_FILE)
target_lon = np.load(CHL_LON_FILE)

print(
    "Target latitude:",
    target_lat.shape
)

print(
    "Target longitude:",
    target_lon.shape
)


# ============================================================
# LOAD TARGET DATES
# ============================================================

with open(DATES_FILE, "r") as f:

    date_strings = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    "Total target dates:",
    len(date_strings)
)


# ============================================================
# CREATE OUTPUT ARRAY
# ============================================================

num_days = len(date_strings)
height = len(target_lat)
width = len(target_lon)

sst_all = np.full(
    (num_days, height, width),
    np.nan,
    dtype=np.float32
)


# ============================================================
# PROCESS EACH YEAR
# ============================================================

processed_days = 0


for year in range(
    START_YEAR,
    END_YEAR + 1
):

    print()
    print("=" * 65)
    print(f"PROCESSING SST {year}")
    print("=" * 65)

    filename = (
        f"sst_{year}_kerala.nc"
    )

    filepath = os.path.join(
        SST_DIR,
        filename
    )

    if not os.path.exists(filepath):

        print(
            f"WARNING: {filename} not found!"
        )

        continue


    # --------------------------------------------------------
    # OPEN DATASET
    # --------------------------------------------------------

    print(
        "Opening:",
        filename
    )

    ds = xr.open_dataset(
        filepath,
        engine="h5netcdf"
    )


    print(
        "Variables:",
        list(ds.data_vars)
    )

    print(
        "Dimensions:",
        dict(ds.sizes)
    )


    # --------------------------------------------------------
    # GET SST VARIABLE
    # --------------------------------------------------------

    if "analysed_sst" in ds:

        sst = ds["analysed_sst"]

    else:

        # Automatically find a likely SST variable
        possible = [
            name
            for name in ds.data_vars
            if "sst" in name.lower()
        ]

        if not possible:

            ds.close()

            raise ValueError(
                f"No SST variable found in {filename}"
            )

        sst = ds[possible[0]]

        print(
            "Using variable:",
            possible[0]
        )


    # --------------------------------------------------------
    # FIND COORDINATE NAMES
    # --------------------------------------------------------

    lat_name = None
    lon_name = None

    for name in sst.coords:

        if name.lower() in [
            "latitude",
            "lat"
        ]:
            lat_name = name

        if name.lower() in [
            "longitude",
            "lon"
        ]:
            lon_name = name


    if lat_name is None:
        raise ValueError(
            f"Latitude coordinate not found in {filename}"
        )

    if lon_name is None:
        raise ValueError(
            f"Longitude coordinate not found in {filename}"
        )


    print(
        "Latitude coordinate:",
        lat_name
    )

    print(
        "Longitude coordinate:",
        lon_name
    )


    # --------------------------------------------------------
    # SORT COORDINATES
    # --------------------------------------------------------

    sst = sst.sortby(lat_name)
    sst = sst.sortby(lon_name)


    # --------------------------------------------------------
    # CONVERT KELVIN → CELSIUS
    # --------------------------------------------------------

    print(
        "Converting SST to Celsius..."
    )

    sst = sst - 273.15


    # --------------------------------------------------------
    # REGRID TO CHLOROPHYLL GRID
    # --------------------------------------------------------

    print(
        "Regridding to 120 × 84..."
    )

    sst_regridded = sst.interp(
        {
            lat_name: xr.DataArray(
                target_lat,
                dims="latitude"
            ),

            lon_name: xr.DataArray(
                target_lon,
                dims="longitude"
            )
        },

        method="linear"
    )


    # --------------------------------------------------------
    # FIND TIME DIMENSION
    # --------------------------------------------------------

    time_name = None

    for name in sst_regridded.dims:

        if name.lower() == "time":

            time_name = name
            break


    if time_name is None:

        ds.close()

        raise ValueError(
            f"Time dimension not found in {filename}"
        )


    # --------------------------------------------------------
    # GET DATES
    # --------------------------------------------------------

    times = sst_regridded[
        time_name
    ].values


    # --------------------------------------------------------
    # SAVE INTO GLOBAL ARRAY
    # --------------------------------------------------------

    for local_index, timestamp in enumerate(times):

        timestamp_str = str(
            np.datetime_as_string(
                timestamp,
                unit="D"
            )
        )

        if timestamp_str in date_strings:

            global_index = date_strings.index(
                timestamp_str
            )

            values = (
                sst_regridded
                .isel(
                    {time_name: local_index}
                )
                .values
            )

            values = values.astype(
                np.float32
            )

            # Remove physically impossible values
            values[
                (values < -5) |
                (values > 50)
            ] = np.nan

            sst_all[
                global_index
            ] = values

            processed_days += 1


    ds.close()

    print(
        f"{year} processed."
    )


# ============================================================
# CREATE SST VALIDITY MASK
# ============================================================

sst_mask = np.isfinite(
    sst_all
).astype(
    np.uint8
)


# ============================================================
# STATISTICS
# ============================================================

valid_values = sst_all[
    np.isfinite(sst_all)
]


print()
print("=" * 65)
print("SST STATISTICS")
print("=" * 65)

print(
    "Processed days:",
    processed_days
)

print(
    "Final shape:",
    sst_all.shape
)

if len(valid_values) > 0:

    print(
        "Minimum SST:",
        np.min(valid_values),
        "°C"
    )

    print(
        "Maximum SST:",
        np.max(valid_values),
        "°C"
    )

    print(
        "Mean SST:",
        np.mean(valid_values),
        "°C"
    )


missing_percentage = (
    np.mean(
        ~np.isfinite(sst_all)
    )
    * 100
)

print(
    "Missing SST:",
    f"{missing_percentage:.2f}%"
)


# ============================================================
# SAVE
# ============================================================

print()
print("=" * 65)
print("SAVING SST")
print("=" * 65)


np.save(
    os.path.join(
        PROCESSED_DIR,
        "sst_2021_2025.npy"
    ),
    sst_all
)


np.save(
    os.path.join(
        PROCESSED_DIR,
        "sst_mask_2021_2025.npy"
    ),
    sst_mask
)


print(
    "Created:"
)

print(
    "data/processed/sst_2021_2025.npy"
)

print(
    "data/processed/sst_mask_2021_2025.npy"
)


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 65)
print("FINAL CHECK")
print("=" * 65)

print(
    "Chlorophyll grid:",
    (
        len(date_strings),
        len(target_lat),
        len(target_lon)
    )
)

print(
    "SST grid:",
    sst_all.shape
)

print()

if sst_all.shape == (
    len(date_strings),
    len(target_lat),
    len(target_lon)
):

    print(
        "✓ SST GRID MATCHES CHLOROPHYLL"
    )

else:

    print(
        "✗ GRID SIZE DOES NOT MATCH"
    )


print()
print("SST PREPROCESSING COMPLETE!")