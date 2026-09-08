import os
import zipfile
import tempfile
import numpy as np
import xarray as xr

# ============================================================
# ERA5 WIND PREPROCESSING
# 2021-2025
#
# Input:
#   data/raw/wind/*.zip
#
# Output:
#   U wind -> (1826, 120, 84)
#   V wind -> (1826, 120, 84)
# ============================================================

RAW_DIR = "data/raw/wind"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load our existing MODIS grid
# ------------------------------------------------------------

target_lat = np.load(
    "data/processed/latitude_kerala_2021_2025.npy"
)

target_lon = np.load(
    "data/processed/longitude_kerala_2021_2025.npy"
)

print("Target latitude shape:", target_lat.shape)
print("Target longitude shape:", target_lon.shape)

# ------------------------------------------------------------
# Load master dates
# ------------------------------------------------------------

with open(
    "data/processed/dates_2021_2025.txt",
    "r"
) as f:

    dates = [
        line.strip()
        for line in f
        if line.strip()
    ]

print("Number of dates:", len(dates))

# ------------------------------------------------------------
# Create output arrays
# ------------------------------------------------------------

num_days = len(dates)
height = len(target_lat)
width = len(target_lon)

u_wind = np.full(
    (num_days, height, width),
    np.nan,
    dtype=np.float32
)

v_wind = np.full(
    (num_days, height, width),
    np.nan,
    dtype=np.float32
)

# ------------------------------------------------------------
# Convert dates to lookup
# ------------------------------------------------------------

date_to_index = {
    date: i
    for i, date in enumerate(dates)
}

# ------------------------------------------------------------
# Find all monthly ZIP files
# ------------------------------------------------------------

zip_files = sorted(
    [
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".zip")
    ]
)

print("\nZIP files found:", len(zip_files))

if len(zip_files) != 60:
    print(
        "WARNING: Expected 60 monthly files "
        f"but found {len(zip_files)}"
    )

# ------------------------------------------------------------
# Process every month
# ------------------------------------------------------------

for file_number, zip_name in enumerate(zip_files, start=1):

    zip_path = os.path.join(
        RAW_DIR,
        zip_name
    )

    print("\n" + "=" * 60)
    print(
        f"Processing {file_number}/{len(zip_files)}: "
        f"{zip_name}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Extract this ZIP temporarily
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_file:

            zip_file.extractall(temp_dir)

        # ----------------------------------------------------
        # Find NetCDF files
        # ----------------------------------------------------

        nc_files = [
            os.path.join(temp_dir, f)
            for f in os.listdir(temp_dir)
            if f.endswith(".nc")
        ]

        if len(nc_files) != 2:

            print(
                "WARNING: Expected 2 NetCDF files, "
                f"found {len(nc_files)}"
            )

            continue

        # ----------------------------------------------------
        # Identify U and V files
        # ----------------------------------------------------

        u_file = None
        v_file = None

        for nc_file in nc_files:

            name = os.path.basename(
                nc_file
            ).lower()

            if "_u_" in name:
                u_file = nc_file

            elif "_v_" in name:
                v_file = nc_file

        if u_file is None or v_file is None:

            print("ERROR: Could not identify U/V files")

            print(nc_files)

            continue

        # ----------------------------------------------------
        # Open datasets
        # ----------------------------------------------------

        ds_u = xr.open_dataset(
            u_file,
            engine="h5netcdf"
        )

        ds_v = xr.open_dataset(
            v_file,
            engine="h5netcdf"
        )

        # ----------------------------------------------------
        # Find the actual data variables
        # ----------------------------------------------------

        u_variable = [
            v for v in ds_u.data_vars
            if v not in ["latitude", "longitude"]
        ][0]

        v_variable = [
            v for v in ds_v.data_vars
            if v not in ["latitude", "longitude"]
        ][0]

        print("U variable:", u_variable)
        print("V variable:", v_variable)

        # ----------------------------------------------------
        # Sort coordinates
        # ----------------------------------------------------

        ds_u = ds_u.sortby(
            ["latitude", "longitude"]
        )

        ds_v = ds_v.sortby(
            ["latitude", "longitude"]
        )

        # ----------------------------------------------------
        # Interpolate ERA5 onto MODIS grid
        # ----------------------------------------------------

        u_interp = ds_u[u_variable].interp(
            latitude=target_lat,
            longitude=target_lon,
            method="linear"
        )

        v_interp = ds_v[v_variable].interp(
            latitude=target_lat,
            longitude=target_lon,
            method="linear"
        )

        # ----------------------------------------------------
        # Put each day into master array
        # ----------------------------------------------------

        for time_index, timestamp in enumerate(
            ds_u["valid_time"].values
        ):

            date_string = str(
                timestamp
            )[:10]

            if date_string not in date_to_index:
                continue

            master_index = date_to_index[
                date_string
            ]

            u_wind[master_index] = (
                u_interp.isel(
                    valid_time=time_index
                ).values
            )

            v_wind[master_index] = (
                v_interp.isel(
                    valid_time=time_index
                ).values
            )

        ds_u.close()
        ds_v.close()

    print(
        f"✓ Finished {zip_name}"
    )

# ============================================================
# Save
# ============================================================

u_output = os.path.join(
    PROCESSED_DIR,
    "wind_u_2021_2025.npy"
)

v_output = os.path.join(
    PROCESSED_DIR,
    "wind_v_2021_2025.npy"
)

np.save(
    u_output,
    u_wind
)

np.save(
    v_output,
    v_wind
)

# ============================================================
# Create masks
# ============================================================

u_mask = ~np.isnan(u_wind)
v_mask = ~np.isnan(v_wind)

np.save(
    os.path.join(
        PROCESSED_DIR,
        "wind_u_mask_2021_2025.npy"
    ),
    u_mask
)

np.save(
    os.path.join(
        PROCESSED_DIR,
        "wind_v_mask_2021_2025.npy"
    ),
    v_mask
)

# ============================================================
# Final diagnostics
# ============================================================

print("\n" + "=" * 60)
print("ERA5 WIND PREPROCESSING COMPLETE")
print("=" * 60)

print("\nU wind shape:")
print(u_wind.shape)

print("\nV wind shape:")
print(v_wind.shape)

print("\nU wind statistics:")

print(
    "Min:",
    np.nanmin(u_wind)
)

print(
    "Max:",
    np.nanmax(u_wind)
)

print(
    "Mean:",
    np.nanmean(u_wind)
)

print("\nV wind statistics:")

print(
    "Min:",
    np.nanmin(v_wind)
)

print(
    "Max:",
    np.nanmax(v_wind)
)

print(
    "Mean:",
    np.nanmean(v_wind)
)

print("\nMissing percentage:")

print(
    "U wind:",
    np.isnan(u_wind).mean() * 100,
    "%"
)

print(
    "V wind:",
    np.isnan(v_wind).mean() * 100,
    "%"
)

print("\nSaved:")

print(u_output)
print(v_output)

print("\n" + "=" * 60)