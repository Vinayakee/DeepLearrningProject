import os
import glob
import numpy as np
import xarray as xr


# ============================================================
# SETTINGS
# ============================================================

CHL_DIR = "data/raw/Chlorophyll-a"
SST_FILE = "data/raw/sst/sst_2024_10_01_to_2024_10_31.nc"

OUTPUT_DIR = "data/processed"

LAT_MIN = 8.0
LAT_MAX = 13.0
LON_MIN = 74.0
LON_MAX = 77.5


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER: CROP LATITUDE CORRECTLY
# ============================================================

def crop_latitude(data, lat_name):

    lat_values = data[lat_name].values

    if lat_values[0] > lat_values[-1]:

        # Latitude is descending
        return data.sel(
            {
                lat_name: slice(
                    LAT_MAX,
                    LAT_MIN
                )
            }
        )

    else:

        # Latitude is ascending
        return data.sel(
            {
                lat_name: slice(
                    LAT_MIN,
                    LAT_MAX
                )
            }
        )


# ============================================================
# 1. FIND MODIS FILES
# ============================================================

print("\n" + "=" * 60)
print("MONTHLY PREPROCESSING")
print("October 2024")
print("=" * 60)

chl_files = sorted(
    glob.glob(
        os.path.join(
            CHL_DIR,
            "AQUA_MODIS.202410*.L3m.DAY.CHL.chlor_a.4km.nc"
        )
    )
)

print("\nMODIS files found:", len(chl_files))

if len(chl_files) == 0:

    raise FileNotFoundError(
        "No MODIS chlorophyll files found."
    )


# ============================================================
# 2. PROCESS MODIS CHLOROPHYLL
# ============================================================

print("\n" + "=" * 60)
print("PROCESSING CHLOROPHYLL-A")
print("=" * 60)

chlorophyll_days = []
chlorophyll_masks = []

reference_lat = None
reference_lon = None


for i, file_path in enumerate(chl_files):

    filename = os.path.basename(file_path)

    print(
        f"\n[{i + 1}/{len(chl_files)}] {filename}"
    )

    try:

        ds = xr.open_dataset(
            file_path,
            engine="h5netcdf"
        )

        chl = ds["chlor_a"]

        # Crop longitude
        chl = chl.sel(
            lon=slice(
                LON_MIN,
                LON_MAX
            )
        )

        # Crop latitude
        chl = crop_latitude(
            chl,
            "lat"
        )

        # Load into memory before closing file
        chl = chl.load()

        ds.close()

        # ----------------------------------------------------
        # Clean invalid values
        # ----------------------------------------------------

        values = chl.values.astype(
            np.float32
        )

        valid_mask = (
            np.isfinite(values)
            & (values > 0)
        )

        cleaned = values.copy()

        cleaned[~valid_mask] = np.nan

        # ----------------------------------------------------
        # Save reference grid
        # ----------------------------------------------------

        if reference_lat is None:

            reference_lat = chl.lat.values

            reference_lon = chl.lon.values

            print(
                "Grid shape:",
                cleaned.shape
            )

        chlorophyll_days.append(
            cleaned
        )

        chlorophyll_masks.append(
            valid_mask.astype(np.uint8)
        )

        valid_percentage = (
            np.sum(valid_mask)
            /
            valid_mask.size
            *
            100
        )

        print(
            f"Valid pixels: {valid_percentage:.2f}%"
        )

    except Exception as error:

        print(
            "ERROR:",
            error
        )


# ============================================================
# 3. STACK CHLOROPHYLL
# ============================================================

chlorophyll = np.stack(
    chlorophyll_days
).astype(np.float32)

chlorophyll_mask = np.stack(
    chlorophyll_masks
).astype(np.uint8)


print("\nChlorophyll final shape:")
print(chlorophyll.shape)

print(
    "Expected: (31, height, width)"
)

print(
    "Overall missing:",
    f"{np.isnan(chlorophyll).mean() * 100:.2f}%"
)


# ============================================================
# 4. PROCESS SST
# ============================================================

print("\n" + "=" * 60)
print("PROCESSING SST")
print("=" * 60)

print("\nOpening:")
print(SST_FILE)

sst_ds = xr.open_dataset(
    SST_FILE,
    engine="h5netcdf"
)

sst = sst_ds["analysed_sst"]

print(
    "Original SST shape:",
    sst.shape
)

# ------------------------------------------------------------
# Convert Kelvin → Celsius
# ------------------------------------------------------------

sst = sst - 273.15

# ------------------------------------------------------------
# Regrid SST to MODIS grid
# ------------------------------------------------------------

print(
    "\nRegridding SST to MODIS grid..."
)

sst_aligned = sst.interp(
    latitude=xr.DataArray(
        reference_lat,
        dims="latitude"
    ),
    longitude=xr.DataArray(
        reference_lon,
        dims="longitude"
    ),
    method="linear"
)

sst_aligned = sst_aligned.load()

sst_ds.close()

# Convert numpy
sst_array = sst_aligned.values.astype(
    np.float32
)

# ------------------------------------------------------------
# Clean SST
# ------------------------------------------------------------

sst_valid_mask = (
    np.isfinite(sst_array)
)

sst_array[~sst_valid_mask] = np.nan


print(
    "\nSST final shape:",
    sst_array.shape
)

print(
    "SST missing:",
    f"{np.isnan(sst_array).mean() * 100:.2f}%"
)


# ============================================================
# 5. CHECK SHAPES
# ============================================================

print("\n" + "=" * 60)
print("FINAL CHECK")
print("=" * 60)

print(
    "Chlorophyll:",
    chlorophyll.shape
)

print(
    "Chlorophyll mask:",
    chlorophyll_mask.shape
)

print(
    "SST:",
    sst_array.shape
)

print(
    "SST mask:",
    sst_valid_mask.shape
)


# ============================================================
# 6. SAVE DATA
# ============================================================

print("\n" + "=" * 60)
print("SAVING PROCESSED DATA")
print("=" * 60)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "chlorophyll_oct2024.npy"
    ),
    chlorophyll
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "chlorophyll_mask_oct2024.npy"
    ),
    chlorophyll_mask
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "sst_oct2024.npy"
    ),
    sst_array
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "sst_mask_oct2024.npy"
    ),
    sst_valid_mask.astype(np.uint8)
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "latitude_kerala.npy"
    ),
    reference_lat
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "longitude_kerala.npy"
    ),
    reference_lon
)


# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)

print("\nCreated files:")

print(
    "data/processed/chlorophyll_oct2024.npy"
)

print(
    "data/processed/chlorophyll_mask_oct2024.npy"
)

print(
    "data/processed/sst_oct2024.npy"
)

print(
    "data/processed/sst_mask_oct2024.npy"
)

print(
    "data/processed/latitude_kerala.npy"
)

print(
    "data/processed/longitude_kerala.npy"
)

print("\nReady for sequence creation.")