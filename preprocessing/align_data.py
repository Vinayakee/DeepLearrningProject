import xarray as xr
import numpy as np
from pathlib import Path


# --------------------------------------------------
# File paths
# --------------------------------------------------

CHL_FILE = Path(
    "data/raw/Chlorophyll-a/"
    "AQUA_MODIS.20241005.L3m.DAY.CHL.chlor_a.4km.nc"
)

SST_FILE = Path(
    "data/raw/sst/"
    "METOFFICE-GLO-SST-L4-REP-OBS-SST_1788640004849.nc"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Kerala region
# --------------------------------------------------

LAT_MIN = 8.0
LAT_MAX = 13.0

LON_MIN = 74.0
LON_MAX = 77.5


# --------------------------------------------------
# Open Chlorophyll
# --------------------------------------------------

print("1. Opening chlorophyll file...", flush=True)

chl_ds = xr.open_dataset(
    CHL_FILE,
    engine="h5netcdf"
)

print("2. Chlorophyll file opened!", flush=True)

chl = chl_ds["chlor_a"]

chl = chl.sel(
    lat=slice(LAT_MAX, LAT_MIN),
    lon=slice(LON_MIN, LON_MAX)
)

# Load only the Kerala crop into memory
chl = chl.load()

print("3. Chlorophyll shape:", chl.shape, flush=True)


# --------------------------------------------------
# Open SST
# --------------------------------------------------

print("4. Opening SST file...", flush=True)

sst_ds = xr.open_dataset(
    SST_FILE,
    engine="h5netcdf"
)

print("5. SST file opened!", flush=True)

sst = sst_ds["analysed_sst"].isel(time=0)

sst = sst.load()

print("6. SST shape:", sst.shape, flush=True)


# --------------------------------------------------
# Convert Kelvin to Celsius
# --------------------------------------------------

print("7. Converting SST to Celsius...", flush=True)

sst = sst - 273.15


# --------------------------------------------------
# Regrid SST
# --------------------------------------------------

print("8. Regridding SST to chlorophyll grid...", flush=True)

sst_aligned = sst.interp(
    latitude=chl.lat,
    longitude=chl.lon,
    method="linear"
)

sst_aligned = sst_aligned.load()

print("9. Regridding complete!", flush=True)


# --------------------------------------------------
# Convert to NumPy
# --------------------------------------------------

chl_data = chl.values.astype(np.float32)
sst_data = sst_aligned.values.astype(np.float32)


# --------------------------------------------------
# Clean invalid values
# --------------------------------------------------

chl_data[~np.isfinite(chl_data)] = np.nan
sst_data[~np.isfinite(sst_data)] = np.nan


# --------------------------------------------------
# Print final shapes
# --------------------------------------------------

print("\nFINAL DATA", flush=True)

print("Chlorophyll:", chl_data.shape, flush=True)
print("SST:", sst_data.shape, flush=True)

print(
    "Chlorophyll NaN:",
    np.isnan(chl_data).sum() / chl_data.size * 100,
    "%",
    flush=True
)

print(
    "SST NaN:",
    np.isnan(sst_data).sum() / sst_data.size * 100,
    "%",
    flush=True
)


# --------------------------------------------------
# Save
# --------------------------------------------------

np.save(
    OUTPUT_DIR / "chlorophyll_aligned_20241005.npy",
    chl_data
)

np.save(
    OUTPUT_DIR / "sst_aligned_20241005.npy",
    sst_data
)


print("\nSaved successfully!", flush=True)

print(
    "data/processed/chlorophyll_aligned_20241005.npy",
    flush=True
)

print(
    "data/processed/sst_aligned_20241005.npy",
    flush=True
)

print("\nALIGNMENT COMPLETE!", flush=True)