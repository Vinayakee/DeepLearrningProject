import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

INPUT_FILE = Path(
    "data/raw/Chlorophyll-a/"
    "AQUA_MODIS.20241005.L3m.DAY.CHL.chlor_a.4km.nc"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Kerala study region
# --------------------------------------------------

# Approximate Kerala coastal region
LAT_MIN = 8.0
LAT_MAX = 13.0

LON_MIN = 74.0
LON_MAX = 77.5


# --------------------------------------------------
# 3. Open NASA NetCDF file
# --------------------------------------------------

print("Opening NASA chlorophyll-a file...")

ds = xr.open_dataset(INPUT_FILE, engine="h5netcdf")

print("File opened successfully!")
print("Variables:", list(ds.data_vars))


# --------------------------------------------------
# 4. Get chlorophyll-a data
# --------------------------------------------------

chlor_a = ds["chlor_a"]

print("\nOriginal data shape:")
print(chlor_a.shape)


# --------------------------------------------------
# 5. Crop to Kerala region
# --------------------------------------------------

print("\nCropping to Kerala region...")

# NASA latitude is descending, so use max -> min
kerala = chlor_a.sel(
    lat=slice(LAT_MAX, LAT_MIN),
    lon=slice(LON_MIN, LON_MAX)
)

print("Kerala crop shape:")
print(kerala.shape)


# --------------------------------------------------
# 6. Remove invalid values
# --------------------------------------------------

print("\nCleaning invalid values...")

data = kerala.values.astype(np.float32)

# Convert NaN and infinity to NaN
data[~np.isfinite(data)] = np.nan

print("Minimum:", np.nanmin(data))
print("Maximum:", np.nanmax(data))
print("NaN percentage:",
      np.isnan(data).sum() / data.size * 100)


# --------------------------------------------------
# 7. Save processed data
# --------------------------------------------------

output_file = OUTPUT_DIR / "chlor_a_20241005_kerala.npy"

np.save(output_file, data)

print("\nSaved:")
print(output_file)


# --------------------------------------------------
# 8. Visualize chlorophyll-a
# --------------------------------------------------

print("\nCreating visualization...")

plt.figure(figsize=(10, 6))

plt.imshow(
    data,
    extent=[
        LON_MIN,
        LON_MAX,
        LAT_MIN,
        LAT_MAX
    ],
    origin="upper",
    aspect="auto"
)

plt.colorbar(label="Chlorophyll-a (mg/m³)")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Aqua MODIS Chlorophyll-a\n"
    "Kerala Coastal Region - 5 October 2024"
)

plt.savefig(
    OUTPUT_DIR / "chlor_a_20241005_kerala.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nDone!")

# --------------------------------------------------
# 9. Process SST
# --------------------------------------------------

SST_FILE = Path(
    "data/raw/sst/"
    "METOFFICE-GLO-SST-L4-REP-OBS-SST_1788640004849.nc"
)

print("\nOpening SST file...")

sst_ds = xr.open_dataset(
    SST_FILE,
    engine="h5netcdf"
)

sst = sst_ds["analysed_sst"].isel(time=0)

print("SST opened successfully!")
print("Original SST shape:", sst.shape)


# --------------------------------------------------
# Convert Kelvin to Celsius
# --------------------------------------------------

sst_celsius = sst - 273.15


# --------------------------------------------------
# Save SST
# --------------------------------------------------

sst_data = sst_celsius.values.astype(np.float32)

sst_data[~np.isfinite(sst_data)] = np.nan

sst_output = OUTPUT_DIR / "sst_20241005_kerala.npy"

np.save(sst_output, sst_data)

print("SST saved:")
print(sst_output)


# --------------------------------------------------
# Visualize SST
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.imshow(
    sst_data,
    extent=[
        74.0,
        77.5,
        8.0,
        13.0
    ],
    origin="upper",
    aspect="auto"
)

plt.colorbar(label="Sea Surface Temperature (°C)")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "OSTIA Sea Surface Temperature\n"
    "Kerala Coastal Region - 5 October 2024"
)

plt.savefig(
    OUTPUT_DIR / "sst_20241005_kerala.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("SST processing complete!")