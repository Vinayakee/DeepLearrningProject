import numpy as np
import matplotlib.pyplot as plt

# Load data
mask = np.load(
    "data/processed/chlorophyll_mask_2021_2025.npy"
)

lat = np.load(
    "data/processed/latitude_kerala_2021_2025.npy"
)

lon = np.load(
    "data/processed/longitude_kerala_2021_2025.npy"
)

# --------------------------------------------------
# PIXELS OBSERVED AT LEAST ONCE
# --------------------------------------------------

ever_observed = np.max(mask, axis=0)

# Convert to 0/1
ocean_mask = ever_observed > 0

print("Grid shape:", ocean_mask.shape)
print("Observed at least once:", np.sum(ocean_mask))
print("Never observed:", np.sum(~ocean_mask))
print(
    "Observed percentage:",
    np.mean(ocean_mask) * 100
)

# --------------------------------------------------
# PLOT
# --------------------------------------------------

plt.figure(figsize=(10, 7))

plt.imshow(
    ocean_mask.astype(float),
    origin="lower",
    extent=[
        lon.min(),
        lon.max(),
        lat.min(),
        lat.max()
    ],
    interpolation="nearest"
)

plt.colorbar(
    ticks=[0, 1],
    label="Observation Status"
)

plt.clim(0, 1)

plt.title(
    "MODIS Observation Coverage - 2021–2025"
)

plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")

plt.tight_layout()

plt.savefig(
    "data/processed/ocean_observation_mask.png",
    dpi=200
)

plt.show()