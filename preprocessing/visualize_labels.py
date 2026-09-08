import numpy as np
import matplotlib.pyplot as plt

# Load labels
labels = np.load("data/processed/hab_labels_oct2024.npy")

# --------------------------------------------------
# FIND DAY WITH MOST VALID DATA
# --------------------------------------------------

valid_counts = []

for day in range(labels.shape[0]):
    valid = np.sum(labels[day] != 255)
    valid_counts.append(valid)

best_day = np.argmax(valid_counts)

print("Best day:", best_day + 1)
print("Valid pixels:", valid_counts[best_day])
print("Missing pixels:", labels.shape[1] * labels.shape[2] - valid_counts[best_day])

# --------------------------------------------------
# PREPARE IMAGE
# --------------------------------------------------

image = labels[best_day].astype(float)

# Hide missing pixels
image[image == 255] = np.nan

# --------------------------------------------------
# PLOT
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.imshow(
    image,
    origin="lower",
    interpolation="nearest",
    vmin=0,
    vmax=2,
    cmap="viridis"
)

cbar = plt.colorbar(ticks=[0, 1, 2])

cbar.ax.set_yticklabels([
    "Normal",
    "Elevated Risk",
    "Strong Bloom Risk"
])

cbar.set_label("HAB Risk Class")

plt.title(
    f"HAB Risk Labels - October 2024 - Day {best_day + 1}"
)

plt.xlabel("Longitude grid")
plt.ylabel("Latitude grid")

plt.tight_layout()

plt.savefig(
    "data/processed/hab_labels_best_day.png",
    dpi=200
)

plt.show()