import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = np.array([
    [388621, 10281, 186934],
    [36525,   883,   26408],
    [13123,   443,   18717]
])


classes = [
    "Normal",
    "Elevated Risk",
    "Strong Bloom"
]


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title(
    "ConvLSTM Confusion Matrix - 2025 Test Set"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)


plt.xticks(
    range(len(classes)),
    classes,
    rotation=20
)

plt.yticks(
    range(len(classes)),
    classes
)


# Add values inside cells

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center"
        )


plt.colorbar(
    label="Number of Pixels"
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.close()


print(
    "✓ Confusion matrix saved:"
)

print(
    output_path
)


print(
    "\n✓ CONFUSION MATRIX COMPLETE"
)