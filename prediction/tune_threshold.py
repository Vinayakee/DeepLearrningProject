import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from training.data_generator import HABDataGenerator


MODEL_PATH = "models/best_convlstm.keras"

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]


print("=" * 70)
print("HAB STRONG BLOOM THRESHOLD TUNING")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✓ Model loaded")


print("\nLoading test data...")

test_generator = HABDataGenerator(
    split="test",
    batch_size=1,
    shuffle=False
)

print(
    f"Test samples: {len(test_generator)}"
)


# ============================================================
# COLLECT PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

all_true = []
all_probabilities = []
all_masks = []

for i in range(len(test_generator)):

    X, y, mask = test_generator[i]

    prediction = model.predict(
        X,
        verbose=0
    )

    all_true.append(
        np.argmax(y, axis=-1).flatten()
    )

    all_probabilities.append(
        prediction.reshape(-1, 3)
    )

    all_masks.append(
        mask.flatten()
    )

    if (i + 1) % 50 == 0:
        print(
            f"Processed "
            f"{i + 1}/{len(test_generator)}"
        )


true_class = np.concatenate(all_true)
probabilities = np.concatenate(all_probabilities)
masks = np.concatenate(all_masks)

valid = masks > 0

true_class = true_class[valid]
probabilities = probabilities[valid]

print(
    f"\nValid pixels: {len(true_class):,}"
)


# ============================================================
# THRESHOLD TEST
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

print(
    "\nThreshold   Precision   Recall      F1"
)

best_threshold = None
best_f1 = -1


for threshold in THRESHOLDS:

    # Start with Normal/Elevated decision
    predicted_class = np.argmax(
        probabilities[:, :2],
        axis=1
    )

    # Strong Bloom override
    strong_mask = (
        probabilities[:, 2] >= threshold
    )

    predicted_class[strong_mask] = 2

    actual_strong = (
        true_class == 2
    )

    predicted_strong = (
        predicted_class == 2
    )

    true_positive = np.sum(
        actual_strong &
        predicted_strong
    )

    false_positive = np.sum(
        (~actual_strong) &
        predicted_strong
    )

    false_negative = np.sum(
        actual_strong &
        (~predicted_strong)
    )

    precision = (
        true_positive /
        max(
            true_positive + false_positive,
            1
        )
    )

    recall = (
        true_positive /
        max(
            true_positive + false_negative,
            1
        )
    )

    f1 = (
        2 * precision * recall /
        max(
            precision + recall,
            1e-12
        )
    )

    print(
        f"{threshold:8.2f}"
        f"{precision * 100:11.2f}%"
        f"{recall * 100:10.2f}%"
        f"{f1 * 100:9.2f}%"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"\nRecommended threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best Strong Bloom F1: "
    f"{best_f1 * 100:.2f}%"
)

print("\n" + "=" * 70)
print("✅ THRESHOLD TUNING COMPLETE")
print("=" * 70)