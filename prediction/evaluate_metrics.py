import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from training.data_generator import HABDataGenerator


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/best_convlstm.keras"

CLASS_NAMES = [
    "Normal",
    "Elevated Risk",
    "Strong Bloom Risk"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("HAB MODEL METRICS")
print("=" * 70)

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✓ Model loaded")


# ============================================================
# TEST DATA
# ============================================================

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
# CONFUSION MATRIX
# ============================================================

confusion = np.zeros(
    (3, 3),
    dtype=np.int64
)


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\nRunning predictions...")

for i in range(len(test_generator)):

    X, y, mask = test_generator[i]

    prediction = model.predict(
        X,
        verbose=0
    )

    true_class = np.argmax(
        y,
        axis=-1
    )

    predicted_class = np.argmax(
        prediction,
        axis=-1
    )

    valid = mask > 0

    true_flat = true_class[valid].flatten()
    pred_flat = predicted_class[valid].flatten()

    for true_value, pred_value in zip(
        true_flat,
        pred_flat
    ):
        confusion[
            true_value,
            pred_value
        ] += 1

    if (i + 1) % 50 == 0:
        print(
            f"Processed "
            f"{i + 1}/{len(test_generator)}"
        )


# ============================================================
# CALCULATE METRICS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION METRICS")
print("=" * 70)

precisions = []
recalls = []
f1_scores = []

for class_id in range(3):

    true_positive = confusion[
        class_id,
        class_id
    ]

    false_positive = (
        np.sum(confusion[:, class_id])
        - true_positive
    )

    false_negative = (
        np.sum(confusion[class_id, :])
        - true_positive
    )

    precision = (
        true_positive /
        max(true_positive + false_positive, 1)
    )

    recall = (
        true_positive /
        max(true_positive + false_negative, 1)
    )

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-12)
    )

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(
        f"\nClass {class_id}: "
        f"{CLASS_NAMES[class_id]}"
    )

    print(
        f"  Precision : {precision:.4f}"
    )

    print(
        f"  Recall    : {recall:.4f}"
    )

    print(
        f"  F1-score  : {f1:.4f}"
    )


# ============================================================
# MACRO METRICS
# ============================================================

macro_precision = np.mean(precisions)
macro_recall = np.mean(recalls)
macro_f1 = np.mean(f1_scores)

total_correct = np.trace(confusion)
total_pixels = np.sum(confusion)

accuracy = (
    total_correct /
    max(total_pixels, 1)
)

print("\n" + "=" * 70)
print("OVERALL METRICS")
print("=" * 70)

print(
    f"\nAccuracy          : {accuracy:.4f}"
)

print(
    f"Macro Precision   : {macro_precision:.4f}"
)

print(
    f"Macro Recall      : {macro_recall:.4f}"
)

print(
    f"Macro F1-score    : {macro_f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "\nRows = Actual"
)

print(
    "Columns = Predicted\n"
)

print(
    "                     Normal   Elevated   Strong"
)

for i in range(3):

    print(
        f"{CLASS_NAMES[i]:20s}"
        f"{confusion[i,0]:9d}"
        f"{confusion[i,1]:11d}"
        f"{confusion[i,2]:9d}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT HAB METRICS")
print("=" * 70)

print(
    f"Strong Bloom Recall : "
    f"{recalls[2] * 100:.2f}%"
)

print(
    f"Strong Bloom F1     : "
    f"{f1_scores[2] * 100:.2f}%"
)

print(
    f"Macro F1            : "
    f"{macro_f1 * 100:.2f}%"
)

print("\n" + "=" * 70)
print("✅ METRICS CALCULATION COMPLETE")
print("=" * 70)