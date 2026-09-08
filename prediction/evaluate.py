import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from training.data_generator import HABDataGenerator


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/best_convlstm.keras"


# ============================================================
# LOSS FUNCTION
# ============================================================

CLASS_WEIGHTS = tf.constant(
    [1.0, 2.0, 3.0],
    dtype=tf.float32
)


def masked_weighted_loss(y_true, y_pred):

    y_pred = tf.clip_by_value(
        y_pred,
        1e-7,
        1.0
    )

    ce = -tf.reduce_sum(
        y_true * tf.math.log(y_pred),
        axis=-1
    )

    valid_mask = tf.reduce_sum(
        y_true,
        axis=-1
    )

    pixel_weights = tf.reduce_sum(
        y_true * CLASS_WEIGHTS,
        axis=-1
    )

    weighted_loss = (
        ce
        * pixel_weights
        * valid_mask
    )

    return (
        tf.reduce_sum(weighted_loss)
        /
        (tf.reduce_sum(valid_mask) + 1e-7)
    )


def masked_accuracy(y_true, y_pred):

    true_class = tf.argmax(
        y_true,
        axis=-1
    )

    predicted_class = tf.argmax(
        y_pred,
        axis=-1
    )

    correct = tf.cast(
        tf.equal(
            true_class,
            predicted_class
        ),
        tf.float32
    )

    valid_mask = tf.reduce_sum(
        y_true,
        axis=-1
    )

    correct *= valid_mask

    return (
        tf.reduce_sum(correct)
        /
        (tf.reduce_sum(valid_mask) + 1e-7)
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("HAB CONVLSTM TEST EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "masked_weighted_loss":
                masked_weighted_loss,
            "masked_accuracy":
                masked_accuracy
        }
    )

    print("✓ Model loaded")

    # --------------------------------------------------------
    # TEST GENERATOR
    # --------------------------------------------------------

    print("\nCreating test generator...")

    test_generator = HABDataGenerator(
        split="test",
        batch_size=1,
        shuffle=False
    )

    print(
        f"Test samples: "
        f"{test_generator.num_samples}"
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING 2025 TEST SET")
    print("=" * 70)

    total_correct = 0
    total_valid = 0

    class_correct = np.zeros(3)
    class_total = np.zeros(3)

    # Confusion matrix
    confusion = np.zeros(
        (3, 3),
        dtype=np.int64
    )

    for i in range(
        len(test_generator)
    ):

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

        # Overall accuracy
        total_correct += np.sum(
            predicted_class[valid]
            ==
            true_class[valid]
        )

        total_valid += np.sum(valid)

        # Per-class statistics
        for class_id in range(3):

            class_pixels = (
                valid
                &
                (true_class == class_id)
            )

            class_total[class_id] += np.sum(
                class_pixels
            )

            class_correct[class_id] += np.sum(
                predicted_class[class_pixels]
                ==
                class_id
            )

        # Confusion matrix
        true_flat = true_class[valid].flatten()
        pred_flat = predicted_class[valid].flatten()

        for t, p in zip(
            true_flat,
            pred_flat
        ):

            confusion[t, p] += 1

        if (i + 1) % 25 == 0:

            print(
                f"Processed "
                f"{i + 1}/"
                f"{len(test_generator)}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    overall_accuracy = (
        total_correct
        /
        max(total_valid, 1)
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"\nValid test pixels: "
        f"{total_valid:,}"
    )

    print(
        f"Overall accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    class_names = [
        "Normal",
        "Elevated Risk",
        "Strong Bloom Risk"
    ]

    print("\nPer-class accuracy:")

    for class_id in range(3):

        accuracy = (
            class_correct[class_id]
            /
            max(class_total[class_id], 1)
        )

        print(
            f"Class {class_id} "
            f"({class_names[class_id]}): "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"  Pixels: "
            f"{int(class_total[class_id]):,}"
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

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
        "                 Normal  Elevated  Strong"
    )

    for i, name in enumerate(
        class_names
    ):

        print(
            f"{name:18s}"
            f"{confusion[i, 0]:8d}"
            f"{confusion[i, 1]:10d}"
            f"{confusion[i, 2]:8d}"
        )

    print("\n" + "=" * 70)
    print("✅ TEST EVALUATION COMPLETE")
    print("=" * 70)