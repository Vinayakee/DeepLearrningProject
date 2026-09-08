import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import tensorflow as tf

from models.convlstm import build_convlstm
from training.data_generator import HABDataGenerator


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 1

TEST_EPOCHS = 30

FULL_EPOCHS = 30

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# GPU / DEVICE INFORMATION
# ============================================================

print("=" * 70)
print("HAB CONVLSTM TRAINING")
print("=" * 70)

print("\nTensorFlow version:")
print(tf.__version__)

print("\nAvailable devices:")

for device in tf.config.list_physical_devices():
    print(" ", device)


# ============================================================
# LOAD DATA GENERATORS
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATA GENERATORS")
print("=" * 70)

train_generator = HABDataGenerator(
    split="train",
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_generator = HABDataGenerator(
    split="validation",
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTraining batches:", len(train_generator))
print("Validation batches:", len(validation_generator))


# ============================================================
# CLASS WEIGHTS
# ============================================================
#
# Class distribution:
#
# 0 = Normal
# 1 = Elevated Risk
# 2 = Strong Bloom Risk
#
# Strong bloom is rarer, so we give it more importance.
# ============================================================

CLASS_WEIGHTS = tf.constant(
    [1.0, 2.0, 3.0],
    dtype=tf.float32
)


# ============================================================
# MASKED + CLASS-WEIGHTED LOSS
# ============================================================

def masked_weighted_loss(y_true, y_pred):
    """
    y_true:
        (batch, 5, 120, 84, 3)

    y_pred:
        (batch, 5, 120, 84, 3)

    Missing target pixels are represented by:
        [0, 0, 0]

    Therefore their loss becomes zero.
    """

    # Prevent log(0)
    y_pred = tf.clip_by_value(
        y_pred,
        1e-7,
        1.0
    )

    # Pixel-wise categorical cross entropy
    ce = -tf.reduce_sum(
        y_true * tf.math.log(y_pred),
        axis=-1
    )

    # Determine valid target pixels
    valid_mask = tf.reduce_sum(
        y_true,
        axis=-1
    )

    # Class weighting
    pixel_weights = tf.reduce_sum(
        y_true * CLASS_WEIGHTS,
        axis=-1
    )

    weighted_loss = (
        ce * pixel_weights
    )

    # Ignore missing pixels
    weighted_loss = (
        weighted_loss * valid_mask
    )

    # Average only over valid pixels
    total_loss = tf.reduce_sum(
        weighted_loss
    )

    valid_pixels = tf.reduce_sum(
        valid_mask
    )

    return total_loss / (
        valid_pixels + 1e-7
    )


# ============================================================
# MASKED ACCURACY
# ============================================================

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

    correct = (
        correct * valid_mask
    )

    return (
        tf.reduce_sum(correct)
        /
        (tf.reduce_sum(valid_mask) + 1e-7)
    )


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)

model = build_convlstm()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=masked_weighted_loss,
    metrics=[
        masked_accuracy
    ]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_convlstm.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        os.path.join(
            OUTPUT_DIR,
            "training_history.csv"
        )
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print("\nBatch size:", BATCH_SIZE)

print(
    "Training samples:",
    train_generator.num_samples
)

print(
    "Validation samples:",
    validation_generator.num_samples
)

print(
    "Test epochs:",
    TEST_EPOCHS
)

print("\nStarting 1-epoch test run...")


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=TEST_EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# TEST RESULT
# ============================================================

print("\n" + "=" * 70)
print("1-EPOCH TRAINING TEST COMPLETE")
print("=" * 70)

print(
    "Training loss:",
    history.history["loss"][-1]
)

print(
    "Validation loss:",
    history.history["val_loss"][-1]
)

print(
    "Training accuracy:",
    history.history["masked_accuracy"][-1]
)

print(
    "Validation accuracy:",
    history.history["val_masked_accuracy"][-1]
)

print(
    "\nBest model saved to:",
    checkpoint_path
)

print("\n" + "=" * 70)
print("✅ TRAINING PIPELINE TEST PASSED")
print("=" * 70)

print(
    "\nIf the loss and validation loss are numbers, "
    "the training pipeline is working."
)

print(
    "\nNext step will be the full 30-epoch training."
)