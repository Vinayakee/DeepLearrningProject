import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_convlstm.keras"
)

SEQUENCE_INDEX = os.path.join(
    PROJECT_ROOT,
    "data",
    "sequences",
    "sequence_index.csv"
)

TRAINING_MEDIANS = os.path.join(
    PROJECT_ROOT,
    "data",
    "sequences",
    "training_medians.npy"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# Strong bloom threshold
STRONG_BLOOM_THRESHOLD = 0.50


# ============================================================
# START
# ============================================================

print("=" * 70)
print("HAB 5-DAY PREDICTION")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained ConvLSTM model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✓ Model loaded")

print(
    "Model input shape :",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# LOAD SEQUENCE INDEX
# ============================================================

print("\nLoading sequence information...")

df = pd.read_csv(
    SEQUENCE_INDEX
)

print(
    "Total sequences:",
    len(df)
)


test_df = df[
    df["split"] == "test"
].copy()


if len(test_df) == 0:

    raise RuntimeError(
        "No test sequences found."
    )


latest = test_df.iloc[-1]


print("\nLatest test sequence:")

print(latest)


# ============================================================
# LOAD TRAINING MEDIANS
# ============================================================

print("\nLoading training medians...")

training_medians = np.load(
    TRAINING_MEDIANS,
    allow_pickle=True
).item()

print(
    "Training median information:",
    training_medians
)


# Convert dictionary values into a 7-element array
channel_order = [
    "chlorophyll",
    "sst",
    "wind_u",
    "wind_v",
    "rainfall",
    "current_u",
    "current_v"
]

training_medians = np.array(
    [
        training_medians[channel]
        for channel in channel_order
    ],
    dtype=np.float32
)


print(
    "Training medians:",
    training_medians
)

print(
    "Training medians:",
    training_medians
)

print(
    "✓ Training medians loaded"
)


# ============================================================
# LOAD ENVIRONMENTAL DATA
# ============================================================

print("\nLoading environmental datasets...")


chl = np.load(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "chlorophyll_2021_2025.npy"
    )
)


sst = np.load(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "sst_2021_2025.npy"
    )
)


wind_u = np.load(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "wind_u_2021_2025.npy"
    )
)


wind_v = np.load(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "wind_v_2021_2025.npy"
    )
)


rainfall = np.load(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "rainfall_2021_2025.npy"
    )
)


current_u = np.load(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "current_u_2021_2025.npy"
    )
)


current_v = np.load(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "current_v_2021_2025.npy"
    )
)


print("✓ All datasets loaded")


# ============================================================
# LOAD DATES
# ============================================================

dates = np.loadtxt(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "dates_2021_2025.txt"
    ),
    dtype=str
)


# ============================================================
# FIND LAST 5 DAYS
# ============================================================

input_indices = np.arange(
    len(dates) - 5,
    len(dates)
)


print("\nInput dates:")


for i in input_indices:

    print(
        " ",
        dates[i]
    )


# ============================================================
# CREATE 7-CHANNEL INPUT
# ============================================================

X = np.stack(
    [
        chl[input_indices],
        sst[input_indices],
        wind_u[input_indices],
        wind_v[input_indices],
        rainfall[input_indices],
        current_u[input_indices],
        current_v[input_indices]
    ],
    axis=-1
)


print(
    "\nInput shape before cleaning:",
    X.shape
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

# This matches the training data generator.
# Missing values are replaced with the
# corresponding training-period median.

for channel in range(
    X.shape[-1]
):

    X[..., channel] = np.where(
        np.isfinite(
            X[..., channel]
        ),
        X[..., channel],
        training_medians[channel]
    )


# Final safety check

X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


print(
    "✓ Missing values handled using training medians"
)


# ============================================================
# CONVERT TO FLOAT32
# ============================================================

X = X.astype(
    np.float32
)


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

X = np.expand_dims(
    X,
    axis=0
)


print(
    "Final model input shape:",
    X.shape
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nRunning ConvLSTM prediction..."
)


prediction = model.predict(
    X,
    verbose=1
)


print(
    "✓ Prediction complete"
)


print(
    "Prediction shape:",
    prediction.shape
)


# ============================================================
# SAVE COMPLETE PREDICTION
# ============================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "latest_prediction.npy"
)


np.save(
    prediction_path,
    prediction
)


print(
    "\n✓ Saved prediction:"
)

print(
    prediction_path
)


# ============================================================
# STRONG BLOOM PROBABILITY
# ============================================================

# Class 2 = Strong Bloom Risk

strong_bloom_probability = (
    prediction[:, :, :, :, 2]
)


probability_path = os.path.join(
    OUTPUT_DIR,
    "strong_bloom_probability.npy"
)


np.save(
    probability_path,
    strong_bloom_probability
)


print(
    "✓ Saved strong bloom probability"
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "5-DAY PREDICTION COMPLETE"
)

print(
    "=" * 70
)


print(
    "Input days : 5"
)

print(
    "Input grid : 120 × 84"
)

print(
    "Channels   : 7"
)

print(
    "Forecast   : 5 days"
)

print(
    "Classes    : 3"
)


print(
    "\nPreprocessing:"
)

print(
    "✓ Training median imputation"
)

print(
    "✓ No additional normalization"
)


print(
    "\nStrong Bloom threshold:",
    STRONG_BLOOM_THRESHOLD
)


print(
    "Maximum Strong Bloom probability:",
    float(
        np.max(
            strong_bloom_probability
        )
    )
)


print(
    "Mean Strong Bloom probability:",
    float(
        np.mean(
            strong_bloom_probability
        )
    )
)


print(
    "\n✓ Prediction files created"
)

print(
    "=" * 70
)