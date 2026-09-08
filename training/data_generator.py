import os
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

PROCESSED_DIR = "data/processed"
SEQUENCE_DIR = "data/sequences"

INPUT_DAYS = 5
OUTPUT_DAYS = 5

HEIGHT = 120
WIDTH = 84

# Seven environmental variables
CHANNEL_NAMES = [
    "chlorophyll",
    "sst",
    "wind_u",
    "wind_v",
    "rainfall",
    "current_u",
    "current_v",
]


# ============================================================
# FILE NAMES
# ============================================================

DATA_FILES = {
    "chlorophyll": "chlorophyll_2021_2025.npy",
    "sst": "sst_2021_2025.npy",
    "wind_u": "wind_u_2021_2025.npy",
    "wind_v": "wind_v_2021_2025.npy",
    "rainfall": "rainfall_2021_2025.npy",
    "current_u": "current_u_2021_2025.npy",
    "current_v": "current_v_2021_2025.npy",
}

MASK_FILES = {
    "chlorophyll": "chlorophyll_mask_2021_2025.npy",
    "sst": "sst_mask_2021_2025.npy",
    "wind_u": "wind_u_mask_2021_2025.npy",
    "wind_v": "wind_v_mask_2021_2025.npy",
    "rainfall": "rainfall_mask_2021_2025.npy",
    "current_u": "current_u_mask_2021_2025.npy",
    "current_v": "current_v_mask_2021_2025.npy",
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading environmental data...")

DATA = {}
MASKS = {}

for name in CHANNEL_NAMES:

    data_path = os.path.join(
        PROCESSED_DIR,
        DATA_FILES[name]
    )

    mask_path = os.path.join(
        PROCESSED_DIR,
        MASK_FILES[name]
    )

    DATA[name] = np.load(
        data_path,
        mmap_mode="r"
    )

    MASKS[name] = np.load(
        mask_path,
        mmap_mode="r"
    )

    print(
        f"✓ {name}: {DATA[name].shape}"
    )


# ============================================================
# LOAD HAB LABELS
# ============================================================

LABELS = np.load(
    os.path.join(
        PROCESSED_DIR,
        "hab_labels_2021_2025.npy"
    ),
    mmap_mode="r"
)

print(
    f"✓ HAB labels: {LABELS.shape}"
)


# ============================================================
# LOAD SEQUENCE INDEX
# ============================================================

SEQUENCE_INDEX = pd.read_csv(
    os.path.join(
        SEQUENCE_DIR,
        "sequence_index.csv"
    )
)

print(
    f"✓ Sequence index: "
    f"{len(SEQUENCE_INDEX)} sequences"
)


# ============================================================
# LOAD TRAINING MEDIANS
# ============================================================

median_path = os.path.join(
    SEQUENCE_DIR,
    "training_medians.npy"
)

training_medians = np.load(
    median_path,
    allow_pickle=True
).item()

print("\nTraining fill values:")

for name in CHANNEL_NAMES:

    print(
        f"  {name}: "
        f"{training_medians[name]:.6f}"
    )


# ============================================================
# GENERATOR
# ============================================================

class HABDataGenerator(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        split="train",
        batch_size=2,
        shuffle=True
    ):

        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Select only requested split
        self.indices = SEQUENCE_INDEX[
            SEQUENCE_INDEX["split"] == split
        ].reset_index(drop=True)

        self.num_samples = len(
            self.indices
        )

        self.order = np.arange(
            self.num_samples
        )

        if self.shuffle:
            np.random.shuffle(
                self.order
            )

        print(
            f"\n{split.upper()} GENERATOR"
        )

        print(
            f"Samples: {self.num_samples}"
        )

        print(
            f"Batch size: {self.batch_size}"
        )


    # --------------------------------------------------------
    # Number of batches
    # --------------------------------------------------------

    def __len__(self):

        return int(
            np.ceil(
                self.num_samples
                / self.batch_size
            )
        )


    # --------------------------------------------------------
    # Shuffle after epoch
    # --------------------------------------------------------

    def on_epoch_end(self):

        self.order = np.arange(
            self.num_samples
        )

        if self.shuffle:

            np.random.shuffle(
                self.order
            )


    # --------------------------------------------------------
    # Load one sample
    # --------------------------------------------------------

    def _load_sample(
        self,
        row
    ):

        input_start = int(
            row["input_start_idx"]
        )

        target_start = int(
            row["target_start_idx"]
        )

        # --------------------------------------------
        # INPUT
        #
        # Shape:
        # (5, 120, 84, 7)
        # --------------------------------------------

        X = np.zeros(
            (
                INPUT_DAYS,
                HEIGHT,
                WIDTH,
                len(CHANNEL_NAMES)
            ),
            dtype=np.float32
        )

        # --------------------------------------------
        # LOAD EACH INPUT DAY
        # --------------------------------------------

        for t in range(INPUT_DAYS):

            day_index = (
                input_start + t
            )

            for c, name in enumerate(
                CHANNEL_NAMES
            ):

                values = np.asarray(
                    DATA[name][day_index],
                    dtype=np.float32
                )

                mask = np.asarray(
                    MASKS[name][day_index]
                )

                # Copy so the memory-mapped
                # source is never modified.
                values = values.copy()

                # ------------------------------------
                # Missing values
                #
                # Fill ONLY missing observations
                # with training median.
                # ------------------------------------

                missing = (
                    ~np.isfinite(values)
                    | (mask == 0)
                )

                values[missing] = (
                    training_medians[name]
                )

                # Safety check
                values[
                    ~np.isfinite(values)
                ] = training_medians[name]

                X[
                    t,
                    :,
                    :,
                    c
                ] = values


        # --------------------------------------------
        # TARGET
        #
        # Shape:
        # (5, 120, 84)
        #
        # Classes:
        # 0 = normal
        # 1 = elevated
        # 2 = strong bloom
        # 255 = missing
        # --------------------------------------------

        y = np.zeros(
            (
                OUTPUT_DAYS,
                HEIGHT,
                WIDTH,
                3
            ),
            dtype=np.float32
        )

        sample_mask = np.zeros(
            (
                OUTPUT_DAYS,
                HEIGHT,
                WIDTH
            ),
            dtype=np.float32
        )

        for t in range(OUTPUT_DAYS):

            day_index = (
                target_start + t
            )

            target = np.asarray(
                LABELS[day_index],
                dtype=np.uint8
            )

            # Valid labels
            valid = (
                target != 255
            )

            # Store valid target mask
            sample_mask[t] = (
                valid.astype(np.float32)
            )

            # ----------------------------------------
            # One-hot encode valid classes
            # ----------------------------------------

            for class_id in range(3):

                y[t, :, :, class_id] = (
                    target == class_id
                ).astype(np.float32)

            # ----------------------------------------
            # Missing pixels have zero target
            # and zero weight.
            # ----------------------------------------

            y[t][~valid] = 0.0


        return X, y, sample_mask


    # --------------------------------------------------------
    # GET BATCH
    # --------------------------------------------------------

    def __getitem__(
        self,
        batch_index
    ):

        start = (
            batch_index
            * self.batch_size
        )

        end = min(
            start + self.batch_size,
            self.num_samples
        )

        batch_positions = self.order[
            start:end
        ]

        batch_rows = self.indices.iloc[
            batch_positions
        ]

        batch_size_actual = len(
            batch_rows
        )

        # --------------------------------------------
        # Allocate batch
        # --------------------------------------------

        X_batch = np.zeros(
            (
                batch_size_actual,
                INPUT_DAYS,
                HEIGHT,
                WIDTH,
                len(CHANNEL_NAMES)
            ),
            dtype=np.float32
        )

        y_batch = np.zeros(
            (
                batch_size_actual,
                OUTPUT_DAYS,
                HEIGHT,
                WIDTH,
                3
            ),
            dtype=np.float32
        )

        mask_batch = np.zeros(
            (
                batch_size_actual,
                OUTPUT_DAYS,
                HEIGHT,
                WIDTH
            ),
            dtype=np.float32
        )

        # --------------------------------------------
        # Fill batch
        # --------------------------------------------

        for i, (_, row) in enumerate(
            batch_rows.iterrows()
        ):

            X, y, mask = self._load_sample(
                row
            )

            X_batch[i] = X
            y_batch[i] = y
            mask_batch[i] = mask

        return (
            X_batch,
            y_batch,
            mask_batch
        )


# ============================================================
# TEST GENERATOR
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("TESTING DATA GENERATOR")
    print("=" * 70)

    generator = HABDataGenerator(
        split="train",
        batch_size=2,
        shuffle=False
    )

    print(
        f"\nNumber of batches: "
        f"{len(generator)}"
    )

    print(
        "\nLoading first batch..."
    )

    X, y, mask = generator[0]

    print("\n" + "=" * 70)
    print("BATCH SHAPES")
    print("=" * 70)

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "mask shape:",
        mask.shape
    )

    # ========================================================
    # CHECK VALUES
    # ========================================================

    print("\n" + "=" * 70)
    print("VALUE CHECK")
    print("=" * 70)

    print(
        "X min:",
        np.min(X)
    )

    print(
        "X max:",
        np.max(X)
    )

    print(
        "X mean:",
        np.mean(X)
    )

    print(
        "NaNs in X:",
        np.isnan(X).sum()
    )

    print(
        "Infs in X:",
        np.isinf(X).sum()
    )

    print(
        "Valid target pixels:",
        int(mask.sum())
    )

    print(
        "Missing target pixels:",
        int(mask.size - mask.sum())
    )

    # ========================================================
    # CHECK ONE-HOT TARGET
    # ========================================================

    target_sum = np.sum(
        y,
        axis=-1
    )

    valid_target_sum = (
        target_sum[mask == 1]
    )

    if len(valid_target_sum) > 0:

        print(
            "One-hot target check:"
        )

        print(
            "  Minimum:",
            valid_target_sum.min()
        )

        print(
            "  Maximum:",
            valid_target_sum.max()
        )

    # ========================================================
    # FINAL
    # ========================================================

    expected_x = (
        2,
        5,
        120,
        84,
        7
    )

    expected_y = (
        2,
        5,
        120,
        84,
        3
    )

    expected_mask = (
        2,
        5,
        120,
        84
    )

    if (
        X.shape == expected_x
        and y.shape == expected_y
        and mask.shape == expected_mask
        and not np.isnan(X).any()
        and not np.isinf(X).any()
    ):

        print("\n" + "=" * 70)
        print("✅ DATA GENERATOR TEST PASSED")
        print("=" * 70)

        print(
            "Input: 5 days × 120 × 84 × 7"
        )

        print(
            "Target: 5 days × 120 × 84 × 3"
        )

        print(
            "Missing target mask: working"
        )

        print(
            "Missing input values: handled"
        )

        print(
            "Ready for ConvLSTM"
        )

        print("=" * 70)

    else:

        print("\n❌ DATA GENERATOR TEST FAILED")