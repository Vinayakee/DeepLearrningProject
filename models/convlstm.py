import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# MODEL SETTINGS
# ============================================================

INPUT_DAYS = 5
OUTPUT_DAYS = 5

HEIGHT = 120
WIDTH = 84
CHANNELS = 7

NUM_CLASSES = 3


# ============================================================
# BUILD CONVLSTM MODEL
# ============================================================

def build_convlstm():

    inputs = layers.Input(
        shape=(
            INPUT_DAYS,
            HEIGHT,
            WIDTH,
            CHANNELS
        ),
        name="environmental_input"
    )

    # --------------------------------------------------------
    # ConvLSTM Layer 1
    # --------------------------------------------------------
    # Learns both:
    #   - spatial patterns
    #   - temporal changes
    # --------------------------------------------------------

    x = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="convlstm_1"
    )(inputs)

    x = layers.BatchNormalization(
        name="batch_norm_1"
    )(x)

    # --------------------------------------------------------
    # ConvLSTM Layer 2
    # --------------------------------------------------------

    x = layers.ConvLSTM2D(
        filters=8,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="convlstm_2"
    )(x)

    x = layers.BatchNormalization(
        name="batch_norm_2"
    )(x)

    # --------------------------------------------------------
    # Convert features to 3 HAB classes
    # --------------------------------------------------------

    outputs = layers.TimeDistributed(
        layers.Conv2D(
            filters=NUM_CLASSES,
            kernel_size=(1, 1),
            padding="same",
            activation="softmax"
        ),
        name="hab_prediction"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="HAB_ConvLSTM"
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BUILDING HAB CONVLSTM MODEL")
    print("=" * 70)

    model = build_convlstm()

    print("\nMODEL SUMMARY")
    print("=" * 70)

    model.summary()

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    print("\nRunning test prediction...")

    dummy_input = tf.random.normal(
        (
            2,
            INPUT_DAYS,
            HEIGHT,
            WIDTH,
            CHANNELS
        )
    )

    prediction = model(
        dummy_input,
        training=False
    )

    print("\n" + "=" * 70)
    print("OUTPUT CHECK")
    print("=" * 70)

    print(
        "Input shape:",
        dummy_input.shape
    )

    print(
        "Output shape:",
        prediction.shape
    )

    print(
        "Output minimum:",
        float(tf.reduce_min(prediction))
    )

    print(
        "Output maximum:",
        float(tf.reduce_max(prediction))
    )

    # --------------------------------------------------------
    # Check probabilities
    # --------------------------------------------------------

    class_sum = tf.reduce_sum(
        prediction,
        axis=-1
    )

    print(
        "Probability sum minimum:",
        float(tf.reduce_min(class_sum))
    )

    print(
        "Probability sum maximum:",
        float(tf.reduce_max(class_sum))
    )

    # --------------------------------------------------------
    # Expected shape
    # --------------------------------------------------------

    expected_shape = (
        2,
        5,
        120,
        84,
        3
    )

    if (
        tuple(prediction.shape)
        == expected_shape
    ):

        print("\n" + "=" * 70)
        print("✅ CONVLSTM MODEL TEST PASSED")
        print("=" * 70)

        print(
            "Input : "
            "(batch, 5, 120, 84, 7)"
        )

        print(
            "Output: "
            "(batch, 5, 120, 84, 3)"
        )

        print(
            "✓ 5-day input"
        )

        print(
            "✓ 5-day output"
        )

        print(
            "✓ 7 environmental channels"
        )

        print(
            "✓ 3 HAB classes"
        )

        print(
            "✓ Probability maps"
        )

        print("=" * 70)

    else:

        print("\n❌ MODEL TEST FAILED")

        print(
            "Expected:",
            expected_shape
        )

        print(
            "Got:",
            tuple(prediction.shape)
        )