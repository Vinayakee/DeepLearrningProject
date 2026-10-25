"""Compact ConvLSTM with a separate future head for each forecast day."""
import tensorflow as tf
from tensorflow.keras import layers, models
import config

INPUT_DAYS = config.INPUT_DAYS
OUTPUT_DAYS = config.OUTPUT_DAYS
HEIGHT, WIDTH = config.HEIGHT, config.WIDTH
CHANNELS = 14  # seven scaled environmental fields and seven availability masks
NUM_CLASSES = 3


def build_convlstm(filters=(8, 4), spatial_downsample=2, channels=CHANNELS):
    if HEIGHT % spatial_downsample or WIDTH % spatial_downsample:
        raise ValueError("Spatial downsampling must divide both grid dimensions.")
    inputs = layers.Input((INPUT_DAYS, HEIGHT, WIDTH, channels), name="environmental_input")
    x = inputs
    if spatial_downsample > 1:
        x = layers.TimeDistributed(layers.AveragePooling2D(spatial_downsample), name="spatial_pool")(x)
    x = layers.ConvLSTM2D(filters[0], 3, padding="same", return_sequences=True,
                          activation="tanh", recurrent_activation="sigmoid", name="convlstm_1")(x)
    # Layer normalization is stable for the small CPU-friendly training batches.
    x = layers.LayerNormalization(name="normalization_1")(x)
    x = layers.ConvLSTM2D(filters[1], 3, padding="same", return_sequences=False,
                          activation="tanh", recurrent_activation="sigmoid", name="convlstm_2")(x)
    x = layers.LayerNormalization(name="normalization_2")(x)
    if spatial_downsample > 1:
        x = layers.UpSampling2D(spatial_downsample, interpolation="bilinear", name="restore_grid")(x)
    # All future heads receive the final state, which encodes the complete input.
    x = layers.Conv2D(OUTPUT_DAYS * NUM_CLASSES, 1, name="forecast_logits")(x)
    x = layers.Reshape((HEIGHT, WIDTH, OUTPUT_DAYS, NUM_CLASSES))(x)
    x = layers.Permute((3, 1, 2, 4))(x)
    outputs = layers.Softmax(axis=-1, name="hab_prediction")(x)
    return models.Model(inputs, outputs, name="HAB_ConvLSTM")


if __name__ == "__main__":
    model = build_convlstm()
    model.summary()
    prediction = model(tf.zeros((1, INPUT_DAYS, HEIGHT, WIDTH, CHANNELS)))
    assert prediction.shape == (1, OUTPUT_DAYS, HEIGHT, WIDTH, NUM_CLASSES)
    tf.debugging.assert_near(tf.reduce_sum(prediction, -1), tf.ones(prediction.shape[:-1]))
    print("Model shape/probability checks passed.")
