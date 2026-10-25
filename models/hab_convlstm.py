"""Zone-level five-day HAB occurrence model; no chlorophyll threshold targets."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package='HAB')
class ZoneAverage(layers.Layer):
    def __init__(self, zone_masks, **kwargs):
        super().__init__(**kwargs)
        masks = np.asarray(zone_masks, np.float32)
        if masks.ndim != 3 or not np.isfinite(masks).all() or np.any(masks < 0) or np.any(masks.sum((1, 2)) <= 0):
            raise ValueError('Nonempty zone masks are required.')
        self.zone_masks = masks.tolist()
        self.normalized = tf.constant(masks / masks.sum((1, 2))[:, None, None])

    def call(self, inputs):
        return tf.einsum('bhwc,zhw->bzc', inputs, self.normalized)

    def get_config(self):
        return {**super().get_config(), 'zone_masks': self.zone_masks}


@tf.keras.utils.register_keras_serializable(package='HAB')
class MaskedBinaryLoss(tf.keras.losses.Loss):
    def __init__(self, positive_weight=1., name='masked_hab_binary_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        if not np.isfinite(positive_weight) or positive_weight <= 0:
            raise ValueError('Positive class weight must be finite and positive.')
        self.positive_weight = float(positive_weight)

    def call(self, y_true, y_pred):
        valid = tf.not_equal(y_true, 255.)
        y = tf.where(valid, y_true, 0.)
        p = tf.clip_by_value(y_pred, 1e-7, 1.-1e-7)
        loss = -(self.positive_weight*y*tf.math.log(p) + (1.-y)*tf.math.log(1.-p))
        return tf.math.divide_no_nan(tf.reduce_sum(tf.where(valid, loss, 0.)),
                                     tf.reduce_sum(tf.cast(valid, tf.float32)))

    def get_config(self):
        return {**super().get_config(), 'positive_weight': self.positive_weight}


def build_hab_convlstm(input_shape, zone_masks):
    days, height, width, channels = input_shape
    if height % 2 or width % 2 or channels != 16 or days < 1:
        raise ValueError('Even grid dimensions and 16 environmental/mask/calendar channels required.')
    if np.asarray(zone_masks).shape[1:] != (height, width):
        raise ValueError('Zone masks differ from the input grid.')
    inputs = layers.Input(input_shape, name='hab_environment')
    x = layers.TimeDistributed(layers.AveragePooling2D(2))(inputs)
    x = layers.ConvLSTM2D(8, 3, padding='same', return_sequences=True)(x)
    x = layers.LayerNormalization()(x)
    x = layers.ConvLSTM2D(4, 3, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    x = ZoneAverage(zone_masks, name='coastal_zone_features')(x)
    x = layers.Dense(5, activation='sigmoid', name='five_future_hab_scores')(x)
    outputs = layers.Permute((2, 1), name='daily_zone_hab_scores')(x)
    return tf.keras.Model(inputs, outputs, name='verified_hab_convlstm')
