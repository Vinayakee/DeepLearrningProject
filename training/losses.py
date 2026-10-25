"""Loss masking occurs once, inside the loss; generators return only X and y."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="HAB")
class MaskedWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights=(1., 1., 1.), name="masked_weighted_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = list(class_weights)

    def call(self, y_true, y_pred):
        valid = tf.reduce_sum(y_true, axis=-1)
        ce = -tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.)), axis=-1)
        weights = tf.reduce_sum(y_true * tf.constant(self.class_weights, tf.float32), axis=-1)
        return tf.math.divide_no_nan(tf.reduce_sum(ce * weights), tf.reduce_sum(valid))

    def get_config(self):
        return {**super().get_config(), "class_weights": self.class_weights}


@tf.keras.utils.register_keras_serializable(package="HAB")
class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.valid = self.add_weight(name="valid", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = tf.reduce_sum(y_true, axis=-1)
        correct = tf.cast(tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1)), tf.float32)
        self.correct.assign_add(tf.reduce_sum(correct * valid))
        self.valid.assign_add(tf.reduce_sum(valid))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.valid)

    def reset_state(self):
        self.correct.assign(0.)
        self.valid.assign(0.)
