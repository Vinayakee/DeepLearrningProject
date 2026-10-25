"""Reuse the original compact trunk with binary satellite-event future heads."""
import tensorflow as tf
from models.convlstm import build_convlstm
from models.hab_convlstm import MaskedBinaryLoss


def build_bloom_event_model():
    base=build_convlstm(channels=16)
    trunk=base.get_layer('forecast_logits').input
    scores=tf.keras.layers.Conv2D(5,1,activation='sigmoid',name='satellite_event_scores')(trunk)
    outputs=tf.keras.layers.Permute((3,1,2),name='five_daily_event_maps')(scores)
    return tf.keras.Model(base.input,outputs,name='Satellite_Bloom_Event_ConvLSTM')
