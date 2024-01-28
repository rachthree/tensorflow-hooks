"""Test utilities."""
import tensorflow as tf


def create_single_layer_model():
    """Create a TF Keras model with a single layer."""
    x = tf.keras.Input((128, 128, 3))
    y = tf.keras.layers.Conv2D(4, (3, 3))(x)
    return tf.keras.Model(x, y)
