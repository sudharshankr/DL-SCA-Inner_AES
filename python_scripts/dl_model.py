import tensorflow as tf
from tensorflow.keras import layers


def define_model() -> tf.keras.Model:
    """
    Architect and Config
    """
    model = tf.keras.Sequential(
        [
            # keras.Input(shape=(2960, 1)),
            # Dense block to reduce the features
            # 1st block of CNN
            layers.Conv1D(64, 11, strides=1, padding="same", activation="relu", input_shape=(2960, 1)),
            layers.AveragePooling1D(pool_size=2, strides=2, padding="valid"),

            # 2nd block
            layers.Conv1D(128, 11, strides=1, padding="same", activation="relu"),
            layers.AveragePooling1D(pool_size=2, strides=2, padding="valid"),

            # 3rd block
            layers.Conv1D(256, 11, strides=1, padding="same", activation="relu"),
            layers.AveragePooling1D(pool_size=2, strides=2, padding="valid"),

            # 4th block
            layers.Conv1D(512, 11, strides=1, padding="same", activation="relu"),
            layers.AveragePooling1D(pool_size=2, strides=2, padding="valid"),

            # 5th block
            layers.Conv1D(512, 11, strides=1, padding="same", activation="relu"),
            layers.AveragePooling1D(pool_size=2, strides=2, padding="valid"),

            # Flattening layer
            layers.Flatten(),

            # FC layer and output
            layers.Dense(4096, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(9, activation='softmax'),
        ]
    )

    return model
