import tensorflow as tf
from tensorflow.keras import layers
import random

def define_model(poi_count) -> tf.keras.Model:
    """
    Architect and Config
    """
    model = tf.keras.Sequential(
        [
            # keras.Input(shape=(2960, 1)),
            # Dense block to reduce the features
            # 1st block of CNN
            layers.Conv1D(64, 11, strides=1, padding="same", activation="relu", input_shape=(poi_count, 1)),
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


def define_random_model(poi_count):
    """
    Architect and Config for generating random models
    @param poi_count: number of points of interest (no. of selected features)
    @type poi_count: int
    """

    neurons = random.choice([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    layers_choice = random.choice([2, 3])
    activation = random.choice(["relu", "selu", "elu", "tanh"])
    kernel_initializer = random.choice(["random_uniform", "glorot_uniform", "he_uniform"])
    conv_layers = random.choice([1, 2, 3, 4])

    kernels = []
    strides = []
    filters = []

    for conv_layer in range(1, conv_layers + 1):
        kernels.append(random.choice([10, 12, 14, 16, 18, 20]))
        strides.append(random.choice([5, 10]))
        if conv_layer == 1:
            filters.append(random.choice([8, 16, 24, 32]))
        else:
            filters.append(filters[conv_layer - 2] * 2)

    random_cnn_hyperparameters = {
        "neurons": neurons,
        "layers": layers_choice,
        "activation": activation,
        "kernel_initializer": kernel_initializer,
        "conv_layers": conv_layers,
        "kernels": kernels,
        "strides": strides,
        "filters": filters,
    }

    model = tf.keras.Sequential()
    for conv_layer in range(1, conv_layers + 1):
        if conv_layer == 1:
            model.add(
                layers.Conv1D(kernel_size=kernels[conv_layer - 1], strides=strides[conv_layer - 1], filters=filters[conv_layer - 1],
                       activation=activation, input_shape=(poi_count, 1)))
        else:
            model.add(
                layers.Conv1D(kernel_size=kernels[conv_layer - 1], strides=strides[conv_layer - 1], filters=filters[conv_layer - 1],
                       activation=activation))

    model.add(layers.Flatten())
    model.add(layers.Dense(neurons, activation=activation, kernel_initializer=kernel_initializer))
    for i in range(layers_choice - 1):
        model.add(layers.Dense(neurons, activation=activation, kernel_initializer=kernel_initializer))
    model.add(layers.Dense(9, activation='softmax'))

    return model, random_cnn_hyperparameters


def rebuild_model(cnn_parameters, poi_count):
    """
    Rebuilding the model from saved model parameters
    @param cnn_parameters: The model parameters loaded from a file
    @type cnn_parameters: dict
    @param poi_count: number of points of interest
    @type poi_count: int
    @return: model with features
    @rtype: tf.keras.Model
    """
    neurons = cnn_parameters["neurons"]
    layers_choice = cnn_parameters["layers"]
    activation = cnn_parameters["activation"]
    kernel_initializer = cnn_parameters["kernel_initializer"]
    conv_layers = cnn_parameters["conv_layers"]
    kernels = cnn_parameters["kernels"]
    strides = cnn_parameters["strides"]
    filters = cnn_parameters["filters"]

    model = tf.keras.Sequential()
    for conv_layer in range(1, conv_layers + 1):
        if conv_layer == 1:
            model.add(
                layers.Conv1D(kernel_size=kernels[conv_layer - 1], strides=strides[conv_layer - 1],
                              filters=filters[conv_layer - 1],
                              activation=activation, input_shape=(poi_count, 1)))
        else:
            model.add(
                layers.Conv1D(kernel_size=kernels[conv_layer - 1], strides=strides[conv_layer - 1],
                              filters=filters[conv_layer - 1],
                              activation=activation))

    model.add(layers.Flatten())
    model.add(layers.Dense(neurons, activation=activation, kernel_initializer=kernel_initializer))
    for i in range(layers_choice - 1):
        model.add(layers.Dense(neurons, activation=activation, kernel_initializer=kernel_initializer))
    model.add(layers.Dense(9, activation='softmax'))
    return model
