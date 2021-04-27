# -*- coding: utf-8 -*-

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.python.client import device_lib

device_lib.list_local_devices()


def load_traces_from_file(filename: str) -> (np.array, np.array, np.array, np.array):
    """
    Loading traces from file for pre-processing
    :param filename: name of the file to load from
    :type filename: str
    :return: profiling and validation traces along with their labels
    """

    in_file = h5py.File(filename, "r")
    # profiling_dataset = in_file['Profiling_traces/traces']
    x_profiling = np.array(in_file['Profiling_traces/traces'])  # , dtype=np.int8)
    y_profiling = np.array(in_file['Profiling_traces/labels'])

    x_validation = np.array(in_file['Validation_traces/traces'])
    y_validation = np.array(in_file['Validation_traces/labels'])

    # data must be reshape to fit models (batch, channel, length)

    x_profiling = x_profiling.reshape((x_profiling.shape[0], x_profiling.shape[1], 1))
    x_validation = x_validation.reshape((x_validation.shape[0], x_validation.shape[1], 1))
    return x_profiling, y_profiling, x_validation, y_validation


def define_model() -> tf.keras.Model:
    """
    Architect and Config
    """
    model = tf.keras.Sequential(
        [
            # keras.Input(shape=(200,1,700)),
            # 1st block of CNN
            layers.Conv1D(64, 11, strides=1, padding="same", activation="relu", input_shape=(700, 1)),
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
            layers.Dense(256, activation=tf.nn.log_softmax),
        ]
    )

    return model


if __name__ == "__main__":
    (profiling_traces, profiling_labels,
     validation_traces, validation_labels) = load_traces_from_file("../data/traces/ASCAD_stored_traces.h5")

    tf.config.list_physical_devices('GPU')  # checking the availability of GPU

    model = define_model()
    model.summary()  # print the summary of the model

    optimizer = tf.keras.optimizers.RMSprop(lr=0.00001)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    # workers = 8
    batch_size = 128
    num_classes = 256
    epochs = 100

    weights_file_name = sys.argv[1]

    model.fit(profiling_traces,
              to_categorical(profiling_labels, num_classes),
              validation_data=(validation_traces, to_categorical(validation_labels, num_classes)),
              epochs=epochs,
              batch_size=batch_size)

    model.save_weights(weights_file_name)    # save weights returned by the model

    # Save the model??
    # Save other logs??
    # #
    # # log = {
    # #     "train_cost": train_cost,
    # #     "train_acc": train_acc,
    # # }
    # # torch.save(log, "ASCAD_data/ASCAD_model/logs_3.pth")

    # the below code can be used if generator is needed

    # for feeding it in sequence, can be used for multiprocessing and increasing the number of workers
    # class SeqGen(keras.utils.Sequence):
    #
    #     def __init__(self, x_set, y_set, batch_size):
    #         self.x, self.y = x_set, y_set
    #         self.batch_size = batch_size
    #
    #     def __len__(self):
    #         return int(np.ceil(len(self.x) / float(self.batch_size)))
    #
    #     def __getitem__(self, idx):
    #         # print("Fetching batch {}".format(idx))
    #
    #         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    #         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    #
    #         return batch_x, batch_y

    # validation_labels1 = to_categorical(validation_labels, num_classes)
    # validation_data=SeqGen(validation_traces, validation_labels1, batch_size),

    # model.fit_generator(generator=SeqGen(X_profiling, Y_profiling, batch_size),
    #                     use_multiprocessing=True,
    #                     workers=workers,
    #                     callbacks = [],
    #                     epochs=epochs)





