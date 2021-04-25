# -*- coding: utf-8 -*-

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# from sca_dl_train_model import define_model

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


def load_attack_traces(filename: str, num_traces_test: int) -> (np.array, np.array, np.array):
    """
    Loading traces required for the attack phase
    :param filename: Name of the file to load the traces from.
    :type filename: str
    :param num_traces_test: number of attack traces to fetch
    :type num_traces_test: int
    :return: Return the attack traces and the corresponding plaintext and key bytes
    :rtype: (np.array, np.array, np.array)
    """

    in_file = h5py.File(filename, "r")
    x_attack = np.array(in_file['Attack_traces/traces'])  # , dtype=np.int8)
    metadata_attack = np.array(in_file['Attack_traces/metadata'])

    x_attack = x_attack[:num_traces_test]  # reshape traces to num traces test
    plaintext_attack = metadata_attack['plaintext'][:num_traces_test]
    key_attack = metadata_attack['key'][:num_traces_test]

    x_attack = x_attack.reshape((x_attack.shape[0], x_attack.shape[1], 1))

    return x_attack, plaintext_attack, key_attack


def define_model() -> tf.keras.Model:
    """
    Define the model's architecture and configuration
    :return: Model
    :rtype: Keras Model
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


def run_attack(model: tf.keras.Model, attack_traces: np.array,
               attack_plaintexts: np.array, real_key: int) -> (list, list, np.array):
    """
    Testing the model with the attack traces
    :param model: Model to be tested, this model should have weights already loaded
    :param attack_traces: The attack traces to be tested
    :param attack_plaintexts: The plaintext of corresponding attack traces
    :param real_key: The real key byte
    :return: The trend in the rank of the real key byte with increasing traces, a count of the traces,
            and the final list of key probabilities
    """
    P_k = np.zeros(256)
    traces_num = []
    rank_traces = []
    count = 0

    for traces in range(0, attack_traces.shape[0], 200):
        predictions = model.predict(attack_traces[count:count + 200])
        plaintexts = attack_plaintexts[count:count + 200]
        for j in range(len(predictions)):
            for k in range(256):
                pt = plaintexts[j][byte_attacked]
                proba = predictions[j][AES_Sbox[pt ^ k]]
                P_k[k] += proba
            rank_traces.append(np.where(P_k.argsort()[::-1] == real_key)[0])
            traces_num.append(count)
        count += 200

    return rank_traces, traces_num, P_k


def plot_graph(ranks, traces_counts, key_probs):
    plt.figure(figsize=(15, 6))
    plt.title('Rank vs Traces Number')
    plt.xlabel('Number of traces')
    plt.ylabel('Rank')
    plt.grid(True)
    plt.plot(traces_counts, ranks)
    plt.show()

    guess = key_probs.argsort()[-1]
    print("This is the key Guess: ", hex(guess), "and its rank: ", ranks[-1])
    if ranks[-1] == 0:
        print("Attack succeeded")


if __name__ == "__main__":
    model = define_model()
    weights_filename = sys.argv[1]
    model.load_weights(weights_filename)  # loading the weights into the model
    (attack_traces, attack_plaintexts, attack_keys) = load_attack_traces('ASCAD_stored_traces.h5', 5000)

    byte_attacked = 2
    real_key = attack_keys[0][byte_attacked]  # for fixed keys
    print("real key byte", real_key)
    (rank_progress, trace_count, key_probabilities) = run_attack(model, attack_traces, attack_plaintexts, real_key)
    plot_graph(rank_progress, trace_count, key_probabilities)  # plotting the rank progress across the attack traces
