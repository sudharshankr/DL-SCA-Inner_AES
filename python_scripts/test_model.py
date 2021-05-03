# -*- coding: utf-8 -*-

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from aeskeyschedule import key_schedule
import time
from tqdm import tqdm

from funcs import write_to_npz, widgets
from leakage_models import leakage_model_round_3
from dl_model import define_model


def return_idx(key, d, g):
    return int('{0:08b}'.format(key) + '{0:08b}'.format(d) + '{0:08b}'.format(g), 2)


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


def run_attack(model: tf.keras.Model, attack_traces: np.array,
               attack_plaintexts: np.array, real_key: int, real_gamma: int, real_delta: int) -> (list, list, np.array):
    """
    Testing the model with the attack traces
    :param model: Model to be tested, this model should have weights already loaded
    :param attack_traces: The attack traces to be tested
    :param attack_plaintexts: The plaintext of corresponding attack traces
    :param real_key: The real key byte
    :return: The trend in the rank of the real key byte with increasing traces, a count of the traces,
            and the final list of key probabilities
    """
    P_k = np.zeros(256 * 256 * 256)
    traces_num = []
    rank_traces = []
    count = 0
    real_idx = return_idx(real_key, real_delta, real_gamma)
    key_guesses = np.array([n for n in range(256*256*256)])
    guesses_range = np.zeros((key_guesses.shape[0], 3))
    for i in range(key_guesses.shape[0]):
        f = '{0:024b}'.format(key_guesses[i])
        guesses_range[i][0] = int(f[:8], 2)
        guesses_range[i][1] = int(f[8:16], 2)
        guesses_range[i][2] = int(f[16:24], 2)
    guesses_range = guesses_range.astype("uint8")
    print("Created guesses list, now bruteforcing. Stand-by and better get some coffee...")
    # key_guess = delta_guess = gamma_guess = np.array([n for n in range(256)])
    # guesses_range = np.array(np.meshgrid(key_guess, delta_guess, gamma_guess)).T.reshape(-1, 3)
    for traces in tqdm(range(0, attack_traces.shape[0], 64), position=0, leave=True):
        start_time = time.time()
        predictions = model.predict(attack_traces[count:count + 64])
        # plaintexts = attack_plaintexts[count:count + 64]
        plaintexts = attack_plaintexts[count:count + 64, byte_attacked]
        # for j in range(len(predictions)):
        #     pt = plaintexts[j][byte_attacked]
        #     for k in range(256):
        #         for delta in range(256):
        #             for gamma in range(256):
        #                 proba = predictions[j][leakage_model(3, pt, k, gamma, delta)]
        #                 P_k[return_idx(k, delta, gamma)] += proba
        #     rank_traces.append(np.where(P_k.argsort()[::-1] == real_idx)[0])
        #     traces_num.append(count)
        for j in tqdm(range(len(predictions)), position=0, leave=True):
            hw = leakage_model_round_3(plaintexts[j], guesses_range)
            P_k += predictions[j][hw]
            # rank_traces.append(int('{0:024b}'.format(int(np.where(P_k.argsort()[::-1] == real_idx)[0]))[:8], 2))
            rank_traces.append(np.where(P_k.argsort()[::-1] == real_idx)[0])
            traces_num.append(j+count)
        loop_time = time.time() - start_time
        print("the loop time for 1 batch of 64 traces: ", loop_time)
        count += 64

    return rank_traces, traces_num, P_k


def plot_graph(ranks, traces_counts, key_probs):
    plt.figure(figsize=(15, 6))
    plt.title('Rank vs Traces Number')
    plt.xlabel('Number of traces')
    plt.ylabel('Rank')
    plt.grid(True)
    plt.plot(traces_counts, ranks)
    plt.show()

    guess = int('{0:024b}'.format(key_probs.argsort()[-1])[:8], 2)
    print("This is the key Guess: ", hex(guess), "and its rank: ", ranks[-1])
    if ranks[-1] == 0:
        print("Attack succeeded")


if __name__ == "__main__":
    model = define_model()
    weights_filename = sys.argv[1]
    model.load_weights(weights_filename)  # loading the weights into the model
    (attack_traces, attack_plaintexts, attack_keys) = load_attack_traces('../data/traces/round_3_traces.h5', 500)

    byte_attacked = 0
    real_key = attack_keys[0][byte_attacked]  # for fixed keys
    round_keys = key_schedule(attack_keys[0])
    real_gamma = round_keys[2][0]
    real_delta = round_keys[1][0]
    print("real key byte", real_key)
    (rank_progress, trace_count, key_probabilities) = run_attack(model, attack_traces, attack_plaintexts, real_key, real_gamma, real_delta)
    plot_graph(rank_progress, trace_count, key_probabilities)  # plotting the rank progress across the attack traces
    # write_to_npz("test_results.npz", rank_progress, trace_count, key_probabilities)
    print()