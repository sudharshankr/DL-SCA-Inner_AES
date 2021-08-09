# -*- coding: utf-8 -*-
import configparser
import json
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from aeskeyschedule import key_schedule
import time
from tqdm import tqdm

from funcs import write_to_npz, widgets, hamming_lookup, aes_sbox, calc_round_key_byte, galois_mult_np, galois_mult
from leakage_models import *
from dl_model import define_model, rebuild_model
from calc_constants import calc_gamma, calc_delta, calc_theta


font = {'family': 'normal',
        'size': 11}

def return_idx_16(key, d):
    return int('{0:08b}'.format(key) + '{0:08b}'.format(d), 2)


def return_idx_24(key, d, g):
    return int('{0:08b}'.format(key) + '{0:08b}'.format(d) + '{0:08b}'.format(g), 2)


def return_idx_32(key, d, g, t):
    return int('{0:08b}'.format(key) + '{0:08b}'.format(d) + '{0:08b}'.format(g) + '{0:08b}'.format(t), 2)


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


def run_attack_round_1(model: tf.keras.Model, attack_traces: np.array,
                       attack_plaintexts: np.array, real_key: int, byte_attacked: int, batch_size: int) -> (
list, list, np.array):
    """
    Attack model for the first round of AES
    @param model:  Model to be tested, this model should have weights already loaded
    @param attack_traces: traces to run the attack on
    @param attack_plaintexts: Plaintexts corresponding to the attack traces
    @param real_key: the real key byte to be guessed
    @param byte_attacked: the index of the byte being attacked
    @param batch_size: the batch size required for the attack using the DL model
    @return: Key ranks, Count of traces and the probabilities of each key guess
    """
    P_k = np.zeros(256)
    traces_num = []
    rank_traces = []
    count = 0

    for traces in range(0, attack_traces.shape[0], batch_size):
        predictions = model.predict(attack_traces[count:count + batch_size])
        plaintexts = attack_plaintexts[count:count + batch_size]
        for j in range(len(predictions)):
            for k in range(256):
                pt = plaintexts[j][byte_attacked]
                proba = predictions[j][hamming_lookup[aes_sbox[pt ^ k]]]
                P_k[k] += proba
            rank_traces.append(np.where(P_k.argsort()[::-1] == real_key)[0])
            traces_num.append(count)
        count += batch_size

    return rank_traces, traces_num, P_k


def run_attack_round_2(model: tf.keras.Model, attack_traces: np.array,
                       attack_plaintexts: np.array, real_key: int, real_delta:int, byte_attacked: int, batch_size: int) -> (
list, list, np.array):
    """
    Attack model for the first round of AES
    @param model:  Model to be tested, this model should have weights already loaded
    @param attack_traces: traces to run the attack on
    @param attack_plaintexts: Plaintexts corresponding to the attack traces
    @param real_key: the real key byte to be guessed
    @param byte_attacked: the index of the byte being attacked
    @param batch_size: the batch size required for the attack using the DL model
    @return: Key ranks, Count of traces and the probabilities of each key guess
    """
    P_k = np.zeros(256 * 256)
    traces_num = []
    rank_traces = []
    count = 0
    real_idx = return_idx_16(real_key, real_delta)
    key_guesses = np.array([n for n in range(256 * 256)])
    guesses_range = np.load("Guesses_range_16.npy")
    print("Created guesses list, now bruteforcing. Stand-by and better get some coffee...")

    for traces in tqdm(range(0, attack_traces.shape[0], batch_size), position=0, leave=True):
        start_time = time.time()
        predictions = model.predict(attack_traces[count:count + batch_size])
        plaintexts = attack_plaintexts[count:count + batch_size, byte_attacked]
        start_time = time.time()
        # TODO: make this faster using a batch computation strategy. Start with the below line
        # hw = calc_hypothesis_round_3_batch(plaintexts, guesses_range, batch_size)
        print(time.time() - start_time)
        for j in tqdm(range(len(predictions)), position=0, leave=True):
            hw = calc_hypothesis_round_2(plaintexts[j], guesses_range)
            P_k += predictions[j][hw]
            # rank_traces.append(int('{0:024b}'.format(int(np.where(P_k.argsort()[::-1] == real_idx)[0]))[:8], 2))
            rank_traces.append(np.where(P_k.argsort()[::-1] == real_idx)[0])
            traces_num.append(j + count)
        loop_time = time.time() - start_time
        print("the loop time for 1 batch of 64 traces: ", loop_time / 60)
        count += batch_size

    return rank_traces, traces_num, P_k

def run_attack_round_3(model: tf.keras.Model, attack_traces: np.array,
                       attack_plaintexts: np.array, real_key: int, real_gamma: int, real_delta: int,
                       batch_size: int) -> (list, list, np.array):
    """
    Testing the model with the attack traces for round 3
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
    real_idx = return_idx_24(real_key, real_delta, real_gamma)
    key_guesses = np.array([n for n in range(256 * 256 * 256)])
    # guesses_range = np.zeros((key_guesses.shape[0], 3))
    # for i in range(key_guesses.shape[0]):
    #     f = '{0:024b}'.format(key_guesses[i])
    #     guesses_range[i][0] = int(f[:8], 2)
    #     guesses_range[i][1] = int(f[8:16], 2)
    #     guesses_range[i][2] = int(f[16:24], 2)
    # guesses_range = guesses_range.astype("uint8")
    guesses_range = np.load("Guesses_range_24.npy")
    print("Created guesses list, now bruteforcing. Stand-by and better get some coffee...")

    for traces in tqdm(range(0, attack_traces.shape[0], batch_size), position=0, leave=True):
        start_time = time.time()
        predictions = model.predict(attack_traces[count:count + batch_size])
        plaintexts = attack_plaintexts[count:count + batch_size, byte_attacked]
        start_time = time.time()
        # TODO: make this faster using a batch computation strategy. Start with the below line
        # hw = calc_hypothesis_round_3_batch(plaintexts, guesses_range, batch_size)
        print(time.time() - start_time)
        for j in tqdm(range(len(predictions)), position=0, leave=True):
            hw = calc_hypothesis_round_3(plaintexts[j], guesses_range)
            P_k += predictions[j][hw]
            # rank_traces.append(int('{0:024b}'.format(int(np.where(P_k.argsort()[::-1] == real_idx)[0]))[:8], 2))
            rank_traces.append(np.where(P_k.argsort()[::-1] == real_idx)[0])
            traces_num.append(j + count)
        loop_time = time.time() - start_time
        print("the loop time for 1 batch of 64 traces: ", loop_time/60)
        count += batch_size

    return rank_traces, traces_num, P_k


def run_attack_round_4(model: tf.keras.Model, attack_traces: np.array,
                       attack_plaintexts: np.array, real_key: int, real_gamma: int, real_delta: int, real_theta: int,
                       batch_size: int) -> (list, list, np.array):
    """
    Testing the model with the attack traces for round 4
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
    real_idx = return_idx_24(real_key, real_delta, real_gamma)
    # key_guesses = np.array([n for n in range(256 * 256 * 256 * 256)])
    # guesses_range = np.zeros((key_guesses.shape[0], 4))
    # print("Preparing guesses...")
    # for i in range(key_guesses.shape[0]):
    #     f = '{0:032b}'.format(key_guesses[i])
    #     guesses_range[i][0] = int(f[:8], 2)
    #     guesses_range[i][1] = int(f[8:16], 2)
    #     guesses_range[i][2] = int(f[16:24], 2)
    #     guesses_range[i][3] = int(f[24:32], 2)
    # guesses_range = guesses_range.astype("uint8")
    # np.save("Guesses_range_32.npy", guesses_range)
    guesses_range = np.load("Guesses_range_24.npy")
    print("Created guesses list, now bruteforcing. Stand-by and better get some coffee...")
    for traces in tqdm(range(0, attack_traces.shape[0], batch_size), position=0, leave=True):
        start_time = time.time()
        predictions = model.predict(attack_traces[count:count + batch_size])
        plaintexts = attack_plaintexts[count:count + batch_size, byte_attacked]
        start_time = time.time()
        # TODO: make this faster using a batch computation strategy. Start with the below line
        # hw = calc_hypothesis_round_3_batch(plaintexts, guesses_range, batch_size)
        # print(time.time() - start_time)
        for j in tqdm(range(len(predictions)), position=0, leave=True):
            hw = calc_hypothesis_round_4(plaintexts[j], guesses_range, real_theta)
            P_k += predictions[j][hw]
            # rank_traces.append(int('{0:024b}'.format(int(np.where(P_k.argsort()[::-1] == real_idx)[0]))[:8], 2))
            rank_traces.append(np.where(P_k.argsort()[::-1] == real_idx)[0])
            traces_num.append(j + count)
        loop_time = time.time() - start_time
        print("the loop time for 1 batch of 64 traces: ", loop_time/60)
        count += batch_size

    return rank_traces, traces_num, P_k



def plot_graph(ranks, traces_counts, key_probs, filename):
    plt.rc('font', **font)
    plt.figure(figsize=(10, 6))
    plt.title('Rank vs Traces Number')
    plt.xlabel('Number of traces')
    plt.ylabel('Rank')
    plt.grid(True)
    plt.plot(traces_counts, ranks)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename, pad_inches=0, bbox_inches='tight', dpi=300)
    # guess = int('{0:024b}'.format(key_probs.argsort()[-1])[:8], 2)
    guess = key_probs.argsort()[-1]
    print("This is the key Guess: ", hex(guess), "and its rank: ", ranks[-1])
    if ranks[-1] == 0:
        print("Attack succeeded")


if __name__ == "__main__":
    # Loading configuration
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read('config.ini')
    attack_traces_count = config['Traces'].getint('AttackEnd') - config['Traces'].getint('AttackStart')
    leakage_config = config['Leakage']
    training_config = config['Training']
    weights_filename = training_config['WeightsFilename']
    model_parameters_file = training_config['ModelParametersFile']
    traces_filename = config['TRS']['TracesStorageFile']
    byte_attacked = leakage_config.getint('TargetKeyByteIndex')
    leakage = leakage_config.getint('LeakageRound')
    hypothesis = leakage_config.getint('HypothesisRound')
    hyp_type = leakage_config['HypothesisType']
    batch_size = training_config.getint('BatchSize')
    model_id = training_config["ModelId"]
    results_filename = "../data/attack_results/round_"+str(hypothesis)+"_random_model_results/results-model_"+model_id+"-leakage_rnd_" + str(leakage) \
                       + "-hypothesis_rnd_" + str(hypothesis) + "-" + hyp_type + "-" + str(byte_attacked) + ".npz"

    plot_filename = "../../result_images/random_cnn_plots/round_"+str(hypothesis)+"/results-model_"+model_id+"-hypothesis_rnd_"+str(hypothesis)
    # Starting attack
    (attack_traces, attack_plaintexts, attack_keys) = load_attack_traces(traces_filename, attack_traces_count)  # loading attack traces
    with open(model_parameters_file, "r") as json_file:
        cnn_parameters = json.load(json_file)
    model = rebuild_model(cnn_parameters, attack_traces.shape[1])

    # model = define_model(attack_traces.shape[1])
    model.load_weights(weights_filename)  # loading the weights into the model
    # (attack_traces, attack_plaintexts, attack_keys) = load_attack_traces(traces_filename, 500)  # loading attack traces
    real_key = attack_keys[0][byte_attacked]  # for fixed keys
    print("real key byte", real_key)

    if hypothesis == 1:
        (rank_progress, trace_count, key_probabilities) = run_attack_round_1(model, attack_traces, attack_plaintexts,
                                                                             real_key, byte_attacked, batch_size)
        plot_graph(rank_progress, trace_count, key_probabilities)  # plotting the rank progress across the attack traces
        write_to_npz(results_filename, rank_progress, trace_count, key_probabilities)

    elif hypothesis == 2:
        constant_byte = 0
        round_keys = key_schedule(attack_keys[0])
        real_delta = galois_mult(aes_sbox[constant_byte ^ attack_keys[0][5]], 3) \
                     ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][10]], 1) \
                     ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][15]], 1) ^ round_keys[1][0]
        (rank_progress, trace_count, key_probabilities) = run_attack_round_2(model, attack_traces, attack_plaintexts,
                                                                             real_key, real_delta, byte_attacked, batch_size)

        plot_graph(rank_progress, trace_count, key_probabilities, plot_filename)  # plotting the rank progress across the attack traces
        write_to_npz(results_filename, rank_progress, trace_count, key_probabilities)


    elif hypothesis == 3:
        constant_byte = 0
        round_keys = key_schedule(attack_keys[0])
        real_delta = galois_mult(aes_sbox[constant_byte ^ attack_keys[0][5]], 3) \
                    ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][10]], 1) \
                    ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][15]], 1) ^ round_keys[1][0]

        gamma_1 = galois_mult(aes_sbox[galois_mult(aes_sbox[constant_byte ^ attack_keys[0][4]], 1) \
                                          ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][9]], 2) \
                                          ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][14]], 3) \
                                          ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][3]],
                                                        1) ^ round_keys[1][5]], 3)

        gamma_2 = galois_mult(aes_sbox[galois_mult(aes_sbox[constant_byte ^ attack_keys[0][8]], 1) \
                                          ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][13]], 1) \
                                          ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][2]], 2) \
                                          ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][7]],
                                                        3) ^ round_keys[1][10]], 1)

        gamma_3 = galois_mult(aes_sbox[galois_mult(aes_sbox[constant_byte ^ attack_keys[0][12]], 3) \
                                          ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][1]], 1) \
                                          ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][6]], 1) \
                                          ^ galois_mult(aes_sbox[constant_byte ^ attack_keys[0][11]],
                                                        2) ^ round_keys[1][15]], 1)
        real_gamma = gamma_1 ^ gamma_2 ^ gamma_3 ^ round_keys[2][0]
        (rank_progress, trace_count, key_probabilities) = run_attack_round_3(model, attack_traces, attack_plaintexts,
                                                                             real_key, real_gamma, real_delta,
                                                                             batch_size)

        plot_graph(rank_progress, trace_count, key_probabilities, plot_filename)  # plotting the rank progress across the attack traces
        write_to_npz(results_filename, rank_progress, trace_count, key_probabilities)

    elif hypothesis == 4:
        real_delta = calc_delta(attack_plaintexts, attack_keys)
        real_gamma = calc_gamma(attack_plaintexts, attack_keys)
        real_theta = calc_theta(attack_plaintexts, attack_keys)
        real_delta = real_delta[0]
        real_gamma = real_gamma[0]
        real_theta = real_theta[0]
        (rank_progress, trace_count, key_probabilities) = run_attack_round_4(model, attack_traces, attack_plaintexts,
                                                                             real_key, real_gamma, real_delta, real_theta,
                                                                             batch_size)

        plot_graph(rank_progress, trace_count, key_probabilities, plot_filename)  # plotting the rank progress across the attack traces
        write_to_npz(results_filename, rank_progress, trace_count, key_probabilities)
    print()
