import concurrent.futures
import itertools
import time
import random
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import sys
import configparser

from aeskeyschedule import key_schedule
from tqdm import tqdm

import Trace as trs
from funcs import hamming_lookup, aes_sbox, calc_round_key_byte, galois_mult
from leakage_models import *
# import cupy as cp
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

h_diff = None
t_diff = None
h_diff_ss = None
t_diff_ss = None

font = {'family': 'normal',
        'size': 11}

# def write_to_cpz(filename, ranks, trace_cnt, key_probs):
#     print("Saving file")
#     output_file = filename
#     cp.savez(output_file, ranks=ranks, trace_cnt=trace_cnt, key_probs=key_probs)


def write_to_npz(filename, ranks, trace_cnt, key_probs=None):
    print("Saving file")
    output_file = filename
    np.savez(output_file, ranks=ranks, trace_cnt=trace_cnt)


def determineTrsSampleCoding(ts):
    if ts._sampleCoding == ts.CodingByte:
        samplesDataType = "int8"
    elif ts._sampleCoding == ts.CodingShort:
        samplesDataType = "int16"
    elif ts._sampleCoding == ts.CodingInt:
        samplesDataType = "int32"
    elif ts._sampleCoding == ts.CodingFloat:
        samplesDataType = "float32"
    else:
        samplesDataType = None
    return samplesDataType


def plot_graph(ranks, traces_counts, key_probs=None):
    plt.rc('font', **font)
    plt.figure(figsize=(10, 6))
    plt.title('Rank vs Traces Number')
    plt.xlabel('Number of traces')
    plt.ylabel('Rank')
    plt.grid(True)
    plt.plot(traces_counts, ranks)
    plt.show()

    # guess = int('{0:024b}'.format(key_probs.argsort()[-1])[:8], 2)
    # guess = key_probs.argsort()[-1]
    # print("This is the key Guess: ", hex(guess), "and its rank: ", ranks[-1])
    # if ranks[-1] == 0:
    #     print("Attack succeeded")


def return_idx_16(key, d):
    return int('{0:08b}'.format(key) + '{0:08b}'.format(d), 2)


def return_correlations(guess_idx):
    # print("Guess: %d" % guess_idx, end='\r', flush=True)
    # pbar.update(1)
    num = (h_diff[:, guess_idx].reshape((h_diff.shape[0], 1)) * t_diff).sum(axis=0)
    den = h_diff_ss[guess_idx] * t_diff_ss
    cpa_output = num / np.sqrt(den)
    max_cpa[guess_idx] = max(abs(cpa_output))
    # return max(abs(cpa_output))


# Even faster correlation trace computation
# Takes the full matrix of predictions instead of just a column
# O - (n,t) array of n traces with t samples each
# P - (n,m) array of n predictions for each of the m candidates
# returns an (m,t) correaltion matrix of m traces t samples each
def correlationTraces(O, P):
    (n, t) = O.shape  # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (np.einsum("nt->t", O, dtype='float64', optimize='optimal') / np.double(n))  # compute O - mean(O)
    DP = P - (np.einsum("nm->m", P, dtype='float64', optimize='optimal') / np.double(n))  # compute P - mean(P)

    numerator = np.einsum("nm,nt->mt", DP, DO, optimize='optimal')
    tmp1 = np.einsum("nm,nm->m", DP, DP, optimize='optimal')
    tmp2 = np.einsum("nt,nt->t", DO, DO, optimize='optimal')
    tmp = np.einsum("m,t->mt", tmp1, tmp2, optimize='optimal')
    denominator = np.sqrt(tmp)

    return numerator / denominator


def calc_avg_key_rank(raw_traces, num_traces, hyp, no_of_experiments):
    temp_key_ranks = []
    for n in tqdm(range(no_of_experiments), position=0, leave=True):
        sample_inst = random.sample(range(0, len(raw_traces)), num_traces)
        test_traces = raw_traces[sample_inst, :]
        test_hyp = hyp[sample_inst]
        # for trace_id in range(it_start, num_traces):
        #     hyp[trace_id, :] = calc_hypothesis_round_2(raw_plaintexts[trace_id, 0], guesses_range)
        max_cpa = np.zeros((1, 1))
        max_cpa = np.concatenate(
            (max_cpa, np.amax(np.abs(correlationTraces(test_traces, hyp[sample_inst])), axis=1).reshape(guesses_range.shape[0], 1)), axis=0)
        max_cpa = max_cpa[1:, 0]
        cpa_refs = np.argsort(max_cpa)[::-1]
        key_rank = np.where(cpa_refs == real_idx)[0]
        temp_key_ranks.append(key_rank[0])

    avg_rank = np.mean(temp_key_ranks)
    return int(avg_rank), max_cpa



if __name__ == '__main__':
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read('config.ini')
    leakage_config = config['Leakage']
    byte_attacked = leakage_config.getint('TargetKeyByteIndex')
    leakage = leakage_config.getint('LeakageRound')
    hypothesis = leakage_config.getint('HypothesisRound')
    hyp_type = leakage_config['HypothesisType']
    results_filename = "../../data/attack_results/cpa_attacks-gaussian-dataset-noise-10-results-leakage_rnd_" + str(leakage) \
                       + "-hypothesis_rnd_" + str(hypothesis) + "-" + hyp_type + "-" + str(byte_attacked) + ".npz"

    # npzfile = np.load("/Users/sud/the_stuff/Studies/Thesis/Data/filtered_traces/filtered-rnd2-traces_37000-38000-new.npz")
    npzfile = np.load("filtered-gaussian-rnd2-traces_37000-38000-new-2.npz")
    raw_traces = npzfile["raw_traces"]
    raw_plaintexts = npzfile["raw_plaintexts"]
    raw_key = npzfile["raw_key"]
    known_key = raw_key[0][0]
    round_keys = key_schedule(raw_key[0])
    print("the real key ", known_key)
    count_traces = []
    key_ranks = []

    constant_byte = 0
    real_delta = galois_mult(aes_sbox[constant_byte ^ raw_key[0][5]], 3) \
                 ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][10]], 1) \
                 ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][15]], 1) ^ round_keys[1][0]

    real_idx = return_idx_16(known_key, real_delta)

    print("Initiating Guesses Range")
    key_guesses = np.array([n for n in range(256 * 256)])
    guesses_range = np.load("../Guesses_range_16.npy")
    no_of_experiments = 100
    # it_start = 0
    max_cpa = None
    hyp = np.zeros((2000, guesses_range.shape[0]))
    for trace_count in range(len(raw_traces)):
        hyp[trace_count, :] = calc_hypothesis_round_2(raw_plaintexts[trace_count, 0], guesses_range)

    start_time = time.time()

    for num_traces in range(10, 20, 10):
        print("Calculating for %d traces" % num_traces)
        avg_rank, max_cpa = calc_avg_key_rank(raw_traces, num_traces, hyp, no_of_experiments)
        print("Key rank: %d" % avg_rank)
        key_ranks.append(avg_rank)
        count_traces.append(num_traces)

    for num_traces in range(100, 2010, 100):
        print("Calculating for %d traces" % num_traces)
        # temp_key_ranks = []
            # print("Experiment: %d\r" % n, end='', flush=True)
            # sample_inst = random.sample(range(0, len(raw_traces)), num_traces)
            # test_traces = raw_traces[sample_inst, :]
            # test_hyp = hyp[sample_inst]
            # # for trace_id in range(it_start, num_traces):
            # #     hyp[trace_id, :] = calc_hypothesis_round_2(raw_plaintexts[trace_id, 0], guesses_range)
            # max_cpa = np.zeros((1, 1))
            # max_cpa = np.concatenate(
            #         (max_cpa, np.amax(np.abs(correlationTraces(test_traces, hyp[sample_inst])), axis=1).reshape(guesses_range.shape[0], 1)), axis=0)
            # max_cpa = max_cpa[1:, 0]
            # cpa_refs = np.argsort(max_cpa)[::-1]
            # key_rank = np.where(cpa_refs == real_idx)[0]
            # # key_ranks.append(np.where(cpa_refs == real_idx)[0])
            # temp_key_ranks.append(key_rank[0])
        avg_rank, max_cpa = calc_avg_key_rank(raw_traces, num_traces, hyp, no_of_experiments)
        print("Key rank: %d" % avg_rank)
        key_ranks.append(avg_rank)
        count_traces.append(num_traces)
        # it_start = num_traces

    time_taken = time.time() - start_time
    print("Total time taken:%.2f seconds, %.2f minutes" % (time_taken, time_taken/60))
    plot_graph(key_ranks, count_traces)
    write_to_npz(results_filename, key_ranks, count_traces)
    print("The guess is: ", int('{0:016b}'.format(np.argsort(max_cpa)[::-1][0])[:8], 2))
