import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import sys
import configparser

from aeskeyschedule import key_schedule
from tqdm import tqdm

import Trace as trs
from funcs import hamming_lookup, aes_sbox, calc_round_key_byte, galois_mult
from leakage_models import calc_hypothesis_round_3
from multiprocessing import Pool
import random

h_diff = None
t_diff = None
h_diff_ss = None
t_diff_ss = None

font = {'family': 'normal',
        'size': 11}

test_traces = None
test_hyp = None

# def write_to_cpz(filename, ranks, trace_cnt, key_probs):
#     print("Saving file")
#     output_file = filename
#     cp.savez(output_file, ranks=ranks, trace_cnt=trace_cnt, key_probs=key_probs)


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


def return_idx(key, d, g):
    return int('{0:08b}'.format(key) + '{0:08b}'.format(d) + '{0:08b}'.format(g), 2)


def write_to_npz(filename, ranks, trace_cnt, key_probs=None):
    print("Saving file")
    output_file = filename
    np.savez(output_file, ranks=ranks, trace_cnt=trace_cnt)


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


def calc_corr(batch_id, batch_step):
    # print("batch: %d" % batch_id)
    # hyp_temp = test_hyp[:, batch_id:batch_id + batch_step]
    temp_hyp = hyp[:, batch_id:batch_id + batch_step]
    corr_matrix = np.amax(np.abs(correlationTraces(test_traces, temp_hyp)), axis=1).reshape(batch_step, 1)
    return corr_matrix


if __name__ == '__main__':
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read('config.ini')
    leakage_config = config['Leakage']
    byte_attacked = leakage_config.getint('TargetKeyByteIndex')
    leakage = leakage_config.getint('LeakageRound')
    hypothesis = leakage_config.getint('HypothesisRound')
    hyp_type = leakage_config['HypothesisType']
    results_filename = "../../data/attack_results/cpa_attacks-misalign-gaussian-dataset-500-600-results-leakage_rnd_" + str(leakage) \
                       + "-hypothesis_rnd_" + str(hypothesis) + "-" + hyp_type + "-" + str(byte_attacked) + ".npz"


    # npzfile = np.load("/Users/sud/the_stuff/Studies/Thesis/Data/filtered_traces/filtered_aligned_traces_58400-59100.npz")
    npzfile = np.load("filtered_misaligned_gaussian_traces-rnd3-58400-59100.npz")
    total_traces = 600
    raw_traces = npzfile["raw_traces"][:total_traces]
    raw_plaintexts = npzfile["raw_plaintexts"][:total_traces]
    raw_key = npzfile["raw_key"][:total_traces]
    # 16000 - 19930
    # tt = raw_traces  # [:2000, 58000:60960]
    known_key = raw_key[0][0]
    round_keys = key_schedule(raw_key[0])
    print("the real key ", known_key)
    num_point = raw_traces.shape[1]
    count_traces = []
    key_ranks = []
    constant_byte = 0
    real_delta = galois_mult(aes_sbox[constant_byte ^ raw_key[0][5]], 3) \
                 ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][10]], 1) \
                 ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][15]], 1) ^ round_keys[1][0]

    gamma_1 = galois_mult(aes_sbox[galois_mult(aes_sbox[constant_byte ^ raw_key[0][4]], 1) \
                                   ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][9]], 2) \
                                   ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][14]], 3) \
                                   ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][3]],
                                                 1) ^ round_keys[1][5]], 3)

    gamma_2 = galois_mult(aes_sbox[galois_mult(aes_sbox[constant_byte ^ raw_key[0][8]], 1) \
                                   ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][13]], 1) \
                                   ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][2]], 2) \
                                   ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][7]],
                                                 3) ^ round_keys[1][10]], 1)

    gamma_3 = galois_mult(aes_sbox[galois_mult(aes_sbox[constant_byte ^ raw_key[0][12]], 3) \
                                   ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][1]], 1) \
                                   ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][6]], 1) \
                                   ^ galois_mult(aes_sbox[constant_byte ^ raw_key[0][11]],
                                                 2) ^ round_keys[1][15]], 1)
    real_gamma = gamma_1 ^ gamma_2 ^ gamma_3 ^ round_keys[2][0]
    real_idx = return_idx(known_key, real_delta, real_gamma)
    print("Initiating Guesses Range")
    key_guesses = np.array([n for n in range(256 * 256 * 256)])
    # guesses_range = np.zeros((key_guesses.shape[0], 3))
    # for i in range(key_guesses.shape[0]):
    #     f = '{0:024b}'.format(key_guesses[i])
    #     guesses_range[i][0] = int(f[:8], 2)
    #     guesses_range[i][1] = int(f[8:16], 2)
    #     guesses_range[i][2] = int(f[16:24], 2)
    # guesses_range = guesses_range.astype("uint8")
    # np.save("Guesses_range.npy", guesses_range)
    guesses_range = np.load("../Guesses_range_24.npy")
    # cpa_output = [0] * guesses_range.shape[0]
    # max_cpa = [0] * guesses_range.shape[0]
    it_start = 0
    no_of_experiments = 1
    hyp = np.zeros((total_traces, guesses_range.shape[0]))
    hyp_start_time = time.time()
    for trace_count in tqdm(range(len(raw_traces)), position=0, leave=True):
        hyp[trace_count, :] = calc_hypothesis_round_3(raw_plaintexts[trace_count, 0], guesses_range)

    print("Time taken to calculate 600 hyp: %d seconds" % (time.time()-hyp_start_time))
    # np.savez("hyp-rnd2-traces-500.npz", hypothesis=hyp)
    start_time = time.time()
    for num_traces in tqdm(range(550, 650, 50), position=0, leave=True):
        print("Calculating for %d traces" % num_traces)
        temp_key_ranks = []
        # hyp = np.zeros((num_traces, guesses_range.shape[0]))
        for n in tqdm(range(no_of_experiments), position=0, leave=True):
            sample_inst = random.sample(range(0, len(raw_traces)), num_traces)
            test_traces = raw_traces[sample_inst, :]
            # for i, trace_id in enumerate(sample_inst):
            #     hyp[i, :] = calc_hypothesis_round_3(raw_plaintexts[trace_id, 0], guesses_range)
            # test_hyp = hyp[sample_inst]
            # for trace_id in range(it_start, num_traces):
            #     hyp[trace_id, :] = calc_hypothesis_round_3(raw_plaintexts[trace_id, 0], guesses_range)
            # print("Done calculating the hypothoses...")
            # print("Starting with the Guesses...")
            max_cpa = np.zeros((1, 1))
            batch_step = int(len(key_guesses) / 256)
            # batches = [(batch_id, batch_step) for batch_id in range(0, len(key_guesses), batch_step)]
            # pool = Pool(processes=2)
            # temp_key_ranks = pool.starmap_async(calc_corr, batches).get()
            for batch_id in tqdm(range(0, len(key_guesses), batch_step), position=0, leave=True):
                # hyp_temp = hyp[sample_inst, batch_id:batch_id + batch_step]
                max_cpa = np.concatenate(
                    (max_cpa, np.amax(np.abs(correlationTraces(test_traces, hyp[sample_inst, batch_id:batch_id + batch_step])), axis=1).reshape(batch_step, 1)), axis=0)
                # max_cpa = np.concatenate((max_cpa, max_cpa_temp), axis=0)
            # map_res = pool.starmap_async(calc_corr, batches).get()
            # for result in map_res:
            #     max_cpa = np.concatenate((max_cpa, result), axis=0)
            # max_cpa = np.concatenate((max_cpa, temp_array), axis=0)
            max_cpa = max_cpa[1:, 0]
            cpa_refs = np.argsort(max_cpa)[::-1]
            key_rank = np.where(cpa_refs == real_idx)[0]
            temp_key_ranks.append(key_rank[0])

        avg_rank = np.mean(temp_key_ranks)
        print("Key Rank: ", avg_rank)
        key_ranks.append(int(avg_rank))
        count_traces.append(num_traces)
        # it_start = num_traces

    time_taken = time.time() - start_time
    print("Total time taken:%.2f seconds, %.2f minutes" % (time_taken, time_taken/60))
    # print(len(max_cpa))
    #[array([2918433]), array([722637]), array([93251]), array([157522]), array([21131]), array([3975]), array([1045]),
     #array([591]), array([220]), array([36]), array([14]), array([22]), array([5]), array([1]), array([0]), array([0])]
    plot_graph(key_ranks, count_traces)
    write_to_npz(results_filename, key_ranks, count_traces)
    print("The guess is: ", int('{0:024b}'.format(np.argsort(max_cpa)[::-1][0])[:8], 2))
    # int('{0:024b}'.format(key_probs.argsort()[-1])[:8], 2)
