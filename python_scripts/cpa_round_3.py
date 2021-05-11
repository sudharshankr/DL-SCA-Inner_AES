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
# import cupy as cp
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

h_diff = None
t_diff = None
h_diff_ss = None
t_diff_ss = None


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


def plot_trace(trace):
    plt.plot(trace)
    plt.show()


def plot_graph(ranks, traces_counts, key_probs=None):
    plt.figure(figsize=(15, 6))
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


def return_correlations(guess_idx):
    # print("Guess: %d" % guess_idx, end='\r', flush=True)
    # pbar.update(1)
    num = (h_diff[:, guess_idx].reshape((h_diff.shape[0], 1)) * t_diff).sum(axis=0)
    den = h_diff_ss[guess_idx] * t_diff_ss
    cpa_output = num / np.sqrt(den)
    max_cpa[guess_idx] = max(abs(cpa_output))
    # return max(abs(cpa_output))


if __name__ == '__main__':
    start_time = 0
    # config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())  # Initializing configuration
    # config.read('config.ini')
    # trs_file_details = config['TRS']
    # in_file = trs_file_details['InputFilename']
    # fixed_key = trs_file_details['key']
    # key = [int(fixed_key[b:b + 2], 16) for b in range(0, len(fixed_key), 2)]
    #
    # ts = trs.TraceSet()
    # ts.open(in_file + ".trs")
    # samplesDataType = determineTrsSampleCoding(ts)
    # print("Preallocating arrays")
    # data_space = int(ts._dataSpace / 2)
    # raw_traces = np.empty(shape=(ts._numberOfTraces, ts._numberOfSamplesPerTrace), dtype=samplesDataType)
    # raw_plaintexts = np.empty(shape=(ts._numberOfTraces, data_space), dtype="uint8")
    # raw_ciphertexts = np.empty(shape=(ts._numberOfTraces, data_space), dtype="uint8")
    # raw_key = np.empty(shape=(ts._numberOfTraces, data_space), dtype="uint8")
    # print("Populating arrays")
    # for i in range(ts._numberOfTraces):
    #     t = ts.getTrace(i)
    #     raw_traces[i, :] = np.array(t._samples, dtype=samplesDataType)
    #     raw_plaintexts[i, :] = np.array(t._data[:data_space], dtype="uint8")
    #     raw_ciphertexts[i, :] = np.array(t._data[data_space:], dtype="uint8")
    #     raw_key[i, :] = np.array(key[:data_space], dtype="uint8")

    # np.savez("trs_file_traces.npz", raw_traces=raw_traces[:2000, 58000:60960], raw_plaintexts=raw_plaintexts[:2000], raw_key=raw_key[:2000])
    npzfile = np.load("trs_file_traces.npz")
    raw_traces = npzfile["raw_traces"]
    raw_plaintexts = npzfile["raw_plaintexts"]
    raw_key = npzfile["raw_key"]
    # plot_trace(raw_traces[0])
    # 16000 - 19930
    tt = raw_traces # [:2000, 58000:60960]
    # r = np.corrcoef(hyp, raw_traces[:, 58000:60960])
    known_key = raw_key[0][0]
    round_keys = key_schedule(raw_key[0])
    print("the real key ", known_key)
    num_point = tt.shape[1]
    # num_traces = tt.shape[0]
    count_traces = []
    key_ranks = []
    total_no_of_traces = 50
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
    guesses_range = np.zeros((key_guesses.shape[0], 3))
    for i in range(key_guesses.shape[0]):
        f = '{0:024b}'.format(key_guesses[i])
        guesses_range[i][0] = int(f[:8], 2)
        guesses_range[i][1] = int(f[8:16], 2)
        guesses_range[i][2] = int(f[16:24], 2)
    guesses_range = guesses_range.astype("uint8")
    # cpa_output = [0] * guesses_range.shape[0]
    max_cpa = [0] * guesses_range.shape[0]
    # jobs = []
    for num_traces in range(25, 30, 10):
        print("Calculating for %d traces" % num_traces)
        hyp = np.zeros((num_traces, guesses_range.shape[0]))
        for trace_id in range(num_traces):
            hyp[trace_id, :] = calc_hypothesis_round_3(raw_plaintexts[trace_id, 0], guesses_range)
        print("Done calculating the hypothoses...")
        # sum_num = np.zeros(num_point)
        # sum_den_1 = np.zeros(num_point)
        # sum_den_2 = np.zeros(num_point)
        # hypothesis based on round 1. (might have to change this)
        # hyp = hamming_lookup[aes_sbox[raw_plaintexts[:num_traces, 0] ^ guess_idx]]
        # hyp = np.zeros(num_traces)
        # for t_num in range(num_traces):
        #     hyp[t_num] = HW[intermediate(plaintext[t_num][byte_idx], guess_idx)]

        h_mean = np.mean(hyp, axis=0, dtype=np.float64)  # Mean of hypothesis
        t_mean = np.mean(tt[:num_traces, :], axis=0, dtype=np.float64)  # Mean of all points in trace
        h_diff = hyp - h_mean
        t_diff = tt[:num_traces, :] - t_mean
        h_diff_ss = (h_diff * h_diff).sum(axis=0)
        t_diff_ss = (t_diff * t_diff).sum(axis=0)
        # For each trace, do the following
        print("Starting with the Guesses...")
        # pbar = tqdm(total=len(key_guesses))
        # pool = ThreadPool(200)
        # max_cpa = pool.map(return_correlations, key_guesses)
        # pool.close()
        # pool.join()
        # pbar.close()
        nprocesses = 1000
        start_time = time.time()
        for guess_idx in range(0, guesses_range.shape[0], nprocesses):
            print("Guess no.: %d and %2f" % (guess_idx, guess_idx / len(key_guesses)), end='\r', flush=True)
            pool = multiprocessing.Pool(nprocesses)
            pool.map(return_correlations, [kg for kg in range(guess_idx, guess_idx+nprocesses)])
            pool.close()
            #
            # p = multiprocessing.Process(target=return_correlations, args=(guess_idx,))
            # jobs.append(p)
            # p.start()
        #     # result = np.matmul(h_diff.transpose(), t_diff) / np.sqrt(np.outer(h_diff_ss, t_diff_ss))
        #     # bound the values to -1 to 1 in the event of precision issues
        #     # cpa_output = np.maximum(np.minimum(result, 1.0), -1.0)
        #     # for t_num in range(num_traces):
        #     #     h_diff = (hyp[t_num, guess_idx] - h_mean[guess_idx])
        #     #
        #     #     sum_num += h_diff * t_diff
        #     #     sum_den_1 += h_diff ** 2
        #     #     sum_den_2 += t_diff ** 2
        #     # cpa_output = sum_num / np.sqrt(sum_den_1 * sum_den_2)
        # for proc in jobs:
        #     proc.join()

        # pbar.close()
        # print("Sub-key = %2d, hyp = %02x" % (b_num, k_guess))

        # Initialize arrays and variables to zero

        # print()
        # plt.plot(max_cpa)
        # plt.show()
        cpa_refs = np.argsort(max_cpa)[::-1]
        # print(cpa_refs)
        # Find Guess Entropy (GE)
        # kr = list(cpa_refs).index(known_key)
        key_ranks.append(np.where(cpa_refs == real_idx)[0])
        count_traces.append(num_traces)

    # print("The key rank: ",kr)
    print(time.time() - start_time)
    print(len(max_cpa))
    print("The guess is: ", np.argsort(max_cpa)[-1])
    # write_to_cpz("cupy_output", ranks=key_ranks, trace_cnt=count_traces, key_probs=cpa_refs)
    # plot_graph(key_ranks, count_traces)
