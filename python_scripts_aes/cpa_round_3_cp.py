import matplotlib.pyplot as plt
import numpy as np
import sys
import configparser

from aeskeyschedule import key_schedule
from tqdm import tqdm

import Trace as trs
from funcs import hamming_lookup, aes_sbox, calc_round_key_byte, galois_mult
from leakage_models import calc_hypothesis_round_3
import cupy as cp


def write_to_cpz(filename, ranks, trace_cnt, key_probs):
    print("Saving file")
    output_file = filename
    cp.savez(output_file, ranks=ranks, trace_cnt=trace_cnt, key_probs=key_probs)


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


if __name__ == '__main__':
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
    # raw_traces = cp.empty(shape=(ts._numberOfTraces, ts._numberOfSamplesPerTrace), dtype=samplesDataType)
    # raw_plaintexts = cp.empty(shape=(ts._numberOfTraces, data_space), dtype="uint8")
    # raw_ciphertexts = cp.empty(shape=(ts._numberOfTraces, data_space), dtype="uint8")
    # raw_key = cp.empty(shape=(ts._numberOfTraces, data_space), dtype="uint8")
    # print("Populating arrays")
    # for i in range(ts._numberOfTraces):
    #     t = ts.getTrace(i)
    #     raw_traces[i, :] = cp.array(t._samples, dtype=samplesDataType)
    #     raw_plaintexts[i, :] = cp.array(t._data[:data_space], dtype="uint8")
    #     raw_ciphertexts[i, :] = cp.array(t._data[data_space:], dtype="uint8")
    #     raw_key[i, :] = cp.array(key[:data_space], dtype="uint8")

    npzfile = cp.load("../data/traces/trs_file_traces.npz")
    raw_traces = npzfile["raw_traces"]
    raw_plaintexts = npzfile["raw_plaintexts"]
    raw_key = npzfile["raw_key"]
    # plot_trace(raw_traces[0])
    # 16000 - 19930
    tt = raw_traces#[:2000, 58000:60960]
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
    key_guesses = cp.array([n for n in range(256 * 256 * 256)])
    guesses_range = cp.zeros((key_guesses.shape[0], 3))
    for i in range(key_guesses.shape[0]):
        f = '{0:024b}'.format(key_guesses[i])
        guesses_range[i][0] = int(f[:8], 2)
        guesses_range[i][1] = int(f[8:16], 2)
        guesses_range[i][2] = int(f[16:24], 2)
    guesses_range = guesses_range.astype("uint8")
    # cpa_output = [0] * guesses_range.shape[0]
    max_cpa = [0] * guesses_range.shape[0]

    for num_traces in range(25, 30, 10):
        print("Calculating for %d traces" % num_traces)
        hyp = cp.zeros((num_traces, guesses_range.shape[0]))
        for trace_id in range(num_traces):
            hyp[trace_id, :] = calc_hypothesis_round_3(raw_plaintexts[trace_id, 0], guesses_range)
        print("Done calculating the hypothoses...")

        h_mean = cp.mean(hyp, axis=0, dtype=cp.float64)  # Mean of hypothesis
        t_mean = cp.mean(tt[:num_traces, :], axis=0, dtype=cp.float64)  # Mean of all points in trace
        h_diff = hyp - h_mean
        t_diff = tt[:num_traces, :] - t_mean
        h_diff_ss = (h_diff * h_diff).sum(axis=0)
        t_diff_ss = (t_diff * t_diff).sum(axis=0)
        # For each trace, do the following
        print("Starting with the Guesses...")
        for guess_idx in range(guesses_range.shape[0]):
            print("Guess no.: %d" % guess_idx, end='\r', flush=True)
            num = (h_diff[:, guess_idx].reshape((h_diff.shape[0], 1)) * t_diff).sum(axis=0)
            den = h_diff_ss[guess_idx] * t_diff_ss
            cpa_output = num / cp.sqrt(den)
            # result = np.matmul(h_diff.transpose(), t_diff) / np.sqrt(np.outer(h_diff_ss, t_diff_ss))
            # bound the values to -1 to 1 in the event of precision issues
            # cpa_output = np.maximum(np.minimum(result, 1.0), -1.0)
            # for t_num in range(num_traces):
            #     h_diff = (hyp[t_num, guess_idx] - h_mean[guess_idx])
            #
            #     sum_num += h_diff * t_diff
            #     sum_den_1 += h_diff ** 2
            #     sum_den_2 += t_diff ** 2
            # cpa_output = sum_num / np.sqrt(sum_den_1 * sum_den_2)
            max_cpa[guess_idx] = max(abs(cpa_output))
        cpa_refs = cp.argsort(max_cpa)[::-1]
        key_ranks.append(cp.where(cpa_refs == real_idx)[0])
        count_traces.append(num_traces)

    # print("The key rank: ",kr)
    print("The guess is: ", cp.argsort(max_cpa)[-1])
    write_to_cpz("cupy_output", ranks=key_ranks, trace_cnt=count_traces, key_probs=cpa_refs)
