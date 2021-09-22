import matplotlib.pyplot as plt
import numpy as np
import sys
import configparser
import Trace as trs
from funcs import hamming_lookup, aes_sbox
from label_traces import LabelledTraces


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
    """
    Plot graph for key ranks vs. trace counts
    @param ranks: List containing ranks
    @param traces_counts: List containing trace counts corresponding to the ranks
    @param key_probs: Probablity of that key guess
    @return: matplotlib plot (can be saved as an image too)
    """
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


if __name__ == '__main__':
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())  # Initializing configuration
    config.read('config.ini')
    trs_file_details = config['TRS']
    in_file = trs_file_details['InputFilename']
    fixed_key = trs_file_details['key']
    key = [int(fixed_key[b:b + 2], 16) for b in range(0, len(fixed_key), 2)]

    ts = trs.TraceSet()
    ts.open(in_file + ".trs")
    samplesDataType = determineTrsSampleCoding(ts)
    print("Preallocating arrays")
    data_space = int(ts._dataSpace / 2)
    raw_traces = np.empty(shape=(ts._numberOfTraces, ts._numberOfSamplesPerTrace), dtype=samplesDataType)
    raw_plaintexts = np.empty(shape=(ts._numberOfTraces, data_space), dtype="uint8")
    raw_ciphertexts = np.empty(shape=(ts._numberOfTraces, data_space), dtype="uint8")
    raw_key = np.empty(shape=(ts._numberOfTraces, data_space), dtype="uint8")
    print("Populating arrays")
    for i in range(ts._numberOfTraces):
        t = ts.getTrace(i)
        raw_traces[i, :] = np.array(t._samples, dtype=samplesDataType)
        raw_plaintexts[i, :] = np.array(t._data[:data_space], dtype="uint8")
        raw_ciphertexts[i, :] = np.array(t._data[data_space:], dtype="uint8")
        raw_key[i, :] = np.array(key[:data_space], dtype="uint8")
    # traces = LabelledTraces(0, 1, 1, "../data/traces/raw_traces/ASCAD.h5")
    # raw_traces = traces.raw_traces
    # raw_plaintexts = traces.raw_plaintext
    # raw_key = traces.raw_key
    # plot_trace(raw_traces[0])
    # 16000 - 19930
    tt = raw_traces[:2000, 16000:19000]
    # r = np.corrcoef(hyp, raw_traces[:, 58000:60960])
    known_key = raw_key[0][0]
    print("the real key ", known_key)
    num_point = tt.shape[1]
    # num_traces = tt.shape[0]
    cpa_output = [0] * 256
    max_cpa = [0] * 256
    count_traces = []
    key_ranks = []
    total_no_of_traces = 100

    for num_traces in range(10, total_no_of_traces, 10):
        print("Calculating for %d traces" % num_traces)
        for guess_idx in range(256):
            # print("Sub-key = %2d, hyp = %02x" % (b_num, k_guess))

            # Initialize arrays and variables to zero
            sum_num = np.zeros(num_point)
            sum_den_1 = np.zeros(num_point)
            sum_den_2 = np.zeros(num_point)
            # hypothesis based on round 1. (might have to change this)
            hyp = hamming_lookup[aes_sbox[raw_plaintexts[:num_traces, 0] ^ guess_idx]]
            # hyp = np.zeros(num_traces)
            # for t_num in range(num_traces):
            #     hyp[t_num] = HW[intermediate(plaintext[t_num][byte_idx], guess_idx)]

            h_mean = np.mean(hyp, dtype=np.float64)  # Mean of hypothesis
            t_mean = np.mean(tt, axis=0, dtype=np.float64)  # Mean of all points in trace

            # For each trace, do the following
            for t_num in range(num_traces):
                h_diff = (hyp[t_num] - h_mean)
                t_diff = tt[t_num, :] - t_mean

                sum_num += h_diff * t_diff
                sum_den_1 += h_diff ** 2
                sum_den_2 += t_diff ** 2

            cpa_output[guess_idx] = sum_num / np.sqrt(sum_den_1 * sum_den_2)
            max_cpa[guess_idx] = max(abs(cpa_output[guess_idx]))

        cpa_refs = np.argsort(max_cpa)[::-1]
        key_ranks.append(np.where(cpa_refs == known_key)[0])
        count_traces.append(num_traces)

    plot_graph(key_ranks, count_traces)



