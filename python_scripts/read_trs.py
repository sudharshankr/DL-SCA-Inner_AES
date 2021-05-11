import matplotlib.pyplot as plt
import numpy as np
import sys
import configparser

from label_traces import LabelledTraces
import Trace as trs

from funcs import hamming_lookup, aes_sbox



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


def write_to_npz(filename, traces, data):
    print("Saving file")
    output_file = filename
    np.savez(output_file, traces=traces, data=data)


def write_metadata_to_file(filename, plaintexts, ciphertexts):
    print("Saving file")
    output_file = filename
    np.savez(output_file, plaintext=plaintexts, ciphertext=ciphertexts)


def allocate_random_keys(keys):
    """
    Generating random keys for sanity checks during training
    @param keys: The portion of the keys you want as random
    """
    for i in range(keys.shape[0]):
        keys[i] = np.random.randint(256, size=(16))


if __name__ == '__main__':
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())    # Initializing configuration
    config.read('config.ini')
    leakage_details = config['Leakage']
    training_details = config['Traces']
    trs_file_details = config['TRS']
    in_file = trs_file_details['InputFilename']
    fixed_key = trs_file_details['key']
    # in_file = "../data/traces/raw_traces/SequenceAcquisition_SW_AES_ENC_3kx16"
    # out_file = "../data/traces/" + sys.argv[1]  # add .h5 extension in the argument
    # key = [b for b in b'\xca\xfe\xba\xbe\xde\xad\xbe\xef\x00\x01\x02\x03\x04\x05\x06\x07']

    key = [int(fixed_key[b:b+2], 16) for b in range(0, len(fixed_key), 2)]
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

    # allocate_random_keys(raw_key[:2500])
    print("Preparing the traces for training")
    traces = LabelledTraces(byte_attacked=leakage_details.getint('TargetKeyByteIndex'),
                            leakage_round=leakage_details.getint('LeakageRound'),
                            hypothesis_round=leakage_details.getint('HypothesisRound'),
                            filename=None, raw_traces=raw_traces,
                            raw_plaintext=raw_plaintexts, raw_key=raw_key)

    traces.prepare_traces_labels(training_details.getint('ProfilingStart'), training_details.getint('ProfilingEnd'),
                                 training_details.getint('ValidationStart'), training_details.getint('ValidationEnd'),
                                 training_details.getint('AttackStart'), training_details.getint('AttackEnd'),
                                 training_details.getint('PoIStart'), training_details.getint('PoIEnd'))

    traces.write_to_file(trs_file_details['TracesStorageFile'])
    # write_metadata_to_file("../data/traces/metadata_output.npz", raw_plaintexts, raw_ciphertexts)
    print()
    # plot_trace(raw_traces[0])

    # traces1 = LabelledTraces(2,1, "../data/traces/raw_traces/ATMega8515_raw_traces.h5")
    # plot_trace(traces1.raw_traces[0])

    # 3rd round traces
    # 57750 - 78670 - total of 19120 features
    # 58000-60960

    # correlation between plaintexts and ciphertexts
    # tt = raw_traces[:30, 58000:60960]
    # # r = np.corrcoef(hyp, raw_traces[:, 58000:60960])
    # print()
    # num_point = tt.shape[1]
    # num_traces = tt.shape[0]
    # cpa_output = [0] * 256
    # max_cpa = [0] * 256
    #
    # for guess_idx in range(256):
    #     # print("Sub-key = %2d, hyp = %02x" % (b_num, k_guess))
    #
    #     # Initialize arrays and variables to zero
    #     sum_num = np.zeros(num_point)
    #     sum_den_1 = np.zeros(num_point)
    #     sum_den_2 = np.zeros(num_point)
    #     hyp = hamming_lookup[aes_sbox[raw_plaintexts[:num_traces, 0] ^ guess_idx]]
    #     # hyp = np.zeros(num_traces)
    #     # for t_num in range(num_traces):
    #     #     hyp[t_num] = HW[intermediate(plaintext[t_num][byte_idx], guess_idx)]
    #
    #     h_mean = np.mean(hyp, dtype=np.float64)  # Mean of hypothesis
    #     t_mean = np.mean(tt, axis=0, dtype=np.float64)  # Mean of all points in trace
    #
    #     # For each trace, do the following
    #     for t_num in range(num_traces):
    #         h_diff = (hyp[t_num] - h_mean)
    #         t_diff = tt[t_num, :] - t_mean
    #
    #         sum_num += h_diff * t_diff
    #         sum_den_1 += h_diff ** 2
    #         sum_den_2 += t_diff ** 2
    #
    #     cpa_output[guess_idx] = sum_num / np.sqrt(sum_den_1 * sum_den_2)
    #     max_cpa[guess_idx] = max(abs(cpa_output[guess_idx]))
    #
    # # print()
    # # plt.plot(max_cpa)
    # # plt.show()
    # plot_trace(max_cpa)
