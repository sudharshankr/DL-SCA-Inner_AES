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


def rearrange_traces(raw_traces, raw_plaintexts, raw_ciphertexts, raw_key):
    """
    Rearragning required while attacking round 4
    @param raw_traces:  raw traceset
    @param raw_plaintexts: raw plaintexts
    @param raw_ciphertexts: raw ciphertexts
    @param raw_key: raw key set
    @return: return the rearranged set of traces, plaintexts, ciphertexts and keys
    """
    temp_traces = np.concatenate((raw_traces[1::2], raw_traces[::2]), axis=0)
    temp_plaintexts = np.concatenate((raw_plaintexts[1::2], raw_plaintexts[::2]), axis=0)
    temp_ciphertexts = np.concatenate((raw_ciphertexts[1::2], raw_ciphertexts[::2]), axis=0)
    temp_key = np.concatenate((raw_key[1::2], raw_key[::2]), axis=0)
    return temp_traces, temp_plaintexts, temp_ciphertexts, temp_key


def add_gaussian_noise(traces, samplesDataType):
    """
    Adding Gaussian noise to the traces
    @param traces: Traces for which the noise needs to be computed
    @type traces: np.ndarray
    @param samplesDataType: Data type of the traces
    @type samplesDataType: string
    @return: Traces with Gaussian noise
    @rtype: np.ndarray
    """
    print("Adding Guassian Noise...")
    mu = np.mean(traces)
    sigma = np.std(traces, ddof=1)
    traces = traces + np.random.normal(mu, sigma, (traces.shape[0], 1))
    traces = traces.astype(samplesDataType)
    return traces


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

    # j = 0
    # for i in range(ts._numberOfTraces):
    #     t = ts.getTrace(i)
    #     raw_traces[i, :] = np.array(t._samples, dtype=samplesDataType)
    #     raw_plaintexts[i, :] = np.array(t._data[:data_space], dtype="uint8")
    #     raw_ciphertexts[i, :] = np.array(t._data[data_space:data_space*2], dtype="uint8")
    #     raw_key[i, :] = np.array(t._data[data_space*2:], dtype="uint8")
    # j = 0
    # k = 0
    # attacking_traces = np.empty(shape=(2000, ts._numberOfSamplesPerTrace), dtype=samplesDataType)
    # attacking_plaintexts = np.empty(shape=(2000, data_space), dtype="uint8")
    # attacking_ciphertexts = np.empty(shape=(2000, data_space), dtype="uint8")
    # attacking_key = np.empty(shape=(2000, data_space), dtype="uint8")
    # for i in range(ts._numberOfTraces):
    #     t = ts.getTrace(i)
    #     if key != list(t._data[data_space*2:]):
    #         raw_traces[j, :] = np.array(t._samples, dtype=samplesDataType)
    #         raw_plaintexts[j, :] = np.array(t._data[:data_space], dtype="uint8")
    #         raw_ciphertexts[j, :] = np.array(t._data[data_space:data_space*2], dtype="uint8")
    #         raw_key[j, :] = np.array(t._data[data_space*2:], dtype="uint8")
    #         j = j+1
    #     else:
    #         attacking_traces[k, :] = np.array(t._samples, dtype=samplesDataType)
    #         attacking_plaintexts[k, :] = np.array(t._data[:data_space], dtype="uint8")
    #         attacking_ciphertexts[k, :] = np.array(t._data[data_space:data_space * 2], dtype="uint8")
    #         attacking_key[k, :] = np.array(t._data[data_space * 2:], dtype="uint8")
    #         k = k+1
    #
    # raw_traces = np.concatenate((raw_traces[:8000, :], attacking_traces), axis=0)
    # raw_plaintexts = np.concatenate((raw_plaintexts[:8000, :], attacking_plaintexts), axis=0)
    # raw_ciphertexts = np.concatenate((raw_ciphertexts[:8000, :], attacking_ciphertexts), axis=0)
    # raw_key = np.concatenate((raw_key[:8000, :], attacking_key), axis=0)

    # j = 5000
    # allocate_random_keys(raw_key[:2500])
    # round 2 - 37000 - 39500
    # round 3 - 58000 - 60960
    # round 4 - 77500 - 80000
    # count = j
    # for i in range(j, ts._numberOfTraces):
    #     t = ts.getTrace(i)
    #     if key == list(t._data[data_space*2:]):
    #         raw_traces[count, :] = np.array(t._samples, dtype=samplesDataType)
    #         raw_plaintexts[count, :] = np.array(t._data[:data_space], dtype="uint8")
    #         raw_ciphertexts[count, :] = np.array(t._data[data_space:data_space * 2], dtype="uint8")
    #         raw_key[count, :] = np.array(t._data[data_space * 2:], dtype="uint8")
    #         count = count+1
    #
    # count = 3000
    # raw_traces = raw_traces[:count, ]
    # raw_plaintexts = raw_plaintexts[:count, ]
    # raw_ciphertexts = raw_ciphertexts[:count, ]
    # raw_key = raw_key[:count, ]
    #
    # # adding gaussian noise
    # raw_traces = add_gaussian_noise(raw_traces, samplesDataType)

    # round 2 - different batch - 25200 - 26200
    ## rearranging is required only for round 4 traces depending on how the dataset is arranged
    ## In our dataset the rearranging was necessary.
    print("Reshuffling traces...")
    (raw_traces, raw_plaintexts, raw_ciphertexts, raw_key) = rearrange_traces(raw_traces, raw_plaintexts, raw_ciphertexts, raw_key) # rearrange to attain the right indexes for profiling and attack


    print("Preparing the traces for training...")
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
    print("Labelled traces written to file %s. Ready to train!" % trs_file_details['TracesStorageFile'])
    # plot_trace(raw_traces[0])