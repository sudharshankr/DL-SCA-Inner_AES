import numpy as np
import configparser
import Trace as trs
import matplotlib.pyplot as plt


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


def add_gaussian_noise(traces, samplesDataType):
    print("Adding Gaussian Noise...")
    mu = np.mean(traces)
    sigma = np.std(traces, ddof=1)
    traces = traces + np.random.normal(mu, sigma, (traces.shape[0], 1))
    traces = traces.astype(samplesDataType)
    return traces


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
    # j = 0
    for i in range(ts._numberOfTraces):
        t = ts.getTrace(i)
        raw_traces[i, :] = np.array(t._samples, dtype=samplesDataType)
        raw_plaintexts[i, :] = np.array(t._data[:data_space], dtype="uint8")
        raw_ciphertexts[i, :] = np.array(t._data[data_space:], dtype="uint8")
        raw_key[i, :] = np.array(key[:data_space], dtype="uint8")

    # for i in range(ts._numberOfTraces):
    #     t = ts.getTrace(i)
    #     if key == list(t._data[data_space*2:]):
    #         raw_traces[j, :] = np.array(t._samples, dtype=samplesDataType)
    #         raw_plaintexts[j, :] = np.array(t._data[:data_space], dtype="uint8")
    #         raw_ciphertexts[j, :] = np.array(t._data[data_space:data_space*2], dtype="uint8")
    #         raw_key[j, :] = np.array(t._data[data_space*2:], dtype="uint8")
    #         j = j+1
    #
    count = 3000
    raw_traces = raw_traces[:count, ]
    raw_plaintexts = raw_plaintexts[:count, ]
    raw_ciphertexts = raw_ciphertexts[:count, ]
    raw_key = raw_key[:count, ]
    #
    raw_traces = add_gaussian_noise(raw_traces, samplesDataType)

    # plt.plot(raw_traces[0])
    # plt.show()
    # round 2 cpa_attacks poi - 37400 - 38400
    # round 4 - 77500 - 80000
    # round 3 cpa_attacks poi - 58400 - 59100
    # profiling_traces = raw_traces[1::2, 77500:80000]
    # profiling_plaintexts = raw_plaintexts[1::2]
    # profiling_key = raw_key[1::2]
    # attack_traces = raw_traces[0::2, 77500:80000]
    # attack_plaintexts = raw_plaintexts[0::2]
    # np.savez("filtered-gaussian-rnd2-traces_37000-38000-new-2.npz", raw_traces=raw_traces[:2000, 37000:38000], raw_plaintexts=raw_plaintexts[:2000], raw_key=raw_key[:2000])
    np.savez("filtered_misaligned_gaussian_traces-rnd3-58400-59100.npz", raw_traces=raw_traces[:2000, 58400:59100],
             raw_plaintexts=raw_plaintexts[:2000], raw_key=raw_key[:2000])