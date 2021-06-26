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

    plt.plot(raw_traces[0])
    plt.show()
    #77500 - 80000
    profiling_traces = raw_traces[1::2, 77500:80000]
    profiling_plaintexts = raw_plaintexts[1::2]
    profiling_key = raw_key[1::2]
    attack_traces = raw_traces[0::2, 77500:80000]
    attack_plaintexts = raw_plaintexts[0::2]
    np.savez("rnd4-traces_77500-80000.npz", raw_traces=raw_traces[:2000, 58400:59100], raw_plaintexts=raw_plaintexts[:2000], raw_key=raw_key[:2000])
