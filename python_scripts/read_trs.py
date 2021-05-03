import matplotlib.pyplot as plt
import numpy as np

from make_traces import LabelledTraces
import Trace as trs


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


if __name__ == '__main__':
    ts = trs.TraceSet()
    filename = "../data/traces/raw_traces/SequenceAcquisition_SW_AES_ENC_3kx16"
    key = [b for b in b'\xca\xfe\xba\xbe\xde\xad\xbe\xef\x00\x01\x02\x03\x04\x05\x06\x07']
    ts.open(filename + ".trs")
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

    print("Preparing the traces for training")
    traces = LabelledTraces(byte_attacked=0, leakage=3, filename=None, raw_traces=raw_traces,
                            raw_plaintext=raw_plaintexts, raw_key=raw_key)
    traces.prepare_traces_labels(0, 2000, 2000, 2500, 2500, 3000, 58000, 60960)
    traces.write_to_file("../data/traces/round_3_traces.h5")
    print()
    # plot_trace(raw_traces[0])

    # traces1 = LabelledTraces(2,1, "../data/traces/raw_traces/ATMega8515_raw_traces.h5")
    # plot_trace(traces1.raw_traces[0])

    # 3rd round traces
    # 57750 - 78670 - total of 19120 features
    # 58000-60960
