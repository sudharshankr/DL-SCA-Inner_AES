import trsfile
import matplotlib.pyplot as plt
import numpy as np
from make_traces import LabelledTraces
import struct
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


if __name__ == '__main__':
    ts = trs.TraceSet()
    filename = "../data/traces/SequenceAcquisition_SW_AES_ENC_3kx16"
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

    # print(raw_traces[0][0:20])
    # print(raw_key[0])
    print("Preparing the traces for training")
    traces = LabelledTraces(byte_attacked=0, leakage=3, filename=None, raw_traces=raw_traces,
                            raw_plaintext=raw_plaintexts, raw_key=raw_key)
    traces.prepare_traces_labels(0, 2000, 2000, 2500, 2500, 3000, 57750, 78670)

    print()

    # print("Saving file")
    # output_file = 'op_traces.npz'
    # np.savez(output_file, traces=traces, data=data)

    # plt.plot(t1)
    # plt.show()
    # 3rd round traces
    # 57750 - 78670 - total of 19120 features
