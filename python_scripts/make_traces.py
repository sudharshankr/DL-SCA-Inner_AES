import h5py
import numpy as np
import copy
from aeskeyschedule import key_schedule

"""## Generate Label for Traces

### Prepare ASCAD Code and Database
"""


# download ASCAD Database : ATMEGA8515 Masking Fixed Key

def galois_mult(a, b):
    """
    Multiplication in the Galois field GF(2^8).
    """
    p = 0
    # hi_bit_set = 0
    for i in range(8):
        if b & 1 == 1:
            p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set == 0x80:
            a ^= 0x1b
        b >>= 1
    return p % 256


v_gmul = np.vectorize(galois_mult)


def calc_round_key_byte(rnd, idx_byte, keys):
    expanded_key_bytes = np.zeros((keys.shape[0],), dtype="uint8")
    for i in range(keys.shape[0]):
        expanded_key_bytes[i] = key_schedule(keys[i])[rnd][idx_byte]
    return expanded_key_bytes


class LabelledTraces:
    def __init__(self, byte_attacked, leakage, filename: str = None, raw_traces: np.array = None,
                 raw_plaintext: np.array = None, raw_key: np.array = None):
        self.target_byte = byte_attacked
        self.leakage = leakage
        if filename is not None:
            traces_file = h5py.File(filename, "r")
            self.raw_traces = traces_file['traces']
            self.raw_plaintext = traces_file['metadata']['plaintext']
            self.raw_key = traces_file['metadata']['key']
        else:
            self.raw_traces = raw_traces
            self.raw_plaintext = raw_plaintext
            self.raw_key = raw_key
        self.profiling_traces = np.array([], self.raw_traces.dtype)
        self.validation_traces = np.array([], self.raw_traces.dtype)
        self.attack_traces = np.array([], self.raw_traces.dtype)
        self.profiling_labels = np.array([], np.uint32)
        self.validation_labels = np.array([], np.uint32)
        self.attack_labels = np.array([], np.uint32)
        self.v_hamming = np.vectorize(self.hamming)
        self.aes_sbox = np.array([])
        self.profiling_index = []
        self.validation_index = []
        self.attack_index = []
        self.poi = []
        self.initialize_sbox()

    def initialize_sbox(self):
        self.aes_sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
        ])

    def hamming(self, n):
        return bin(int(n)).count("1")

    # extend for different leakage models, currently implemented for hamming weight
    def labelize(self, plaintexts, keys):
        constant_byte = 0
        if self.leakage == 1:
            return self.v_hamming(self.aes_sbox[plaintexts[:, self.target_byte] ^ keys[:, self.target_byte]])
        elif self.leakage == 3:
            delta = constant_byte ^ calc_round_key_byte(1, 0, keys)
            gamma = constant_byte ^ calc_round_key_byte(2, 0, keys)
            m1 = m2 = 2
            # S(m1 * S(m2 * S(p0 ^ k0) xor delta) xor gamma)
            leakage = self.v_hamming(self.aes_sbox[
                v_gmul(
                    self.aes_sbox[
                        v_gmul(
                            self.aes_sbox[plaintexts[:, self.target_byte] ^ keys[:, self.target_byte]], m2)
                        ^ delta], m1)
                ^ gamma])
            return leakage

    def prepare_traces_labels(self, profiling_start=0, profiling_end=50000, validation_start=50000,
                              validation_end=55000,
                              attack_start=55000, attack_end=60000, poi_start=45400, poi_end=46100):
        self.profiling_index = [n for n in range(profiling_start, profiling_end)]
        self.validation_index = [n for n in range(validation_start, validation_end)]
        self.attack_index = [n for n in range(attack_start, attack_end)]
        self.poi = [n for n in range(poi_start, poi_end)]

        self.profiling_traces = copy.deepcopy(self.raw_traces[profiling_start:profiling_end, poi_start:poi_end])
        self.validation_traces = copy.deepcopy(self.raw_traces[validation_start:validation_end, poi_start:poi_end])
        self.attack_traces = copy.deepcopy(self.raw_traces[attack_start:attack_end, poi_start:poi_end])

        self.profiling_labels = self.labelize(self.raw_plaintext[self.profiling_index],
                                              self.raw_key[self.profiling_index])
        self.validation_labels = self.labelize(self.raw_plaintext[self.validation_index],
                                               self.raw_key[self.validation_index])
        self.attack_labels = self.labelize(self.raw_plaintext[self.attack_index], self.raw_key[self.attack_index])

    def write_to_file(self, filename):
        # make output to h5 file
        out_file = h5py.File(filename, "w")

        profiling_traces_group = out_file.create_group("Profiling_traces")
        attack_traces_group = out_file.create_group("Attack_traces")
        validation_traces_group = out_file.create_group("Validation_traces")

        profiling_traces_group.create_dataset(name="traces", data=self.profiling_traces,
                                              dtype=self.profiling_traces.dtype)
        validation_traces_group.create_dataset(name="traces", data=self.validation_traces,
                                               dtype=self.validation_traces.dtype)
        attack_traces_group.create_dataset(name="traces", data=self.attack_traces, dtype=self.attack_traces.dtype)

        profiling_traces_group.create_dataset(name="labels", data=self.profiling_labels,
                                              dtype=self.profiling_labels.dtype)
        validation_traces_group.create_dataset(name="labels", data=self.validation_labels,
                                               dtype=self.validation_labels.dtype)
        attack_traces_group.create_dataset(name="labels", data=self.attack_labels, dtype=self.attack_labels.dtype)

        metadata_type = np.dtype([
            ("plaintext", self.raw_plaintext.dtype, (len(self.raw_plaintext[0]),)),
            ("key", self.raw_key.dtype, (len(self.raw_key[0]),)),
        ])

        profiling_metadata = np.array([(self.raw_plaintext[n], self.raw_key[n]) for n in self.profiling_index],
                                      dtype=metadata_type)
        profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type)

        validation_metadata = np.array([(self.raw_plaintext[n], self.raw_key[n]) for n in self.validation_index],
                                       dtype=metadata_type)
        validation_traces_group.create_dataset("metadata", data=validation_metadata, dtype=metadata_type)

        attack_metadata = np.array([(self.raw_plaintext[n], self.raw_key[n]) for n in self.attack_index],
                                   dtype=metadata_type)
        attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type)

        out_file.flush()
        out_file.close()


if __name__ == "__main__":
    # traces = LabelledTraces(2, "../data/traces/ATMega8515_raw_traces.h5")
    # traces.prepare_traces_labels()
    # traces.write_to_file("../data/traces/ASCAD_stored_traces.h5")
    print(galois_mult(0, 2))