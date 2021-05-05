import h5py
import numpy as np
import copy

from funcs import galois_mult_np, calc_round_key_byte, hamming_lookup, v_hamming, aes_sbox

"""## Generate Label for Traces

### Prepare ASCAD Code and Database
"""

# download ASCAD Database : ATMEGA8515 Masking Fixed Key


class LabelledTraces:
    def __init__(self, byte_attacked: int, leakage_round: int, hypothesis_round: int, filename: str = None,
                 raw_traces: np.array = None, raw_plaintext: np.array = None, raw_key: np.array = None):
        self.target_byte = byte_attacked
        self.hypothesis_round = hypothesis_round
        self.leakage_round = leakage_round
        self.hypothesis_type = "hw"
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
        self.profiling_index = []
        self.validation_index = []
        self.attack_index = []
        self.poi = []

    # extend for different leakage models, currently implemented for hamming weight
    def labelize(self, plaintexts: np.array, keys:np.array) -> np.array:
        """
        Labels the traces according to the target byte hypothesis (target round leakage)
        :param plaintexts: the plaintexts
        :param keys: the keys
        :return: the hypothesis for all the selected traces
        """
        if self.hypothesis_round == 1:
            return hamming_lookup[aes_sbox[plaintexts[:, self.target_byte] ^ keys[:, self.target_byte]]]
        elif self.hypothesis_round == 3:
            constant_byte = 0
            delta = constant_byte ^ calc_round_key_byte(1, 0, keys)
            gamma = constant_byte ^ calc_round_key_byte(2, 0, keys)
            m1 = m2 = 2
            # S(m1 * S(m2 * S(p0 ^ k0) xor delta) xor gamma)
            hypothesis = hamming_lookup[aes_sbox[
                                         galois_mult_np(
                                             aes_sbox[
                                                 galois_mult_np(
                                                     aes_sbox[
                                                         plaintexts[:, self.target_byte] ^ keys[:, self.target_byte]],
                                                     m2)
                                                ^ delta], m1)
                                         ^ gamma]]
            return hypothesis

    def prepare_traces_labels(self, profiling_start=0, profiling_end=50000, validation_start=50000,
                              validation_end=55000,
                              attack_start=55000, attack_end=60000, poi_start=45400, poi_end=46100):
        """
        Pre-processing of traces. Extract the required traces and label them from the raw traces.
        All the indices are relative to the absolute traces.
        :param profiling_start: Start index for profiling
        :param profiling_end: End index for profiling
        :param validation_start: Start index for validation
        :param validation_end: End index for validation
        :param attack_start: Start index for attack
        :param attack_end: End index for attack
        :param poi_start: Points of interest start
        :param poi_end: Points of interest end
        """
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
        """
        Write the processed traces - Profiling, Validation and Attack traces - to a file
        Output file name - <leakage_round>-<hypothesis_round>-<hypothesis_type>.h5
        example : leakage_rnd_3-hyp_rnd_1-hw.h5
        """
        # filename = "leakage_rnd_" + str(self.leakage_round)+"-hyp_rnd_" + \
        #            str(self.hypothesis_round) + "-" + self.hypothesis_type + ".h5"
        out_file = h5py.File(filename, "w")     # make output to h5 file

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


# some code to check if everything is working fine
# if __name__ == "__main__":
#     traces = LabelledTraces(2,1, "../data/traces/ATMega8515_raw_traces.h5")
#     plt.plot(traces.raw_traces[0])
#     plt.show()
    # traces.prepare_traces_labels()
#     # traces.write_to_file("../data/traces/ASCAD_stored_traces.h5")
#     print(galois_mult(0, 2))
