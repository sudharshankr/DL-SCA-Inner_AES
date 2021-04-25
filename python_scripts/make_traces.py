# -*- coding: utf-8 -*-

import h5py
import numpy as np

"""## Generate Label for Traces

### Prepare ASCAD Code and Database
"""

# download ASCAD Database : ATMEGA8515 Masking Fixed Key


# in_file = h5py.File("/Users/sud/the_stuff/Studies/Thesis/Data/ASCAD_data/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5", "r")
in_file = h5py.File("/home/nfs/sudharshankuma/ATMega8515_raw_traces.h5", "r")

raw_traces = in_file['traces']
raw_metadata = in_file['metadata']

raw_plaintext = raw_metadata['plaintext']
raw_key = raw_metadata['key']


num_traces_profiling_start = 0
num_traces_profiling_stop = 50000
num_traces_validation_start = 50000
num_traces_validation_stop = 55000
num_traces_attack_start = 55000
num_traces_attack_stop = 60000

# selecting a specific range known to show good classification.
# Need to try this feature selection with Pearson's or PCA
point_interest_start = 45400
point_interest_stop = 46100

profiling_index = [n for n in range(num_traces_profiling_start, num_traces_profiling_stop)]
validation_index = [n for n in range(num_traces_validation_start, num_traces_validation_stop)]
attack_index = [n for n in range(num_traces_attack_start, num_traces_attack_stop)]
target_point = [n for n in range(point_interest_start, point_interest_stop)]

# make array for traces and metadatas

raw_traces_profiling = np.zeros([len(profiling_index), len(target_point)], raw_traces.dtype)
label_profiling = np.zeros(len(profiling_index), np.uint32)

raw_traces_validation = np.zeros([len(validation_index), len(target_point)], raw_traces.dtype)
label_validation = np.zeros(len(validation_index), np.uint32)

raw_traces_attack = np.zeros([len(attack_index), len(target_point)], raw_traces.dtype)
label_attack = np.zeros(len(attack_index), np.uint32)

idx_traces = 0
for traces in profiling_index:
    np.copyto(raw_traces_profiling[idx_traces], raw_traces[traces, point_interest_start:point_interest_stop])
    idx_traces += 1

idx_traces = 0
for traces in attack_index:
    np.copyto(raw_traces_attack[idx_traces], raw_traces[traces, point_interest_start:point_interest_stop])
    idx_traces += 1

idx_traces = 0
for traces in validation_index:
    np.copyto(raw_traces_validation[idx_traces], raw_traces[traces, point_interest_start:point_interest_stop])
    idx_traces += 1

AES_Sbox = np.array([
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


# make labelize function
def labelize(plaintexts, keys, byte_attacked):
    return AES_Sbox[plaintexts[:, byte_attacked] ^ keys[:, byte_attacked]]


# byte attacked
byte_attacked = 2

profiling_labels = labelize(raw_plaintext[profiling_index], raw_key[profiling_index], byte_attacked)
validation_labels = labelize(raw_plaintext[validation_index], raw_key[validation_index], byte_attacked)
attack_labels = labelize(raw_plaintext[attack_index], raw_key[attack_index], byte_attacked)

# make output to h5 file

out_file = h5py.File("ASCAD_stored_traces.h5", "w")

profiling_traces_group = out_file.create_group("Profiling_traces")
attack_traces_group = out_file.create_group("Attack_traces")
validation_traces_group = out_file.create_group("Validation_traces")

profiling_traces_group.create_dataset(name="traces", data=raw_traces_profiling, dtype=raw_traces_profiling.dtype)
validation_traces_group.create_dataset(name="traces", data=raw_traces_validation, dtype=raw_traces_validation.dtype)
attack_traces_group.create_dataset(name="traces", data=raw_traces_attack, dtype=raw_traces_attack.dtype)


profiling_traces_group.create_dataset(name="labels", data=profiling_labels, dtype=profiling_labels.dtype)
validation_traces_group.create_dataset(name="labels", data=validation_labels, dtype=validation_labels.dtype)
attack_traces_group.create_dataset(name="labels", data=attack_labels, dtype=attack_labels.dtype)


metadata_type = np.dtype([
    ("plaintext", raw_plaintext.dtype, (len(raw_plaintext[0]),)),
    ("key", raw_key.dtype, (len(raw_key[0]),)),
])

profiling_metadata = np.array([(raw_plaintext[n], raw_key[n]) for n in profiling_index], dtype=metadata_type)
profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type)

validation_metadata = np.array([(raw_plaintext[n], raw_key[n]) for n in validation_index], dtype=metadata_type)
validation_traces_group.create_dataset("metadata", data=validation_metadata, dtype=metadata_type)

attack_metadata = np.array([(raw_plaintext[n], raw_key[n]) for n in attack_index], dtype=metadata_type)
attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type)

out_file.flush()
out_file.close()