from funcs import *


def calc_terms(plaintexts, keys, idx=(0, 0, 0, 0), galois_multipliers=(2, 3, 1, 1), xor_key=(0, 0)):
    # xor_key = (rnd, byte_idx)
    A = aes_sbox[galois_mult_np(aes_sbox[plaintexts[:, idx[0]] ^ keys[:, idx[0]]], galois_multipliers[0])
                 ^ galois_mult_np(aes_sbox[plaintexts[:, idx[1]] ^ keys[:, idx[1]]], galois_multipliers[1])
                 ^ galois_mult_np(aes_sbox[plaintexts[:, idx[2]] ^ keys[:, idx[2]]], galois_multipliers[2])
                 ^ galois_mult_np(aes_sbox[plaintexts[:, idx[3]] ^ keys[:, idx[3]]], galois_multipliers[3])
                 ^ calc_round_key_byte(xor_key[0], xor_key[1], keys)]
    return A


def calc_delta(plaintexts, keys):
    delta = galois_mult_np(aes_sbox[plaintexts[:, 5] ^ keys[:, 5]], 3) \
            ^ galois_mult_np(aes_sbox[plaintexts[:, 10] ^ keys[:, 10]], 1) \
            ^ galois_mult_np(aes_sbox[plaintexts[:, 15] ^ keys[:, 15]], 1) ^ calc_round_key_byte(1, 0, keys)
    return delta


def calc_gamma(plaintexts, keys):
    gamma_1 = galois_mult_np(aes_sbox[galois_mult_np(aes_sbox[plaintexts[:, 4] ^ keys[:, 4]], 1) \
                                      ^ galois_mult_np(aes_sbox[plaintexts[:, 9] ^ keys[:, 9]], 2) \
                                      ^ galois_mult_np(aes_sbox[plaintexts[:, 14] ^ keys[:, 14]], 3) \
                                      ^ galois_mult_np(aes_sbox[plaintexts[:, 3] ^ keys[:, 3]],
                                                       1) ^ calc_round_key_byte(1, 5, keys)], 3)

    gamma_2 = galois_mult_np(aes_sbox[galois_mult_np(aes_sbox[plaintexts[:, 8] ^ keys[:, 8]], 1) \
                                      ^ galois_mult_np(aes_sbox[plaintexts[:, 13] ^ keys[:, 13]], 1) \
                                      ^ galois_mult_np(aes_sbox[plaintexts[:, 2] ^ keys[:, 2]], 2) \
                                      ^ galois_mult_np(aes_sbox[plaintexts[:, 7] ^ keys[:, 7]],
                                                       3) ^ calc_round_key_byte(1, 10, keys)], 1)

    gamma_3 = galois_mult_np(aes_sbox[galois_mult_np(aes_sbox[plaintexts[:, 12] ^ keys[:, 12]], 3) \
                                      ^ galois_mult_np(aes_sbox[plaintexts[:, 1] ^ keys[:, 1]], 1) \
                                      ^ galois_mult_np(aes_sbox[plaintexts[:, 6] ^ keys[:, 6]], 1) \
                                      ^ galois_mult_np(aes_sbox[plaintexts[:, 11] ^ keys[:, 11]],
                                                       2) ^ calc_round_key_byte(1, 15, keys)], 1)
    gamma = gamma_1 ^ gamma_2 ^ gamma_3 ^ calc_round_key_byte(2, 0, keys)
    return gamma


def calc_theta(plaintexts, keys):
    u_1 = aes_sbox[galois_mult_np(calc_terms(plaintexts, keys, idx=(4, 9, 14, 3), galois_multipliers=(2, 3, 1, 1), xor_key=(1, 4)), 1)
                   ^ galois_mult_np(calc_terms(plaintexts, keys, idx=(8, 13, 2, 7), galois_multipliers=(1, 2, 3, 1), xor_key=(1, 9)), 2)
                   ^ galois_mult_np(calc_terms(plaintexts, keys, idx=(12, 1, 6, 11), galois_multipliers=(1, 1, 2, 3), xor_key=(1, 14)), 3)
                   ^ galois_mult_np(calc_terms(plaintexts, keys, idx=(0, 5, 10, 15), galois_multipliers=(3, 1, 1, 2), xor_key=(1, 3)), 1)
                   ^ calc_round_key_byte(2, 5, keys)]
    u_2 = aes_sbox[galois_mult_np(calc_terms(plaintexts, keys, idx=(8, 13, 2, 7), galois_multipliers=(2, 3, 1, 1), xor_key=(1, 8)), 1)
                   ^ galois_mult_np(calc_terms(plaintexts, keys, idx=(12, 1, 6, 11), galois_multipliers=(1, 2, 3, 1), xor_key=(1, 13)), 1)
                   ^ galois_mult_np(calc_terms(plaintexts, keys, idx=(0, 5, 10, 15), galois_multipliers=(1, 1, 2, 3), xor_key=(1, 2)), 2)
                   ^ galois_mult_np(calc_terms(plaintexts, keys, idx=(4, 9, 14, 3), galois_multipliers=(3, 1, 1, 2), xor_key=(1, 7)), 3)
                   ^ calc_round_key_byte(2, 10, keys)]
    u_3 = aes_sbox[galois_mult_np(calc_terms(plaintexts, keys, idx=(12, 1, 6, 11), galois_multipliers=(2, 3, 1, 1), xor_key=(1, 12)), 3)
                   ^ galois_mult_np(calc_terms(plaintexts, keys, idx=(0, 5, 10, 15), galois_multipliers=(1, 2, 3, 1), xor_key=(1, 1)), 1)
                   ^ galois_mult_np(calc_terms(plaintexts, keys, idx=(4, 9, 14, 3), galois_multipliers=(1, 1, 2, 3), xor_key=(1, 6)), 1)
                   ^ galois_mult_np(calc_terms(plaintexts, keys, idx=(8, 13, 2, 7), galois_multipliers=(3, 1, 1, 2), xor_key=(1, 11)), 2)
                   ^ calc_round_key_byte(2, 15, keys)]
    theta_1 = galois_mult_np(u_1, 3)
    theta_2 = galois_mult_np(u_2, 1)
    theta_3 = galois_mult_np(u_3, 1)
    theta = theta_1 ^ theta_2 ^ theta_3 ^ calc_round_key_byte(3, 0, keys)
    return theta
