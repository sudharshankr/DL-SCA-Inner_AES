from funcs import hamming_lookup, aes_sbox, galois_mult_np, hamming


def leakage_model_round_1(plaintext_byte, key_byte):
    return hamming(aes_sbox[plaintext_byte ^ key_byte])


def leakage_model_round_3(plaintext_byte, key_guesses):# key_byte, gamma=0, delta=0):
    m1 = m2 = 2
    # S(m1 * S(m2 * S(p0 ^ k0) xor delta) xor gamma)
    hw = hamming_lookup[aes_sbox[
                            galois_mult_np(
                                aes_sbox[
                                    galois_mult_np(
                                        aes_sbox[
                                            plaintext_byte ^ key_guesses[:, 0]],
                                        m2)
                                    ^ key_guesses[:, 1]], m1)
                            ^ key_guesses[:, 2]]]

    return hw
