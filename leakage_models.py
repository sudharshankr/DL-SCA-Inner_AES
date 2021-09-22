from funcs import hamming_lookup, aes_sbox, galois_mult_np, hamming


def calc_hypothesis_round_1(plaintext_byte, key_byte):
    """
    Calculate hypothesis matrix for round 1
    """
    return hamming_lookup[aes_sbox[plaintext_byte ^ key_byte[:,0]]]


def calc_hypothesis_round_2(plaintext_byte, key_guesses):
    """
    Calculate hypothesis matrix for round 2
    """
    m1 = 2
    # S(m2 * S(p0 ^ k0) xor delta)
    hw = hamming_lookup[aes_sbox[
                galois_mult_np(
                    aes_sbox[
                        plaintext_byte ^ key_guesses[:, 0]],
                    m1)
                ^ key_guesses[:, 1]]]

    return hw


def calc_hypothesis_round_3(plaintext_byte, key_guesses):# key_byte, gamma=0, delta=0):
    """
    Calculate hypothesis matrix for round 3
    """
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


def calc_hypothesis_round_3_batch(plaintexts, key_guesses, batch_size):
    """
    Compute hypothesis matrix for round 3.
    @param batch_size: Batch size of the intermediate hypothesis matrix
    """
    m1 = m2 = 2
    # S(m1 * S(m2 * S(p0 ^ k0) xor delta) xor gamma)
    hw = hamming_lookup[aes_sbox[
        galois_mult_np(
            aes_sbox[
                galois_mult_np(
                    aes_sbox[
                        plaintexts.reshape(batch_size, 1) ^ key_guesses[:, 0]],
                    m2)
                ^ key_guesses[:, 1]], m1)
        ^ key_guesses[:, 2]]]

    return hw

def calc_hypothesis_round_4(plaintext_byte, key_guesses, real_theta):
    """
    Calculate hypothesis matrix for round 4
    """
    m1 = m2 = m3 = 2
    # S(m1 * S(m2 * S(p0 ^ k0) xor delta) xor gamma)
    hw = hamming_lookup[aes_sbox[galois_mult_np(
        aes_sbox[
            galois_mult_np(
                aes_sbox[
                    galois_mult_np(
                        aes_sbox[
                            plaintext_byte ^ key_guesses[:, 0]],
                        m2)
                    ^ key_guesses[:, 1]], m1)
            ^ key_guesses[:, 2]], m3)
        ^ real_theta]]

    return hw
