
import numpy as np


def prepare_24():
    key_guesses = np.array([n for n in range(256 * 256 * 256)])
    guesses_range = np.zeros((key_guesses.shape[0], 3))
    print("Preparing guesses...")
    for i in range(key_guesses.shape[0]):
        f = '{0:024b}'.format(key_guesses[i])
        guesses_range[i][0] = int(f[:8], 2)
        guesses_range[i][1] = int(f[8:16], 2)
        guesses_range[i][2] = int(f[16:24], 2)
    guesses_range = guesses_range.astype("uint8")
    np.save("../Guesses_range_24.npy", guesses_range)

def prepare_16():
    key_guesses = np.array([n for n in range(256 * 256)])
    guesses_range = np.zeros((key_guesses.shape[0], 2))
    print("Preparing guesses...")
    for i in range(key_guesses.shape[0]):
        f = '{0:016b}'.format(key_guesses[i])
        guesses_range[i][0] = int(f[:8], 2)
        guesses_range[i][1] = int(f[8:16], 2)
    guesses_range = guesses_range.astype("uint8")
    np.save("../Guesses_range_16.npy", guesses_range)


if __name__ == "__main__":
    prepare_24()