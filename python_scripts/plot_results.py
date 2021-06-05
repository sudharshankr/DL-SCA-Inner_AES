import numpy as np
import matplotlib.pyplot as plt


def plot_graph(ranks, traces_counts):
    plt.figure(figsize=(15, 6))
    plt.title('Key Rank vs Traces Number')
    plt.xlabel('Number of traces')
    plt.ylabel('Key Rank')
    plt.grid(True)
    plt.plot(traces_counts, ranks)
    plt.show()
#
#
# npzfile = np.load("../data/attack_results/results_round_3_leakage_1.npz")
# ranks = npzfile["ranks"]
# trace_cnt = npzfile["trace_cnt"]
#
# plot_graph(ranks, trace_cnt)

# np1 = np.load("validation_file.npy")
# np2 = np.load("validation_file_1.npy")
ranks = [45, 37, 47, 42, 18, 15]
traces = [i for i in range(10,100,5)]
l = len(traces) - len(ranks)
for j in range(l):
    ranks.append(0)
plot_graph(ranks, traces)
print()
