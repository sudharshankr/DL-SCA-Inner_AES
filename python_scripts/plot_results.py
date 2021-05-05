import numpy as np
import matplotlib.pyplot as plt


def plot_graph(ranks, traces_counts):
    plt.figure(figsize=(15, 6))
    plt.title('Rank vs Traces Number')
    plt.xlabel('Number of traces')
    plt.ylabel('Rank')
    plt.grid(True)
    plt.plot(traces_counts, ranks)
    plt.show()


npzfile = np.load("../data/attack_results/results_round_3_leakage_1.npz")
ranks = npzfile["ranks"]
trace_cnt = npzfile["trace_cnt"]

plot_graph(ranks, trace_cnt)
