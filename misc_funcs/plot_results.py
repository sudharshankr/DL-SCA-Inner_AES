import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
import Trace as trs
import configparser
import matplotlib.gridspec as gridspec

font = {'family': 'normal',
        'size': 11}


def read_trs_file(in_file, count_ret_traces, poi_start, poi_end):
    ts = trs.TraceSet()
    ts.open(in_file)
    samplesDataType = "int8"
    print("Preallocating arrays")
    data_space = int(ts._dataSpace / 2)
    raw_traces = np.empty(shape=(ts._numberOfTraces, ts._numberOfSamplesPerTrace), dtype=samplesDataType)
    print("Populating arrays")

    for i in range(count_ret_traces):
        t = ts.getTrace(i)
        raw_traces[i, :] = np.array(t._samples, dtype=samplesDataType)

    return raw_traces[:, poi_start:poi_end]


def plot_graph(dl_ranks, dl_traces_counts, cpa_ranks, cpa_traces_counts, filename):
    plt.rc('font', **font)
    plt.figure(figsize=(10, 6))
    plt.title('Rank vs Traces Number')
    plt.xlabel('Number of traces')
    plt.ylabel('Rank')
    plt.grid(True)
    plt.plot(dl_traces_counts, dl_ranks)
    # plt.plot(cpa_traces_counts, cpa_ranks)
    # plt.legend(["DL-SCA", "CPA"])
    plt.ticklabel_format(useOffset=False, style='plain')

    # location for the zoomed portion
    # sub_axes = plt.axes([.75, .4, .18, .16])
    #
    # # plot the zoomed portion
    # sub_axes.plot(dl_traces_counts[400:500], dl_ranks[400:500])
    # # sub_axes.set_xticks([2, 4, 6, 8, 10])
    # sub_axes.plot(cpa_traces_counts[23:26], cpa_ranks[23:26])
    # sub_axes.ticklabel_format(useOffset=False, style='plain')
    # sub_axes.set_yticks([0, 500000, 700000, 1000000])
    # # sub_axes.set_yticks([100, 300, 500, 700, 900, 1100])
    # # sub_axes.legend(["DL-SCA", "CPA"])
    # # sub_axes.set_title("Zoomed-in")
    #
    # sub_axes_dl = plt.axes([.75, .64, .18, .12])
    #
    # # plot the zoomed portion
    # sub_axes_dl.plot(dl_ranks[0:10])
    # sub_axes_dl.set_xticks([2, 4, 6, 8, 10])
    # sub_axes_dl.set_title("Zoomed-in look")
    # sub_axes.plot(cpa_traces_counts[23:26], cpa_ranks[23:26])
    # sub_axes.ticklabel_format(useOffset=False, style='plain')
    # sub_axes.set_yticks([0, 500000, 700000, 1000000])
    # plt.setp(sub_axes)

    # plt.show()
    plt.tight_layout()
    plt.savefig(filename, pad_inches=0, bbox_inches='tight', dpi=300)


def plot_traces(trace_01, trace_02, trace_11, trace_12, trace_21, trace_22, trace_31, trace_32):
    plt.rc('font', **font)
    fig = plt.figure(figsize=(10, 6))
    # plt.title('Rank vs Traces Number')
    # plt.xlabel('Number of traces')
    # plt.ylabel('Rank')
    # plt.grid(True)
    # plt.plot(trace_1)
    # plt.plot(trace_2)
    # plt.legend(["Misaligned Traces", "Aligned Traces"])
    # plt.show()
    xrange = [n for n in range(57000, 61000)]
    gs = gridspec.GridSpec(3, 1)
    # fig.legend(["Misaligned Traces", "Aligned Traces"])
    # first plot
    ax = fig.add_subplot(gs[0])
    l1, l2 = ax.plot(xrange, trace_01, '-', xrange, trace_02, '-')
    # l2 = ax.plot(xrange, trace_02)
    ax.set_ylabel(r'Trace 1', size=14)
    # plt.legend(["Misaligned Traces", "Aligned Traces"])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        labelbottom='off')  # labels along the bottom edge are off

    # second plot
    ax = fig.add_subplot(gs[1], sharex=ax)
    ax.plot(xrange, trace_11)
    ax.plot(xrange, trace_12)
    ax.set_ylabel(r'Trace 2', size=14)
    # plt.legend(["Misaligned Traces", "Aligned Traces"])
    # plt.xlim([57000, 61000])

    ax = fig.add_subplot(gs[2], sharex=ax)
    ax.plot(xrange, trace_21)
    ax.plot(xrange, trace_22)
    ax.set_ylabel(r'Trace 3', size=14)
    # plt.legend(["Misaligned Traces", "Aligned Traces"])

    # ax = fig.add_subplot(gs[3], sharex=ax)
    # ax.plot(xrange, trace_31)
    # ax.plot(xrange, trace_32)
    # ax.set_ylabel(r'Trace 4', size=14)
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    fig.legend((l1, l2), ('Misaligned', 'Aligned'), 'upper right')
    plt.show()


# Plotting results for round 3 dl and cpa_attacks
dl_results = np.load(
    "../../data/attack_results/sanity checks/sanity_check_results/incorr_keys-results-leakage_rnd_3-hypothesis_rnd_3-hw-0.npz")
cpa_results = np.load(
    "../../data/attack_results/round_2_results/results_wo_gaussian_noise/CPA/cpa-new-results-leakage_rnd_2-hypothesis_rnd_2-hw-0.npz")

dl_ranks = dl_results["ranks"]
dl_traces_cnt = dl_results["trace_cnt"]

cpa_ranks = cpa_results["ranks"]
cpa_traces_cnt = cpa_results["trace_cnt"]

plot_graph(dl_ranks, dl_traces_cnt, cpa_ranks, cpa_traces_cnt, '../../../result_images/new_images/wrong-keys.png')

# Plotting traces for difference between aligned and misaligned traces
# traces_misaligned = read_trs_file("../data/traces/raw_traces/SequenceAcquisition_SW_AES_ENC_3kx16.trs", 5, 57000, 61000)
# traces_aligned = read_trs_file("../data/traces/raw_traces/SequenceAcquisition_SW_AES_ENC_3kx16_StaticAlign.trs", 5, 57000, 61000)
#
# plot_traces(traces_misaligned[1], traces_aligned[1], traces_misaligned[2], traces_aligned[2], traces_misaligned[3], traces_aligned[3], traces_misaligned[4], traces_aligned[4])


# Plotting results for round 2 dl and cpa_attacks
# dl_npzfile = np.load("../data/attack_results/variable-profiling-results-leakage_rnd_2-hypothesis_rnd_2-hw-0.npz")
# dl_ranks = dl_npzfile["ranks"]
# dl_trace_cnt = dl_npzfile["trace_cnt"]
#
# cpa_npzfile = np.load("../data/attack_results/cpa_attacks-new-results-leakage_rnd_2-hypothesis_rnd_2-hw-0.npz")
# cpa_ranks = cpa_npzfile["ranks"]
# cpa_trace_cnt = cpa_npzfile["trace_cnt"]
#
#
# plot_graph(dl_ranks, dl_trace_cnt, cpa_ranks, cpa_trace_cnt)

# np1 = np.load("validation_file.npy")
# np2 = np.load("validation_file_1.npy")
# ranks = [45, 37, 47, 42, 18, 15]
# traces = [i for i in range(10,100,5)]
# l = len(traces) - len(ranks)
